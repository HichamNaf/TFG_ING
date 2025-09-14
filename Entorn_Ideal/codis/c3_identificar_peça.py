#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-down → DBSCAN → clúster superior → (NORMALITZACIÓ/ESCALAT) → PCA+ICP vs. SINGLES.

Novetat: abans de PCA/ICP, s'escalen i es centren els núvols perquè
la SINGLE tingui la mateixa escala que el clúster visible.

Escalat robust:
  s = median_kNN(target, k=8) / median_kNN(source, k=8)

Comentaris en català.
"""

import argparse, json, math, copy, re
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# -------------------- Càmera / Ray casting --------------------

def intr_from_fov(width: int, height: int, fov_deg: float):
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    return fx, fy, cx, cy

def lookat(eye, center, up=np.array([0,1,0.0], float)):
    eye = np.asarray(eye,float); center = np.asarray(center,float); up = np.asarray(up,float)
    f = center - eye; f /= (np.linalg.norm(f)+1e-12)
    s = np.cross(f, up); s /= (np.linalg.norm(s)+1e-12)
    u = np.cross(s, f)
    R = np.stack([s, u, f], 1)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = eye
    return T

def raycast_depth_points(mesh_legacy: o3d.geometry.TriangleMesh,
                         width: int, height: int, fov_deg: float,
                         T_wc: np.ndarray, near: float, far: float):
    fx, fy, cx, cy = intr_from_fov(width, height, fov_deg)
    R = T_wc[:3,:3]; t = T_wc[:3,3]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy))

    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)  # (H,W)

    zc = np.ones_like(uu, np.float32)
    xc = (uu - cx) / fx
    yc = (vv - cy) / fy
    dirs_cam = np.stack([xc, yc, zc], axis=-1)
    dirs_cam /= (np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-12)

    dirs_world = dirs_cam.reshape(-1,3) @ R.T
    origins = np.repeat(t.reshape(1,3), dirs_world.shape[0], axis=0)

    rays = np.hstack([origins.astype(np.float32), dirs_world.astype(np.float32)])
    ans  = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans['t_hit'].numpy().reshape(height, width)

    depth = t_hit.copy()
    valid = np.isfinite(depth) & (depth > near) & (depth < far)
    depth[~valid] = np.nan

    flat = valid.reshape(-1)
    if not np.any(flat):
        return depth, np.empty((0,3), dtype=np.float32)

    tv = t_hit.reshape(-1)[flat].reshape(-1,1)
    O  = origins[flat]
    D  = dirs_world[flat]
    pts = O + D * tv
    return depth, pts.astype(np.float32)

# -------------------- Transformacions / PCA --------------------

def center_mesh_to_floor(mesh: o3d.geometry.TriangleMesh, z_floor: float = 0.0):
    m = copy.deepcopy(mesh)
    aabb = m.get_axis_aligned_bounding_box()
    minb = aabb.get_min_bound(); maxb = aabb.get_max_bound()
    cx = 0.5*(minb[0] + maxb[0]); cy = 0.5*(minb[1] + maxb[1])
    m.translate((-cx, -cy, 0.0))
    aabb2 = m.get_axis_aligned_bounding_box()
    minz = aabb2.get_min_bound()[2]
    m.translate((0.0, 0.0, z_floor - minz))
    return m

def pca_frame(points: np.ndarray):
    C = points.mean(axis=0)
    X = points - C
    H = X.T @ X / max(1, len(points)-1)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    if np.linalg.det(vecs) < 0:
        vecs[:, -1] *= -1.0
    return C, vecs  # centre, R(3x3)

def build_T_from_RT(R, t):
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
    return T

def frame_mesh(C, R, scale=0.02):
    fr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    T = build_T_from_RT(R, C)
    return fr.transform(T)

# -------------------- Coloring --------------------

def colorize_by_labels(pts: np.ndarray, labels: np.ndarray):
    K = labels.max()+1 if labels.size else 0
    rng = np.random.default_rng(1)
    palette = rng.random((max(K,1), 3), dtype=np.float32)
    cols = np.zeros((pts.shape[0],3), np.float32)
    for i, lab in enumerate(labels):
        if lab >= 0:
            cols[i] = palette[lab % K]
        else:
            cols[i] = np.array([0.6,0.6,0.6], np.float32)
    return cols

# -------------------- Normalització / Escalat --------------------

def median_knn_scale(points: np.ndarray, k: int = 8) -> float:
    """Escala robusta: mediana de la distància al k-è veí (k>=2)."""
    if points.shape[0] < k+1:
        # fallback: diagonal AABB
        mn, mx = points.min(0), points.max(0)
        diag = np.linalg.norm(mx - mn) + 1e-9
        return diag / 20.0  # heurística suau
    tree = cKDTree(points)
    d, _ = tree.query(points, k=k+1)
    dk = d[:, -1]  # dist al k-è
    return float(np.median(dk) + 1e-12)

def normalize_and_scale(source_pts: np.ndarray, target_pts: np.ndarray, k:int=8):
    """
    Calcula factor s perquè source tingui la mateixa escala robusta que target.
      s = med_kNN(target)/med_kNN(source)
    Retorna source_escalat, s
    """
    s_src = median_knn_scale(source_pts, k=k)
    s_tgt = median_knn_scale(target_pts, k=k)
    s = s_tgt / s_src
    return (source_pts * s), s

# -------------------- ICP helpers --------------------

def to_pcd(points: np.ndarray):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(np.float32)))

def prepare_for_icp(src_pts: np.ndarray, tgt_pts: np.ndarray, voxel: float, normal_rad: float):
    src = to_pcd(src_pts).voxel_down_sample(voxel)
    tgt = to_pcd(tgt_pts).voxel_down_sample(voxel)
    if not src.has_normals():
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=50))
    if not tgt.has_normals():
        tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=50))
    return src, tgt

# -------------------- MAIN --------------------

def main():
    ap = argparse.ArgumentParser("Top-down + DBSCAN + (escala/normalitza) + PCA+ICP vs. SINGLES")
    ap.add_argument("--scene_stl", required=True, help="STL del conjunt (metres)")
    ap.add_argument("--tuned_json", required=True, help="tuned_info.json (eps/minPts per distància)")
    ap.add_argument("--singles_root", required=True, help="Arrel 'singles/' (dist_*.m/points3d/*.npy)")
    ap.add_argument("--height_m", type=float, required=True, help="Alçada càmera (m)")
    ap.add_argument("--z_floor", type=float, default=0.0, help="Cota z del terra (m)")

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov_deg", type=float, default=60.0)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=3.0)

    ap.add_argument("--icp_thresh", type=float, default=0.02, help="Radi màxim correspondències ICP (m)")
    ap.add_argument("--max_views", type=int, default=120, help="#vistes SINGLE a provar (0=totes)")
    ap.add_argument("--voxel_factor", type=float, default=1.0, help="voxel≈factor*eps per a ICP")
    ap.add_argument("--k_scale", type=int, default=8, help="k per a med_kNN (escala robusta)")
    args = ap.parse_args()

    # 1) Conjunt centrat al terra
    scene_mesh_in = o3d.io.read_triangle_mesh(args.scene_stl); scene_mesh_in.compute_vertex_normals()
    if scene_mesh_in.is_empty():
        raise RuntimeError("No s'ha pogut llegir --scene_stl")
    scene_mesh = center_mesh_to_floor(scene_mesh_in, args.z_floor)

    # 2) Càmera top-down
    aabb = scene_mesh.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    target = np.array([0.0, 0.0, args.z_floor + 0.5*ext[2]], dtype=float)
    eye    = np.array([0.0, 0.0, args.height_m], dtype=float)
    T_wc   = lookat(eye, target, up=np.array([0,1,0], float))

    # 3) Ray cast
    depth, pts = raycast_depth_points(scene_mesh, args.width, args.height, args.fov_deg,
                                      T_wc, args.near, args.far)
    if pts.shape[0] == 0:
        raise RuntimeError("Sense punts visibles (revisa near/far/FOV/alçada).")

    d_min = float(np.nanmin(depth[np.isfinite(depth)]))
    print(f"→ Distància estimada al punt més alt: {d_min:.4f} m")

    # 4) eps/minPts per distància
    tuned = json.loads(Path(args.tuned_json).read_text())
    def closest_key(d):
        best_k, best_diff = None, 1e9
        for k in tuned.keys():
            try:
                val = float(k)
            except Exception:
                continue
            diff = abs(val - d)
            if diff < best_diff:
                best_k, best_diff = k, diff
        return best_k
    k_near = closest_key(d_min)
    if k_near is None:
        raise RuntimeError("No hi ha distància compatible al tuned_info.json.")
    eps = float(tuned[k_near]["eps"])
    min_pts = int(tuned[k_near]["min_points"])
    print(f"→ Distància bin={k_near} m  → eps={eps:.6f}  min_points={min_pts}")

    # 5) DBSCAN
    pcd_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.array(pcd_scene.cluster_dbscan(eps=eps, min_points=min_pts))
    K = labels.max()+1 if labels.size and labels.max()>=0 else 0
    print(f"→ DBSCAN: {K} clústers (+ soroll)")
    if K == 0:
        raise RuntimeError("No s'han trobat clústers. Revisa eps/min_points o dades.")

    # clúster més alt
    z_vals = np.asarray(pts)[:,2]
    best_id, best_z = None, -1e9
    clusters_idx = []
    for k in range(K):
        idx = np.where(labels == k)[0]
        clusters_idx.append(idx)
        if idx.size == 0: continue
        zmax = float(np.max(z_vals[idx]))
        if zmax > best_z:
            best_z = zmax; best_id = k
    idx_top = clusters_idx[best_id]
    pts_top = pts[idx_top]
    print(f"→ Clúster superior: id={best_id}  punts={len(idx_top)}  z_max={best_z:.4f} m")

    # PCA del clúster top
    C_top, R_top = pca_frame(pts_top)
    frame_top = frame_mesh(C_top, R_top, scale=0.03)

    # 6) Carrega vistes SINGLE de la distància escollida
    singles_root = Path(args.singles_root)
    dist_dir = singles_root / f"dist_{float(k_near):.2f}m" / "points3d"
    if not dist_dir.exists():
        candidates = list((singles_root).glob("dist_*m/points3d"))
        def parse_val(p):
            m = re.search(r"dist_([0-9.]+)m", str(p))
            return float(m.group(1)) if m else 1e9
        if candidates:
            dist_dir = min(candidates, key=lambda p: abs(parse_val(p) - float(k_near)))
    view_files = sorted(dist_dir.glob("*.npy"))
    if not view_files:
        raise RuntimeError(f"No s'han trobat SINGLE npy a {dist_dir}")
    if args.max_views > 0:
        view_files = view_files[:args.max_views]
    print(f"→ Provaré {len(view_files)} vistes SINGLE de {dist_dir.parent.name}")

    # target per ICP
    voxel = max(1e-4, args.voxel_factor * eps)  # voxel ~ eps
    normal_rad = max(0.01, 2.0 * eps)

    pcd_top = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_top))
    pcd_top.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=60))
    pcd_top_ds = pcd_top.voxel_down_sample(voxel)

    # 7) Prova vistes: ESCALAR→PCA init→ICP
    best = None  # (fitness, -rmse, file, T_best)
    for vf in view_files:
        src_pts_raw = np.load(vf).astype(np.float32)
        if src_pts_raw.ndim != 2 or src_pts_raw.shape[1] != 3 or src_pts_raw.shape[0] < 20:
            continue

        # (a) Escalat robust perquè la SINGLE tingui mateixa escala que el clúster
        src_scaled, s_iso = normalize_and_scale(src_pts_raw, pts_top, k=args.k_scale)

        # (b) PCA init (sobre dades ja escalades)
        C_s, R_s = pca_frame(src_scaled)
        C_t, R_t = C_top, R_top

        R0 = R_t @ R_s.T
        t0 = C_t - (R0 @ C_s)
        T0 = build_T_from_RT(R0, t0)

        # (c) ICP point-to-plane (sobre downsample i normals)
        src_ds = o3d.geometry.PointCloud(o3d.utility.Vector3dVector((src_scaled @ R0.T) + t0))
        src_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_rad, max_nn=60))
        src_ds = src_ds.voxel_down_sample(voxel)

        icp = o3d.pipelines.registration.registration_icp(
            source=src_ds, target=pcd_top_ds,
            max_correspondence_distance=args.icp_thresh,
            init=np.eye(4),
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        fitness = icp.fitness
        rmse = icp.inlier_rmse
        T_full = icp.transformation @ T0

        key = (fitness, -rmse)
        if (best is None) or (key > (best[0], best[1])):
            best = (fitness, -rmse, vf.name, T_full, s_iso)

    if best is None:
        raise RuntimeError("Cap vista SINGLE s'ha pogut alinear (prova pujar --icp_thresh o --max_views).")

    best_fitness, best_rmse_neg, best_name, T_best, s_used = best
    print(f"→ Millor vista: {best_name}  fitness={best_fitness:.3f}  rmse={-best_rmse_neg:.6f}  (escala s={s_used:.6f})")
    print("Transformació final (T_best 4x4):\n", np.array_str(T_best, precision=6, suppress_small=True))

    # 8) Visualització
    cols_scene = colorize_by_labels(pts, labels)
    pcd_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd_scene.colors = o3d.utility.Vector3dVector(cols_scene)

    best_src = np.load(dist_dir / best_name).astype(np.float32)
    best_src_scaled = best_src * s_used
    pcd_best = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(best_src_scaled))
    pcd_best.transform(T_best)
    pcd_best.paint_uniform_color([0.0, 0.2, 1.0])  # blau

    pcd_top.paint_uniform_color([1.0, 0.0, 0.0])   # vermell
    frame_top = frame_mesh(C_top, R_top, scale=0.03)

    o3d.visualization.draw_geometries(
        [pcd_scene, pcd_best, frame_top],
        window_name="Clúster superior + millor SINGLE (escala + PCA + ICP)"
    )

if __name__ == "__main__":
    main()
