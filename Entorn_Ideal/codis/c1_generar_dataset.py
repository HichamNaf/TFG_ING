#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset generator (simple):

Modes de funcionament
---------------------
1) AUTOMÀTIC (--auto):
   Només cal especificar --piece_stl i --out_dir.
   Usa valors per defecte sensats:
     - Distàncies: 0.30, 0.40, 0.50, 0.60 (m)
     - Singles: az=0..360 step 30, el=5..85 step 10
     - Scenes: 120 escenes/dist, 3..18 peces, caixa 0.25x0.25 m
     - Càmera: 640x480, FOV=60°, near=0.05, far=2.0
     - Rotacions 3D uniformes per a SCENES
    exemple: 
        python generar_dataset.py --piece_stl model.stl --out_dir dataset_anyangle --auto


2) PARAMETRITZABLE (sense --auto): 
    python generar_dataset.py --piece_stl model.stl --out_dir dataset_anyangle 
    --dists 0.35 0.45 0.55 
    --scenes_per_dist 150 --min_objs 4 --max_objs 20 
    --az_step 20 --el_start 10 --el_stop 80 --el_step 10 
    --bin_w 0.30 --bin_d 0.30 --gap_factor 1.08 
    --width 800 --height 600 --fov_deg 65 --seed 42

Estructura de sortida
---------------------
out_dir/
    singles/
        dist_0.30m/
            points3d/       (Numpy .npy, punts 3D visibles)
            depth_npy/      (Numpy .npy, profunditat amb NaN)
            mask_npy/       (Numpy .npy, màscara booleana)
            depth_png/      (PNG 8-bit, profunditat normalitzada)
            mask_png/       (PNG 8-bit, màscara binària)
            meta/           (JSON, metadades càmera, peça, punts)
        dist_0.40m/
        ...
        dist_0.60m/
    scenes/
        dist_0.30m/
            points3d/       (Numpy .npy, punts 3D visibles)
            depth_npy/      (Numpy .npy, profunditat amb NaN)
            mask_npy/       (Numpy .npy, màscara booleana)
            depth_png/      (PNG 8-bit, profunditat normalitzada)
            mask_png/       (PNG 8-bit, màscara binària)
            meta/           (JSON, metadades càmera, peça, punts, poses)
            gt_counts.csv   (CSV, nom_escena,num_obj)
        dist_0.40m/
        ...
        dist_0.60m/

Ray casting amb Open3D (headless).
"""

import argparse, json, math, random, copy
from pathlib import Path
import numpy as np
import open3d as o3d

# PNG writers
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None
    from PIL import Image

# -------------------- Càmera / geom --------------------

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

def spherical_cam_Twc(center, dist_m, az_deg, el_deg):
    az = math.radians(az_deg); el = math.radians(el_deg)
    x = center[0] + dist_m * math.cos(el) * math.cos(az)
    y = center[1] + dist_m * math.cos(el) * math.sin(az)
    z = center[2] + dist_m * math.sin(el)
    eye = np.array([x, y, z], float)
    up = np.array([0, 0, 1.0], float)
    return lookat(eye, center, up=up)

def raycast_depth_mask_points(mesh_legacy: o3d.geometry.TriangleMesh,
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
    mask = np.isfinite(depth) & (depth > near) & (depth < far)
    depth[~mask] = np.nan

    flat = mask.reshape(-1)
    if not np.any(flat):
        return depth, mask, np.empty((0,3), dtype=np.float32)

    tv = t_hit.reshape(-1)[flat].reshape(-1,1)
    O  = origins[flat]
    D  = dirs_world[flat]
    pts = O + D * tv
    return depth, mask, pts.astype(np.float32)

# -------------------- Peça i col·locació --------------------

def center_mesh_to_bottom(mesh: o3d.geometry.TriangleMesh):
    m = copy.deepcopy(mesh)
    aabb = m.get_axis_aligned_bounding_box()
    minb = aabb.get_min_bound(); maxb = aabb.get_max_bound()
    cx = 0.5*(minb[0]+maxb[0]); cy = 0.5*(minb[1]+maxb[1]); z0 = minb[2]
    m.translate((-cx, -cy, -z0))
    return m, m.get_axis_aligned_bounding_box()

def footprint_radius_xy_from_aabb(aabb: o3d.geometry.AxisAlignedBoundingBox):
    ext = aabb.get_extent()
    w, d, h = float(ext[0]), float(ext[1]), float(ext[2])
    # Radi conservador (circum-esfera) per evitar solapament sota rotació
    return 0.5 * math.sqrt(w*w + d*d + h*h)

def sample_non_overlapping_xy(n, bin_w, bin_d, rad, gap_factor, max_tries=4000):
    centers = []
    need = gap_factor * (2.0 * rad)
    x_min, x_max = -bin_w/2 + rad, bin_w/2 - rad
    y_min, y_max = -bin_d/2 + rad, bin_d/2 - rad
    if x_min >= x_max or y_min >= y_max:
        return centers
    tries = 0
    while len(centers) < n and tries < max_tries:
        tries += 1
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        ok = True
        for (px,py) in centers:
            if (x-px)**2 + (y-py)**2 < (need**2):
                ok = False; break
        if ok:
            centers.append((x,y))
    return centers

def random_rotation_matrix():
    # Shoemake (uniforme a SO(3))
    u1, u2, u3 = np.random.rand(3)
    q1 = math.sqrt(1-u1) * math.sin(2*math.pi*u2)
    q2 = math.sqrt(1-u1) * math.cos(2*math.pi*u2)
    q3 = math.sqrt(u1)   * math.sin(2*math.pi*u3)
    q4 = math.sqrt(u1)   * math.cos(2*math.pi*u3)
    x, y, z, w = q1, q2, q3, q4
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1-2*(x*x+z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def rotate_and_place_on_floor(mesh0: o3d.geometry.TriangleMesh, R: np.ndarray, cx: float, cy: float, z_floor: float):
    inst = copy.deepcopy(mesh0)
    inst.rotate(R, center=(0,0,0))
    aabb = inst.get_axis_aligned_bounding_box()
    minz = float(aabb.get_min_bound()[2])
    dz = z_floor - minz
    inst.translate((cx, cy, dz))
    return inst

# -------------------- PNG helpers --------------------

def save_png_depth(depth: np.ndarray, path_png: Path):
    finite = np.isfinite(depth)
    if not np.any(finite):
        img = np.zeros(depth.shape, dtype=np.uint8)
    else:
        d = depth.copy()
        dmin, dmax = np.nanmin(d), np.nanmax(d)
        if dmax - dmin < 1e-9:
            img = np.zeros(depth.shape, dtype=np.uint8)
        else:
            norm = (d - dmin) / (dmax - dmin)
            norm = 1.0 - norm
            norm[~finite] = 0.0
            img = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    if imageio:
        imageio.imwrite(str(path_png), img)
    else:
        Image.fromarray(img).save(str(path_png))

def save_png_mask(mask: np.ndarray, path_png: Path):
    img = (mask.astype(np.uint8) * 255)
    if imageio:
        imageio.imwrite(str(path_png), img)
    else:
        Image.fromarray(img).save(str(path_png))

# -------------------- Generadors --------------------

def gen_singles(piece_mesh: o3d.geometry.TriangleMesh, out_dist: Path, *,
                width:int, height:int, fov_deg:float,
                dist_m:float, z_floor:float,
                az_start:float, az_stop:float, az_step:float,
                el_start:float, el_stop:float, el_step:float,
                near:float, far:float):
    (out_dist/"points3d").mkdir(parents=True, exist_ok=True)
    (out_dist/"depth_npy").mkdir(parents=True, exist_ok=True)
    (out_dist/"mask_npy").mkdir(parents=True, exist_ok=True)
    (out_dist/"depth_png").mkdir(parents=True, exist_ok=True)
    (out_dist/"mask_png").mkdir(parents=True, exist_ok=True)
    (out_dist/"meta").mkdir(parents=True, exist_ok=True)

    aabb = piece_mesh.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    center_world = np.array([0.0, 0.0, z_floor + 0.5*ext[2]], float)

    fx, fy, cx, cy = intr_from_fov(width, height, fov_deg)

    az_vals = np.arange(az_start, az_stop+1e-6, az_step, dtype=float)
    el_vals = np.arange(el_start, el_stop+1e-6, el_step, dtype=float)

    idx = 0
    for el in el_vals:
        for az in az_vals:
            T_wc = spherical_cam_Twc(center_world, dist_m, az, el)
            inst = copy.deepcopy(piece_mesh)
            inst.translate((0.0, 0.0, z_floor))
            depth, mask, pts = raycast_depth_mask_points(inst, width, height, fov_deg, T_wc, near, far)

            stem = f"single_{idx:04d}_az{int(round(az)):03d}_el{int(round(el)):02d}"
            np.save(out_dist/"points3d"/f"{stem}.npy", pts.astype(np.float32))
            np.save(out_dist/"depth_npy"/f"{stem}.npy", depth.astype(np.float32))
            np.save(out_dist/"mask_npy"/f"{stem}.npy", mask.astype(np.bool_))
            save_png_depth(depth, out_dist/"depth_png"/f"{stem}.png")
            save_png_mask(mask,  out_dist/"mask_png"/f"{stem}.png")

            meta = {
                "type": "single",
                "distance_m": float(dist_m),
                "angles": {"az_deg": float(az), "el_deg": float(el)},
                "z_floor": float(z_floor),
                "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "T_wc": T_wc.tolist(),
                "piece_aabb": {"w": float(ext[0]), "d": float(ext[1]), "h": float(ext[2])},
                "points_N": int(pts.shape[0]),
            }
            (out_dist/"meta"/f"meta_{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            idx += 1

def gen_scenes(piece_mesh: o3d.geometry.TriangleMesh, out_dist: Path, *,
               scenes_per_dist:int, min_objs:int, max_objs:int,
               bin_w:float, bin_d:float, gap_factor:float,
               width:int, height:int, fov_deg:float, dist_m:float, z_floor:float,
               near:float, far:float):
    (out_dist/"points3d").mkdir(parents=True, exist_ok=True)
    (out_dist/"depth_npy").mkdir(parents=True, exist_ok=True)
    (out_dist/"mask_npy").mkdir(parents=True, exist_ok=True)
    (out_dist/"depth_png").mkdir(parents=True, exist_ok=True)
    (out_dist/"mask_png").mkdir(parents=True, exist_ok=True)
    (out_dist/"meta").mkdir(parents=True, exist_ok=True)

    # Càmera FIXA TOP-DOWN (el=90º)
    aabb = piece_mesh.get_axis_aligned_bounding_box()
    ext = aabb.get_extent()
    target = np.array([0.0, 0.0, z_floor + 0.5*ext[2]], float)
    eye = target + np.array([0.0, 0.0, dist_m], float)
    T_wc = lookat(eye, target, up=np.array([0,1,0], float))
    fx, fy, cx, cy = intr_from_fov(width, height, fov_deg)

    rad = footprint_radius_xy_from_aabb(aabb)

    lines = []
    for i in range(scenes_per_dist):
        k = random.randint(min_objs, max_objs)
        centers = sample_non_overlapping_xy(k, bin_w, bin_d, rad, gap_factor)
        if len(centers) < k:
            k = len(centers)

        scene = o3d.geometry.TriangleMesh()
        poses = []
        for (cx, cy) in centers:
            R = random_rotation_matrix()
            inst = rotate_and_place_on_floor(piece_mesh, R, cx, cy, z_floor)
            scene += inst
            poses.append({"R": R.tolist(), "tx": float(cx), "ty": float(cy), "tz": float(z_floor)})

        depth, mask, pts = raycast_depth_mask_points(scene, width, height, fov_deg, T_wc, near, far)

        stem = f"scene_{i:04d}_N{k}"
        np.save(out_dist/"points3d"/f"{stem}.npy", pts.astype(np.float32))
        np.save(out_dist/"depth_npy"/f"{stem}.npy", depth.astype(np.float32))
        np.save(out_dist/"mask_npy"/f"{stem}.npy", mask.astype(np.bool_))
        save_png_depth(depth, out_dist/"depth_png"/f"{stem}.png")
        save_png_mask(mask,  out_dist/"mask_png"/f"{stem}.png")

        meta = {
            "type": "scene",
            "distance_m": float(dist_m),
            "num_objects": int(k),
            "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy, "cx": cx, "cy": cy},
            "T_wc": T_wc.tolist(),
            "bin_size": {"w": bin_w, "d": bin_d},
            "z_floor": float(z_floor),
            "piece_aabb": {"w": float(ext[0]), "d": float(ext[1]), "h": float(ext[2])},
            "poses": poses,
            "points_N": int(pts.shape[0]),
        }
        (out_dist/"meta"/f"meta_{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        lines.append(f"{stem},{k}")

        if (i+1) % max(1, scenes_per_dist//10) == 0:
            print(f"  [{dist_m:.2f} m] scenes {i+1}/{scenes_per_dist}")

    (out_dist/"gt_counts.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✔ Distància {dist_m:.2f} m → {scenes_per_dist} escenes (gt_counts.csv)")

# -------------------- MAIN --------------------

def main():
    ap = argparse.ArgumentParser("Dataset (Singles esfèrics) + (Scenes top-down amb orientació 3D aleatòria)")
    ap.add_argument("--piece_stl", required=True, help="STL de la peça (metres)")
    ap.add_argument("--out_dir", required=True, help="Carpeta arrel de sortida")

    # Mode automàtic
    ap.add_argument("--auto", action="store_true", help="Generació automàtica amb valors per defecte sensats")

    # Parametritzable (només s'usen si NO es passa --auto)
    ap.add_argument("--dists", type=float, nargs="+", help="Distàncies càmera-objecte (m)")
    ap.add_argument("--az_start", type=float, default=0.0)
    ap.add_argument("--az_stop",  type=float, default=360.0)
    ap.add_argument("--az_step",  type=float, default=30.0)
    ap.add_argument("--el_start", type=float, default=5.0)
    ap.add_argument("--el_stop",  type=float, default=85.0)
    ap.add_argument("--el_step",  type=float, default=10.0)
    ap.add_argument("--scenes_per_dist", type=int, default=120)
    ap.add_argument("--min_objs", type=int, default=3)
    ap.add_argument("--max_objs", type=int, default=18)
    ap.add_argument("--bin_w", type=float, default=0.25)
    ap.add_argument("--bin_d", type=float, default=0.25)
    ap.add_argument("--gap_factor", type=float, default=1.05)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fov_deg", type=float, default=60.0)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--far", type=float, default=2.0)
    ap.add_argument("--z_floor", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Defaults automàtics
    if args.auto:
        if not args.dists:
            args.dists = [0.30, 0.40, 0.50, 0.60]
        # la resta d'opcions ja tenen default sensat

    # Sanity si NO auto i no especifiques distàncies
    if not args.auto and not args.dists:
        raise SystemExit("Cal passar --dists si no uses --auto (ex: --dists 0.30 0.40 0.50)")

    random.seed(args.seed); np.random.seed(args.seed)

    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    singles_root = out_root / "singles"
    scenes_root  = out_root / "scenes"

    mesh_in = o3d.io.read_triangle_mesh(args.piece_stl)
    if mesh_in.is_empty():
        raise RuntimeError("No s'ha pogut llegir --piece_stl")
    mesh_in.compute_vertex_normals()
    piece, aabb = center_mesh_to_bottom(mesh_in)
    ext = aabb.get_extent()
    print(f"→ Peça recentrada: AABB w={ext[0]:.4f} d={ext[1]:.4f} h={ext[2]:.4f} m")

    for dist in args.dists:
        out_dist_s = singles_root / f"dist_{dist:.2f}m"
        gen_singles(piece, out_dist_s,
                    width=args.width, height=args.height, fov_deg=args.fov_deg,
                    dist_m=dist, z_floor=args.z_floor,
                    az_start=args.az_start, az_stop=args.az_stop, az_step=args.az_step,
                    el_start=args.el_start, el_stop=args.el_stop, el_step=args.el_step,
                    near=args.near, far=args.far)
        print(f"✔ Singles {dist:.2f} m generats.")

        out_dist_c = scenes_root / f"dist_{dist:.2f}m"
        gen_scenes(piece, out_dist_c,
                   scenes_per_dist=args.scenes_per_dist,
                   min_objs=args.min_objs, max_objs=args.max_objs,
                   bin_w=args.bin_w, bin_d=args.bin_d, gap_factor=args.gap_factor,
                   width=args.width, height=args.height, fov_deg=args.fov_deg,
                   dist_m=dist, z_floor=args.z_floor, near=args.near, far=args.far)

    print("\nDataset generat correctament.")

if __name__ == "__main__":
    main()
