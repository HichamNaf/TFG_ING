#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, math
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# ---------------- 2D→3D ----------------

def depth_to_points3d(depth_m: np.ndarray, intr: dict):
    H, W = depth_m.shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    vs, us = np.where(np.isfinite(depth_m) & (depth_m > 0))
    if vs.size == 0:
        return np.empty((0,3), np.float32)
    z = depth_m[vs, us]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], 1).astype(np.float32)

def points_from_bbox(depth_m: np.ndarray, intr: dict, xyxy, pad_px:int=0):
    H, W = depth_m.shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    x1 = max(0, x1 - pad_px); y1 = max(0, y1 - pad_px)
    x2 = min(W, x2 + pad_px); y2 = min(H, y2 + pad_px)
    crop = depth_m[y1:y2, x1:x2]
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    vs, us = np.where(np.isfinite(crop) & (crop > 0))
    if vs.size == 0:
        return np.empty((0,3), np.float32), (x1,y1,x2,y2)
    z = crop[vs, us]
    uu, vv = us + x1, vs + y1
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], 1).astype(np.float32), (x1,y1,x2,y2)

def project_m_to_px(length_m: float, z_m: float, intr: dict) -> int:
    fx = intr["fx"]
    px = int(round(fx * (length_m / max(z_m, 1e-6))))
    return max(1, px)

# ---------------- ROI top (punts més alts) ----------------

def top_peak_window(pts: np.ndarray, piece_size_m: float|None, base_dz: float = 0.006, fallback=True):
    if pts.shape[0] == 0:
        return pts, 0.0, False
    z = pts[:,2]
    z_min = float(np.min(z))
    q75  = float(np.quantile(z, 0.75)) if pts.shape[0] >= 8 else z_min + base_dz
    dz = max(base_dz, q75 - z_min)
    if piece_size_m is not None:
        dz = min(dz, 0.9*piece_size_m)
    mask = z <= (z_min + dz)
    sel = pts[mask]
    if fallback and sel.shape[0] < 40:
        p20 = float(np.quantile(z, 0.20))
        mask2 = z <= p20
        sel2 = pts[mask2]
        if sel2.shape[0] > sel.shape[0]:
            return sel2, (p20 - z_min), True
    return sel, dz, False

# ---------------- PCA a XY i oriented box ----------------

def pca_xy(points_xy: np.ndarray):
    C = points_xy.mean(0)
    X = points_xy - C
    H = (X.T @ X) / max(1, len(points_xy)-1)
    vals, vecs = np.linalg.eigh(H)
    order = np.argsort(vals)[::-1]
    vals = vals[order]; vecs = vecs[:, order]
    # e0 = eix llarg (variància gran), e1 = eix curt
    e0 = vecs[:,0]; e1 = vecs[:,1]
    if np.linalg.det(vecs) < 0:
        e1 = -e1
    return C, e0, e1, vals

def oriented_extent_xy(points_xy: np.ndarray, C, e0, e1):
    X = points_xy - C
    u = X @ e0  # coordenada llarg
    v = X @ e1  # coordenada curt
    umin, umax = float(np.min(u)), float(np.max(u))
    vmin, vmax = float(np.min(v)), float(np.max(v))
    L = umax - umin  # llargària
    W = vmax - vmin  # amplada
    return (umin, umax, vmin, vmax, L, W), u, v

# ---------------- Frames i transformacions ----------------

def build_T_yaw(yaw_rad: float, center_xyz: np.ndarray):
    c,s = np.cos(yaw_rad), np.sin(yaw_rad)
    R = np.array([[ c,-s, 0],
                  [ s, c, 0],
                  [ 0, 0, 1]], dtype=float)  # gir al pla XY càmera
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = center_xyz
    return T

def transform_points_inverse(T: np.ndarray, P: np.ndarray):
    Rin = T[:3,:3].T
    tin = -Rin @ T[:3,3]
    return (P @ Rin.T) + tin

# ---------------- Geometria pinça i mètriques ----------------

def cylinder_collision_mask(P_world: np.ndarray, T_grasp: np.ndarray,
                            width: float, finger_radius: float, finger_depth: float):
    """Retorna booleà per punts dins *qualsevol* cilindre (coords gripper: eix Z)."""
    if P_world.shape[0] == 0:
        return np.zeros((0,), bool)
    P = transform_points_inverse(T_grasp, P_world)
    X, Y, Z = P[:,0], P[:,1], P[:,2]
    inZ = (Z >= -finger_depth/2) & (Z <= +finger_depth/2)
    # cilindre esquerre a x=-w/2
    rA2 = (X + width/2)**2 + Y**2
    inA = inZ & (rA2 <= (finger_radius*finger_radius))
    # cilindre dret a x=+w/2
    rB2 = (X - width/2)**2 + Y**2
    inB = inZ & (rB2 <= (finger_radius*finger_radius))
    return inA | inB, P

def contact_band_counts(P_roi_in_gripper: np.ndarray, width: float,
                        finger_radius: float, finger_depth: float, band: float,
                        end_clear_u_min: float, end_clear_u_max: float):
    """
    Comptem contactes a banda [r∈(R-band, R+band)] i dins de z-range, i també
    **allunyats dels extrems** al llarg de l’eix llarg (u ≡ e0).
    S’assumeix que, en coords ‘gripper’, l’eix llarg és l’eix ‘u’ (definim això a l’ús).
    Aquí, però, usem l’eix ‘y’ del gripper com a llarg i ‘x’ com a tancament.
    """
    if P_roi_in_gripper.shape[0] == 0:
        return 0, 0
    X, Y, Z = P_roi_in_gripper[:,0], P_roi_in_gripper[:,1], P_roi_in_gripper[:,2]
    inZ = (Z >= -finger_depth/2) & (Z <= +finger_depth/2)

    # llindars radials de contacte
    rA = np.sqrt((X + width/2)**2 + Y**2)
    rB = np.sqrt((X - width/2)**2 + Y**2)
    bandA = inZ & (np.abs(rA - finger_radius) <= band)
    bandB = inZ & (np.abs(rB - finger_radius) <= band)

    # allunyar-se dels extrems al llarg de l’eix llarg (usem Y com a “u”)
    in_end = (Y >= end_clear_u_min) & (Y <= end_clear_u_max)
    cA = int(np.count_nonzero(bandA & in_end))
    cB = int(np.count_nonzero(bandB & in_end))
    return cA, cB

# ---------------- Visuals ----------------

def create_cyl_fingers(width: float, radius: float, depth: float, res=30):
    A = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=depth, resolution=res, split=1)
    B = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=depth, resolution=res, split=1)
    A.translate((-width/2, 0, 0))
    B.translate((+width/2, 0, 0))
    A.compute_vertex_normals(); B.compute_vertex_normals()
    A.paint_uniform_color([0.1, 0.8, 0.1]); B.paint_uniform_color([0.1, 0.8, 0.1])
    return A + B

# ---------------- MAIN ----------------

def main():
    ap = argparse.ArgumentParser("YOLO → ROI → Grasp auto (evita cantonades, antipodal, seguretat)")
    ap.add_argument("--shot_dir", required=True, help="Carpeta amb rgb.png, depth.npy, meta.json")
    ap.add_argument("--weights", required=True, help="Pesos YOLO")
    ap.add_argument("--out", default="out_auto_grasp_v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)

    # pinça
    ap.add_argument("--max_open", type=float, default=0.05)
    ap.add_argument("--finger_radius", type=float, default=0.005)
    ap.add_argument("--finger_depth", type=float, default=0.04)

    # qualitat / seguretat
    ap.add_argument("--contact_band", type=float, default=0.002, help="± banda radial per considerar contacte")
    ap.add_argument("--safety_clear", type=float, default=0.003, help="distància mínima a punts fora ROI")
    ap.add_argument("--end_clear_frac", type=float, default=0.18, help="fracció per evitar extrems (u)")

    # ROI / alçada
    ap.add_argument("--piece_size_m", type=float, default=None)
    ap.add_argument("--approach_offset", type=float, default=0.010)

    # cerca
    ap.add_argument("--yaw_span_deg", type=float, default=50.0)
    ap.add_argument("--yaw_step_deg", type=float, default=10.0)
    ap.add_argument("--shift_long_mm", type=float, default=20.0)
    ap.add_argument("--shift_long_steps", type=int, default=5)

    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Carrega dades
    shot = Path(args.shot_dir)
    rgb = cv2.imread(str(shot / "rgb.png"))
    depth_m = np.load(shot / "depth.npy").astype(np.float32)
    meta = json.loads((shot / "meta.json").read_text())
    intr = meta["intrinsics"]
    depth_m[depth_m <= 0] = np.nan

    # YOLO
    model = YOLO(args.weights)
    res = model.predict(rgb[..., ::-1], imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
    if res.boxes is None or res.boxes.xyxy.shape[0] == 0:
        raise SystemExit("❌ Cap detecció YOLO.")
    confs = res.boxes.conf.cpu().numpy()
    xyxy = res.boxes.xyxy.cpu().numpy()[np.argmax(confs)]
    x1,y1,x2,y2 = [int(round(v)) for v in xyxy]

    # padding
    patch = depth_m[max(0,y1):min(depth_m.shape[0],y2), max(0,x1):min(depth_m.shape[1],x2)]
    z_med = np.nanmedian(patch[np.isfinite(patch)]) if np.isfinite(patch).any() else np.nan
    pad_px = 12
    if np.isfinite(z_med) and args.piece_size_m is not None:
        pad_px = max(pad_px, project_m_to_px(args.piece_size_m*0.35, z_med, intr))

    # punts dins bbox
    pts_bbox, (x1p,y1p,x2p,y2p) = points_from_bbox(depth_m, intr, xyxy, pad_px=pad_px)
    if pts_bbox.shape[0] < 30:
        raise SystemExit("❌ Pocs punts dins del bbox.")

    # ROI del cap
    pts_peak, dz, used_fallback = top_peak_window(pts_bbox, args.piece_size_m, fallback=True)
    if pts_peak.shape[0] == 0:
        pts_peak = pts_bbox.copy(); used_fallback = True

    # núvol complet per seguretat
    pts_all = depth_to_points3d(depth_m, intr)
    if pts_all.shape[0] == 0:
        raise SystemExit("❌ Sense punts al núvol complet.")

    # PCA a XY de la ROI — e0: llarg, e1: curt
    Cxy, e0, e1, _ = pca_xy(pts_peak[:,:2])
    (umin, umax, vmin, vmax, L, W), u, v = oriented_extent_xy(pts_peak[:,:2], Cxy, e0, e1)

    # yaw base: volem dits paral·lels a e0 (llarg) → tancament sobre e1 (curt)
    yaw_base = math.degrees(math.atan2(e0[1], e0[0]))
    # centre XY i Z
    z_min = float(np.min(pts_peak[:,2]))
    z_center = max(1e-3, z_min - args.approach_offset)
    center_xy = Cxy.copy()
    center0 = np.array([center_xy[0], center_xy[1], z_center], float)

    # end-clear al llarg del llarg (u=Y gripper): evitem ±extrems
    end_clear = args.end_clear_frac * L
    u_min_ok = umin + end_clear
    u_max_ok = umax - end_clear

    # cerca de yaw (al voltant del base) i desplaçament al llarg del llarg
    span = args.yaw_span_deg; step = max(1.0, args.yaw_step_deg)
    k = int(round(span/step))
    yaw_list = [yaw_base + i*step for i in range(-k, k+1)]

    if args.shift_long_steps <= 1:
        shifts_u = [0.0]
    else:
        sh_max = args.shift_long_mm / 1000.0
        shifts_u = list(np.linspace(-sh_max, +sh_max, args.shift_long_steps))

    # amplada automàtica de pinça
    width0 = min(args.max_open, max(0.0, W + 0.003))  # W (eix curt) + marge
    widths = sorted(set([width0,
                         min(args.max_open, width0 + 0.005),
                         min(args.max_open, width0 + 0.010)]))

    # punts ROI i no-ROI (per seguretat)
    # (Approx) fem una màscara no-ROI: dist XY al centre i Z fora del cap
    dxy_all = np.linalg.norm(pts_all[:,:2] - center_xy[None,:], axis=1)
    non_roi_mask = (dxy_all > max(0.5*L, 0.5*W)) | (pts_all[:,2] > (z_min + dz + 1e-3))
    pts_env = pts_all[non_roi_mask]  # entorn per col·lisions
    if pts_env.shape[0] == 0:
        pts_env = pts_all  # fallback conservador

    best = None  # (score DESC), detalls…

    for yaw_deg in yaw_list:
        yaw_rad = math.radians(yaw_deg)
        # defineix eixos del gripper: X=closing, Y=long, Z=approach (match amb gir Z)
        # volem que el vector llarg e0 projecti en +Y_gripper → per això centre després
        Rz = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad)],
                       [math.sin(yaw_rad),  math.cos(yaw_rad)]], float)
        # projectem punts ROI a coords gripper XY per saber u=Y
        XY_roi = (pts_peak[:,:2] - center_xy[None,:]) @ Rz.T
        Y_roi = XY_roi[:,1]
        # marges d’extrems en coords gripper: fem servir mateix u_min_ok/u_max_ok
        for du in shifts_u:
            # si el desplaçament du porta el centre fora zona interior, penalitzem menys candidats
            center_xy_cand = center_xy + du * e0
            center_cand = np.array([center_xy_cand[0], center_xy_cand[1], z_center], float)

            # recalcular coords ROI en aquest centre (gripper coords)
            XY_roi_cand = (pts_peak[:,:2] - center_xy_cand[None,:]) @ Rz.T
            Y_roi_cand = XY_roi_cand[:,1]

            # filtre d’extrems: punts considerats "interiors"
            in_end = (Y_roi_cand >= (u_min_ok - center_xy.dot(e0) + center_xy_cand.dot(e0))) & \
                     (Y_roi_cand <= (u_max_ok - center_xy.dot(e0) + center_xy_cand.dot(e0)))

            for w in widths:
                T = build_T_yaw(yaw_rad, center_cand)

                # 1) col·lisió dura amb entorn (no-ROI)
                coll_env_mask, _ = cylinder_collision_mask(pts_env, T, w, args.finger_radius, args.finger_depth)
                if np.any(coll_env_mask):
                    continue  # descartat

                # 2) seguretat: mínima distància radial respecte entorn
                # aproximem: punts env a coords gripper
                P_env = transform_points_inverse(T, pts_env)
                X, Y, Z = P_env[:,0], P_env[:,1], P_env[:,2]
                inZ = (Z >= -args.finger_depth/2) & (Z <= +args.finger_depth/2)
                rA = np.sqrt((X + w/2)**2 + Y**2)
                rB = np.sqrt((X - w/2)**2 + Y**2)
                dA = rA - args.finger_radius
                dB = rB - args.finger_radius
                # només considerem punts dins banda Z (possibles col·lisions d’alçada)
                safety_min = float(np.min(np.concatenate([dA[inZ], dB[inZ]]))) if np.any(inZ) else 1e9
                if safety_min < args.safety_clear:
                    continue  # descartat per proximitat

                # 3) contactes antipodals sobre ROI (banda de contacte)
                P_roi = transform_points_inverse(T, pts_peak)
                cA, cB = contact_band_counts(P_roi, w, args.finger_radius, args.finger_depth,
                                             band=args.contact_band,
                                             end_clear_u_min=(u_min_ok - center_xy.dot(e0) + center_xy_cand.dot(e0)),
                                             end_clear_u_max=(u_max_ok - center_xy.dot(e0) + center_xy_cand.dot(e0)))
                contacts = min(cA, cB)  # volem els dos dits amb suport

                # 4) centrament (penalitza estar massa cap a l’extrem)
                # usem desviació mitjana de Y_roi_cand respecte 0 quan "in_end"
                if np.any(in_end):
                    cent_pen = float(np.mean(np.abs(Y_roi_cand[in_end])))
                else:
                    cent_pen = float(np.mean(np.abs(Y_roi_cand))) if Y_roi_cand.size else 1e3

                # score (més alt millor): contactes + seguretat - penalització
                score = (contacts) + 100.0 * min(safety_min, 0.02) - 5.0 * cent_pen

                if (best is None) or (score > best[0]):
                    best = (score, yaw_deg, du, w, center_cand, T, contacts, safety_min, cent_pen)

    if best is None:
        raise SystemExit("❌ Cap candidat de presa supera seguretat. Prova a pujar --safety_clear o ampliar ROI.")

    score, yaw_deg_sel, du_sel, width_sel, center_sel, T_sel, contacts_sel, safety_sel, cent_pen_sel = best

    print(f"→ Seleccionat:"
          f" yaw={yaw_deg_sel:.1f}°, shift_long={du_sel*1000:.1f} mm,"
          f" width={width_sel*1000:.1f} mm, contacts={contacts_sel},"
          f" safety={safety_sel*1000:.1f} mm, score={score:.1f}")

    # ---------- Overlay 2D ----------
    rgb_vis = rgb.copy()
    cv2.rectangle(rgb_vis, (x1p,y1p), (x2p,y2p), (0,255,255), 2)
    txt = f"yaw={yaw_deg_sel:.1f}  w={width_sel*1000:.0f}mm  saf={safety_sel*1000:.0f}mm  c={contacts_sel}"
    cv2.putText(rgb_vis, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    if used_fallback:
        cv2.putText(rgb_vis, "ROI fallback", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    cv2.imwrite(str(out_dir/"overlay_bbox.png"), rgb_vis)

    # ---------- JSON resultat ----------
    out_meta = dict(
        center_xyz=center_sel.tolist(),
        yaw_deg=float(yaw_deg_sel),
        width_m=float(width_sel),
        finger_radius_m=float(args.finger_radius),
        finger_depth_m=float(args.finger_depth),
        contacts=int(contacts_sel),
        safety_min_m=float(safety_sel),
        score=float(score),
        bbox_xyxy=[int(x1p), int(y1p), int(x2p), int(y2p)],
        roi_fallback=bool(used_fallback),
        L=float(L), W=float(W), end_clear_frac=float(args.end_clear_frac)
    )
    (out_dir/"grasp.json").write_text(json.dumps(out_meta, indent=2), encoding="utf-8")

    # ---------- Visualització 3D ----------
    pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_all))
    pcd_all.paint_uniform_color([0.72,0.72,0.72])
    pcd_roi = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_peak))
    pcd_roi.paint_uniform_color([1.0,0.0,0.0])

    gripper = create_cyl_fingers(width=width_sel,
                                 radius=args.finger_radius,
                                 depth=args.finger_depth)
    gripper.transform(T_sel)

    o3d.visualization.draw_geometries(
        [pcd_all, pcd_roi, gripper],
        window_name="Núvol (gris) + ROI (vermell) + Pinça (verd) — v2 (antipodal & seguretat)"
    )

if __name__ == "__main__":
    main()
