#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UR3 + RealSense dataset generator (real captures + scenes):

Modes de funcionament
---------------------
1) CAPTURA INDIVIDUALS (rotacions UR3 sobre peça centrada):
   - Especificar --stl i --out_dir.
   - El robot fa sweeps sobre RX i RY dins límits definits (--rx_min, --rx_max, --ry_min, --ry_max).
   - També executa combinacions RXxRY amb nombre de passos (--steps_combo).
   - Cada captura desa:
       * RGB + overlay
       * Depth (PNG + NPY)
       * Màscara STL (PNG + NPY)
       * Metadades amb posició TCP i intrínsecs ideals
   exemple:
       python captura_centered_allinone.py --stl model.stl --out_dir dataset_rs --stl_units mm 
           --cam_roll_deg -90 --roll_mode post_world 
           --steps_x 5 --steps_y 5 --rx_min -35 --rx_max 40 --ry_min -28 --ry_max 20

2) GENERACIÓ ESCENES (a partir de captures reals):
   - Un cop completada la fase de captures, es combinen aleatòriament depth_npy + mask_npy
     per crear escenes amb diverses instàncies de la peça.
   - Paràmetres de síntesi: --scenes_per_dist, --min_objs, --max_objs, --place_gap_px.
   - Les escenes reprodueixen l'estructura del dataset sintètic, incloent points3d i gt_counts.csv.
   exemple:
       python captura_centered_allinone.py --stl model.stl --out_dir dataset_rs --stl_units mm 
           --scenes_per_dist 120 --min_objs 3 --max_objs 12 --place_gap_px 4

Estructura de sortida
---------------------
out_dir/
    run_YYYYMMDD_HHMMSS/       (captura única amb segell temporal)
        rgb_png/               (imatges RGB originals)
        rgb_overlay_png/       (RGB amb contorn màscara STL)
        depth_png/             (PNG 8-bit de profunditat normalitzada)
        mask_png/              (PNG binàries de la màscara STL)
        depth_npy/             (profunditat amb NaN, Numpy .npy)
        depth_masked_npy/      (profunditat limitada a la màscara, Numpy .npy)
        mask_npy/              (màscara booleana, Numpy .npy)
        meta/                  (JSON amb intrínsecs ideals, TCP i info de captura)
    scenes/
        dist_<Zc>m/            (escenes combinades a partir de captures reals)
            points3d/          (punts 3D visibles, Numpy .npy)
            depth_npy/         (profunditat amb NaN)
            mask_npy/          (màscara booleana)
            depth_png/         (PNG 8-bit, profunditat normalitzada)
            mask_png/          (PNG 8-bit, màscara binària)
            meta/              (metadades per escena)
            gt_counts.csv      (CSV amb recompte de peces per escena)

Notes
-----
- La càmera RealSense s'utilitza com a font de RGB-D real.
- El braç UR3 executa moviments RX/RY controlats via RTDE.
- Les escenes es generen en postprocessament combinant màscares + profunditats reals.
"""
import argparse, os, json, math, time, socket, datetime, csv, random, copy
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import pyrealsense2 as rs
from rtde_receive import RTDEReceiveInterface

# ───────── Config robot ─────────
IP    = "192.168.1.104"
ACC   = 0.35
VEL   = 0.12
BLEND = 0.0
DWELL = 3.0
BASE_EXTRA = 4.0

J_BASE = [math.radians(0.0),
          math.radians(-111.0),
          math.radians(-88.0),
          math.radians(-70.0),
          math.radians(90.0),
          math.radians(0.0)]

# Límits i passos EDITABLES
RX_MIN_DEG = -35.0
RX_MAX_DEG = +40.0
RY_MIN_DEG = -28.0
RY_MAX_DEG = +20.0
STEPS_X = 5
STEPS_Y = 5

# Combinacions
RX_TARGETS_DEG = [-20.0, -10.0, +10.0]
RY_TARGETS_DEG = [-15.0,  -5.0,  +5.0, +15.0]
STEPS_COMBO    = 6

RETRY_ON_SMALL_MOVE = True
SMALL_MOVE_DEG = 1.0
MAX_RETRY = 1

# ───────── Càmera ─────────
WIDTH, HEIGHT, FPS = 640, 480, 30
MASK_SHRINK_PX = 2
MIN_PTS_CROP = 50

# ───────── Helpers PNG ─────────
def write_png_gray(img_uint8, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_uint8)

# ───────── Àlgebra ─────────
def Rx(a): ca,sa=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]],float)
def Ry(a): ca,sa=math.cos(a),math.sin(a); return np.array([[ca,0,sa],[0,1,0],[-sa,0,ca]],float)
def Rz(a): ca,sa=math.cos(a),math.sin(a); return np.array([[ca,-sa,0],[sa,ca,0],[0,0,1]],float)

def rotvec_to_R(v):
    vx,vy,vz = float(v[0]), float(v[1]), float(v[2])
    th = math.sqrt(vx*vx+vy*vy+vz*vz)
    if th < 1e-12: return np.eye(3)
    kx,ky,kz = vx/th, vy/th, vz/th
    K = np.array([[0,-kz,ky],[kz,0,-kx],[-ky,kx,0]], float)
    return np.eye(3) + math.sin(th)*K + (1-math.cos(th))*(K@K)

def R_to_rotvec(R):
    tr = float(np.trace(R)); ct = max(-1.0, min(1.0, (tr-1.0)/2.0))
    th = math.acos(ct)
    if th < 1e-12: return np.array([0.0,0.0,0.0], float)
    rx = (R[2,1]-R[1,2])/(2*math.sin(th))
    ry = (R[0,2]-R[2,0])/(2*math.sin(th))
    rz = (R[1,0]-R[0,1])/(2*math.sin(th))
    return np.array([rx,ry,rz], float) * th

def compose_rot_abs(before_rotvec, delta_rotvec):
    Rb = rotvec_to_R(before_rotvec); Rd = rotvec_to_R(delta_rotvec)
    return R_to_rotvec(Rd @ Rb)

def robust_delta_mag_deg(before_rotvec, after_rotvec):
    Rb = rotvec_to_R(before_rotvec); Ra = rotvec_to_R(after_rotvec)
    dv = R_to_rotvec(Ra @ Rb.T)
    return math.degrees(float(np.linalg.norm(dv)))

def euler_zyx_from_R(R):
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-9
    if not singular:
        yaw   = math.atan2(R[1,0], R[0,0])
        pitch = math.atan2(-R[2,0], sy)
        roll  = math.atan2(R[2,1], R[2,2])
    else:
        yaw   = math.atan2(-R[0,1], R[1,1])
        pitch = math.atan2(-R[2,0], sy)
        roll  = 0.0
    return yaw, pitch, roll

# ───────── URScript ─────────
def _send_program(lines):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(3.0)
    try:
        s.connect((IP, 30002))
        s.send(b"def prog():\n")
        for ln in lines: s.send(ln.encode("utf-8"))
        s.send(b"  sync()\nend\n")
    finally:
        s.close()

def movej(joints_rad, a=1.2, v=0.7, t=0.0, r=0.0):
    j = joints_rad
    _send_program([f"  movej([{j[0]},{j[1]},{j[2]},{j[3]},{j[4]},{j[5]}], a={a}, v={v}, t={t}, r={r})\n"])

def movel_abs_pose(pose_vec, a=ACC, v=VEL, r=BLEND):
    x,y,z,rx,ry,rz = pose_vec
    _send_program([
        f"  tgt = p[{x},{y},{z},{rx},{ry},{rz}]\n",
        f"  movel(tgt, a={a}, v={v}, t=0, r={r})\n",
    ])

def dash_stop():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(3.0)
        s.connect((IP, 29999)); s.send(b"stop\n")
        try: s.recv(1024)
        except: pass
        s.close()
    except: pass

# ───────── RTDE ─────────
class PoseReader:
    def __init__(self, ip, csv_path="movement_log_simple.csv", verbose=True):
        self.rtde = RTDEReceiveInterface(ip)
        self.csv_path = csv_path
        self.verbose = verbose
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["ts","phase","axis","cmd_drx","cmd_dry","cmd_drz",
                            "before_rx","before_ry","before_rz",
                            "after_rx","after_ry","after_rz","got_deg","got_deg_robust"])

    def tcp_pose(self):
        return np.array(self.rtde.getActualTCPPose(), float)

    def log(self, phase, axis, drx,dry,drz, before, after):
        d = after[3:] - before[3:]
        got = math.degrees(np.linalg.norm(d))
        grob = robust_delta_mag_deg(before[3:], after[3:])
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ts,phase,axis, f"{drx:.6f}",f"{dry:.6f}",f"{drz:.6f}",
                        f"{before[3]:.6f}",f"{before[4]:.6f}",f"{before[5]:.6f}",
                        f"{after[3]:.6f}",f"{after[4]:.6f}",f"{after[5]:.6f}",
                        f"{got:.3f}", f"{grob:.3f}"])
        if self.verbose:
            print(f"[{phase}/{axis}] |Δ|_robust = {grob:.2f}°")

# ───────── RealSense ─────────
def intr_from_rs(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return dict(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
                W=intr.width, H=intr.height, depth_scale=depth_scale)

def intr_from_fov(width: int, height: int, fov_deg: float):
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    fy = fx
    return fx, fy, cx, cy

# ───────── Raycast màscara ─────────
def _center_mesh_inplace(mesh: o3d.geometry.TriangleMesh):
    ctr = mesh.get_center()
    mesh.translate((-ctr[0], -ctr[1], -ctr[2]))

def raycast_mask_stl_in_cam(mesh_legacy, fx, fy, cx, cy, width, height, dist_m, R_mesh_in_cam):
    inst = o3d.geometry.TriangleMesh(mesh_legacy)
    _center_mesh_inplace(inst)
    inst.rotate(R_mesh_in_cam, center=(0,0,0))
    inst.translate((0,0,dist_m))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(inst))

    u = np.arange(width, dtype=np.float32); v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    zc = np.ones_like(uu, np.float32)
    xc = (uu - cx) / fx; yc = (vv - cy) / fy
    dirs_cam = np.stack([xc, yc, zc], axis=-1)
    dirs_cam /= (np.linalg.norm(dirs_cam, axis=-1, keepdims=True) + 1e-12)

    origins = np.zeros_like(dirs_cam)
    rays = np.concatenate([origins.reshape(-1,3), dirs_cam.reshape(-1,3)], axis=1).astype(np.float32)
    ans  = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans['t_hit'].numpy().reshape(height, width)
    return (np.isfinite(t_hit)).astype(np.uint8)

# ───────── Orientació mesh↔càmera ─────────
def compute_R_mesh_cam(R_cam_tcp, tcp_rotvec, base_cam_R, cam_roll_rad, roll_mode="post_world"):
    R_tcp = rotvec_to_R(tcp_rotvec)
    if roll_mode == "pre_map":
        R_roll = Rz(cam_roll_rad)
        R_cam = (R_roll @ R_cam_tcp) @ R_tcp
        return R_cam.T @ base_cam_R
    elif roll_mode == "post_world":
        R_cam = R_cam_tcp @ R_tcp
        R_mesh_cam_no_roll = R_cam.T @ base_cam_R
        return Rz(cam_roll_rad) @ R_mesh_cam_no_roll
    elif roll_mode == "post_cam":
        R_cam = R_cam_tcp @ R_tcp
        return (R_cam.T @ base_cam_R) @ Rz(cam_roll_rad)
    else:
        R_cam = R_cam_tcp @ R_tcp
        return R_cam.T @ base_cam_R

# ───────── Overlays ─────────
def make_rgb_contour_overlay(color_bgr, mask01):
    vis = color_bgr.copy()
    if mask01 is None:
        return vis
    edges = cv2.Canny((mask01*255).astype(np.uint8), 50, 150)
    vis[edges>0] = (0,255,255)
    return vis

# ───────── Vista de centrament ─────────
def preview_center_overlay(reader, pipeline, profile, align, mesh,
                            R_cam_tcp_base, base_cam_R, cam_roll_rad, roll_mode,
                            init_shrink_px=3):
    intr = intr_from_rs(profile)
    shrink_px = int(init_shrink_px)

    print("[i] Vista de centrament: BASE (B), ajusta la peça a la creu i prem 's'  |  q sortir")
    print("    Tecles: [/] erosió  b BASE  s començar  q/ESC sortir")

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        dfrm = frames.get_depth_frame(); cfrm = frames.get_color_frame()
        if not dfrm or not cfrm:
            cv2.waitKey(1); continue

        z16 = np.asanyarray(dfrm.get_data())
        color = np.asanyarray(cfrm.get_data())
        H,W = z16.shape; cx,cy = W//2, H//2

        depth_scale = intr['depth_scale']
        patch = z16[max(0,cy-3):min(H,cy+4), max(0,cx-3):min(W,cx+4)]
        valid = patch[patch>0]
        Zc = float(valid.mean()*depth_scale) if valid.size>0 else float('nan')

        tcp = reader.tcp_pose()
        R_mesh_cam = compute_R_mesh_cam(R_cam_tcp_base, tcp[3:], base_cam_R, cam_roll_rad, roll_mode)

        mask01 = None
        if np.isfinite(Zc):
            fx, fy, icx, icy = intr_from_fov(W, H, 60.0)
            mask01 = raycast_mask_stl_in_cam(
                mesh, fx, fy, icx, icy, W, H, dist_m=Zc, R_mesh_in_cam=R_mesh_cam
            ).astype(np.uint8)
            if shrink_px > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*shrink_px+1, 2*shrink_px+1))
                mask01 = cv2.erode((mask01*255).astype(np.uint8), k, iterations=1) // 255

        vis = make_rgb_contour_overlay(color, mask01)
        cv2.drawMarker(vis, (cx,cy), (0,0,0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(vis, (cx,cy), (255,255,255), cv2.MARKER_CROSS, 14, 2)

        txt = f"Zc: {Zc:.3f} m | shrink: {shrink_px}px"
        cv2.putText(vis, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,0), 2, cv2.LINE_AA)

        d8 = cv2.convertScaleAbs(z16, alpha=0.03)
        d8 = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
        small = cv2.resize(d8, (W//3, H//3))
        vis[0:small.shape[0], -small.shape[1]:] = small

        cv2.imshow("Centrament (RGB + contorn màscara)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')): return None
        if k == ord('b'):
            print("➡️  BASE…"); movej(J_BASE, a=1.2, v=0.7); time.sleep(BASE_EXTRA)
        if k == ord('['): shrink_px = max(0, shrink_px-1)
        if k == ord(']'): shrink_px += 1
        if k == ord('s') and np.isfinite(Zc):
            # retorna intr + Zc inicial (distància única)
            intr['Zc_init'] = Zc
            intr['fov_deg'] = 60.0
            return intr

# ───────── Guardat ─────────
def ensure_dirs(root: str):
    for sub in ("rgb_png","rgb_overlay_png","depth_png","mask_png","depth_npy","depth_masked_npy","mask_npy","meta"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

def make_rgb_overlay_save(rgb_bgr, mask01):
    if rgb_bgr is None: return None
    edges = cv2.Canny((mask01.astype(np.uint8)*255), 50, 150)
    overlay = rgb_bgr.copy()
    overlay[edges>0] = (0,255,255)
    return overlay

def save_sample(out_dir, stem, rgb_bgr, depth_m, mask01, meta):
    ensure_dirs(out_dir)
    depth_masked = depth_m.copy()
    depth_masked[mask01 == 0] = 0.0

    if rgb_bgr is not None:
        cv2.imwrite(os.path.join(out_dir, "rgb_png",   f"{stem}.png"), rgb_bgr)
        ov = make_rgb_overlay_save(rgb_bgr, mask01)
        if ov is not None:
            cv2.imwrite(os.path.join(out_dir, "rgb_overlay_png", f"{stem}.png"), ov)
    d8 = cv2.normalize(depth_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "depth_png", f"{stem}.png"), d8)
    cv2.imwrite(os.path.join(out_dir, "mask_png",  f"{stem}.png"), (mask01.astype(np.uint8)*255))

    np.save(os.path.join(out_dir, "depth_npy",        f"{stem}.npy"), depth_m.astype(np.float32))
    np.save(os.path.join(out_dir, "depth_masked_npy", f"{stem}.npy"), depth_masked.astype(np.float32))
    np.save(os.path.join(out_dir, "mask_npy",         f"{stem}.npy"), mask01.astype(bool))

    with open(os.path.join(out_dir, "meta", f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

# ───────── Captura (màscara centrada) ─────────
def capture_mask_centered(depth_frame, color_frame, intr, mesh, R_mesh_cam, mask_shrink_px=MASK_SHRINK_PX):
    z16 = np.asanyarray(depth_frame.get_data())
    depth_m = z16.astype(np.float32) * intr['depth_scale']
    H,W = depth_m.shape; cx, cy = W//2, H//2

    patch = depth_m[max(0,cy-3):min(H,cy+4), max(0,cx-3):min(W,cx+4)]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size == 0: return None
    Zc = float(np.median(patch))

    fx, fy, icx, icy = intr_from_fov(W, H, intr.get('fov_deg',60.0))
    mdl_mask = raycast_mask_stl_in_cam(
        mesh, fx, fy, icx, icy, W, H, dist_m=Zc, R_mesh_in_cam=R_mesh_cam
    )
    if mask_shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*mask_shrink_px+1, 2*mask_shrink_px+1))
        mdl_mask = cv2.erode((mdl_mask*255).astype(np.uint8), k, iterations=1) // 255

    mask01 = (mdl_mask > 0).astype(np.uint8)
    pts_in = int(np.sum((mask01>0) & np.isfinite(depth_m) & (depth_m>0)))
    if pts_in < MIN_PTS_CROP:
        return None

    color_bgr = np.asanyarray(color_frame.get_data()) if color_frame else None
    meta = {
        "Zc_m": float(Zc),
        "intrinsics_ideal": {"W": W, "H": H,
                             "fx": float(fx), "fy": float(fy),
                             "cx": float(icx), "cy": float(icy)},
        "pts_in_mask": int(pts_in)
    }
    return dict(rgb=color_bgr, depth_m=depth_m, mask01=mask01, meta=meta)

# ───────── Helpers d’orientació relativa ─────────
def orientation_relative_to_base(after_rotvec, R_tcp_base0):
    R_after = rotvec_to_R(after_rotvec)
    R_rel   = R_after @ R_tcp_base0.T
    rv_rel  = R_to_rotvec(R_rel)
    rv_rel_deg = (np.degrees(rv_rel)).tolist()
    angle_deg  = float(np.degrees(np.linalg.norm(rv_rel)))
    yaw,pitch,roll = euler_zyx_from_R(R_rel)
    eul_zyx_deg = [float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))]
    return rv_rel_deg, angle_deg, eul_zyx_deg

# ───────── Moviment + captura ─────────
def mov_and_capture_step(reader, pipeline, align, mesh, intr,
                         base_cam_R, R_cam_tcp, cam_roll_rad, roll_mode,
                         R_tcp_base0,
                         axis, step_deg, out_dir, idx):
    drx = math.radians(step_deg) if axis=="rx" else 0.0
    dry = math.radians(step_deg) if axis=="ry" else 0.0
    before = reader.tcp_pose()
    target_rot = compose_rot_abs(before[3:], np.array([drx, dry, 0.0], float))
    tgt = np.array([before[0], before[1], before[2], target_rot[0], target_rot[1], target_rot[2]], float)

    def once():
        movel_abs_pose(tgt, a=ACC, v=VEL, r=BLEND)
        time.sleep(DWELL)
        after = reader.tcp_pose()
        return after, robust_delta_mag_deg(before[3:], after[3:])

    after, grob = once()
    if RETRY_ON_SMALL_MOVE and grob < SMALL_MOVE_DEG and MAX_RETRY>0:
        after, grob = once()
    if grob < SMALL_MOVE_DEG:
        print(f"   ❌ {axis} {step_deg:+.2f}° massa petit, s’omet.")
        return idx

    R_mesh_cam = compute_R_mesh_cam(R_cam_tcp, after[3:], base_cam_R, cam_roll_rad, roll_mode)
    frames = pipeline.wait_for_frames(); frames = align.process(frames)
    dfrm = frames.get_depth_frame(); cfrm = frames.get_color_frame()
    if not dfrm or not cfrm:
        print("   ❌ frame invàlid"); return idx
    snap = capture_mask_centered(dfrm, cfrm, intr, mesh, R_mesh_cam, mask_shrink_px=MASK_SHRINK_PX)
    if snap is None:
        print("   ❌ captura invàlida"); return idx

    rv_rel_deg, angle_deg, eul_zyx_deg = orientation_relative_to_base(after[3:], R_tcp_base0)
    rx_rel, ry_rel, rz_rel = rv_rel_deg

    tag = f"{axis}{int(round(step_deg)):+d}"
    stem = f"{idx:04d}_{tag}_rx{rx_rel:+.1f}_ry{ry_rel:+.1f}_rz{rz_rel:+.1f}"
    meta = dict(
        axis=axis, delta_deg=float(step_deg),
        tcp_after=[float(x) for x in after.tolist()],
        roll_mode=roll_mode, cam_roll_deg=math.degrees(cam_roll_rad),
        mask_shrink_px=int(MASK_SHRINK_PX),
        rel_rotvec_deg=rv_rel_deg,
        rel_angle_deg=angle_deg,
        euler_zyx_deg=eul_zyx_deg
    )
    meta.update(snap["meta"])
    save_sample(out_dir, stem, snap["rgb"], snap["depth_m"], snap["mask01"], meta)
    print(f"   💾 {stem}  (pts_in_mask={snap['meta']['pts_in_mask']})")
    return idx+1

def mov_and_capture_combo(reader, pipeline, align, mesh, intr,
                          base_cam_R, R_cam_tcp, cam_roll_rad, roll_mode,
                          R_tcp_base0,
                          step_rx_deg, step_ry_deg, out_dir, idx):
    drx = math.radians(step_rx_deg)
    dry = math.radians(step_ry_deg)
    before = reader.tcp_pose()
    target_rot = compose_rot_abs(before[3:], np.array([drx, dry, 0.0], float))
    tgt = np.array([before[0], before[1], before[2], target_rot[0], target_rot[1], target_rot[2]], float)

    def once():
        movel_abs_pose(tgt, a=ACC, v=VEL, r=BLEND)
        time.sleep(DWELL)
        after = reader.tcp_pose()
        return after, robust_delta_mag_deg(before[3:], after[3:])

    after, grob = once()
    if RETRY_ON_SMALL_MOVE and grob < SMALL_MOVE_DEG and MAX_RETRY>0:
        after, grob = once()
    if grob < SMALL_MOVE_DEG:
        print(f"   ❌ combo ({step_rx_deg:+.2f},{step_ry_deg:+.2f})° massa petit, s’omet.")
        return idx

    R_mesh_cam = compute_R_mesh_cam(R_cam_tcp, after[3:], base_cam_R, cam_roll_rad, roll_mode)
    frames = pipeline.wait_for_frames(); frames = align.process(frames)
    dfrm = frames.get_depth_frame(); cfrm = frames.get_color_frame()
    if not dfrm or not cfrm:
        print("   ❌ frame invàlid"); return idx
    snap = capture_mask_centered(dfrm, cfrm, intr, mesh, R_mesh_cam, mask_shrink_px=MASK_SHRINK_PX)
    if snap is None:
        print("   ❌ captura invàlida"); return idx

    rv_rel_deg, angle_deg, eul_zyx_deg = orientation_relative_to_base(after[3:], R_tcp_base0)
    rx_rel, ry_rel, rz_rel = rv_rel_deg

    tag = f"rx{int(round(step_rx_deg)):+d}_ry{int(round(step_ry_deg)):+d}"
    stem = f"{idx:04d}_{tag}_rx{rx_rel:+.1f}_ry{ry_rel:+.1f}_rz{rz_rel:+.1f}"
    meta = dict(
        axis="rx+ry",
        delta_rx_deg=float(step_rx_deg), delta_ry_deg=float(step_ry_deg),
        tcp_after=[float(x) for x in after.tolist()],
        roll_mode=roll_mode, cam_roll_deg=math.degrees(cam_roll_rad),
        mask_shrink_px=int(MASK_SHRINK_PX),
        rel_rotvec_deg=rv_rel_deg,
        rel_angle_deg=angle_deg,
        euler_zyx_deg=eul_zyx_deg
    )
    meta.update(snap["meta"])
    save_sample(out_dir, stem, snap["rgb"], snap["depth_m"], snap["mask01"], meta)
    print(f"   💾 {stem}  (pts_in_mask={snap['meta']['pts_in_mask']})")
    return idx+1

def sweep_from_zero(reader, pipeline, align, mesh, intr,
                    base_cam_R, R_cam_tcp, cam_roll_rad, roll_mode,
                    R_tcp_base0,
                    axis, limit_deg, steps, out_dir, start_idx):
    ensure_dirs(out_dir)
    idx = start_idx

    if limit_deg > 0:
        print(f"\n▶ {axis.upper()} 0 → +{limit_deg:.1f}° en {steps} passos")
        step = (limit_deg / max(1, steps))
        for _ in range(steps):
            idx = mov_and_capture_step(reader, pipeline, align, mesh, intr,
                                       base_cam_R, R_cam_tcp, cam_roll_rad, roll_mode,
                                       R_tcp_base0,
                                       axis, step, out_dir, idx)
        print("  ↩︎ BASE…")
        if not go_to_base(reader, a=1.2, v=0.7, r=0.0, settle_s=0.5, tol_deg=2.0, retries=2):
            print("   ❌ No s’ha pogut garantir BASE. S’intenta una darrera vegada més suau…")
            go_to_base(reader, a=0.8, v=0.5, r=0.0, settle_s=0.7, tol_deg=3.0, retries=1)


    return idx

def do_combos_from_base(reader, pipeline, align, mesh, intr,
                        base_cam_R, R_cam_tcp, cam_roll_rad, roll_mode,
                        R_tcp_base0,
                        rx_targets_deg, ry_targets_deg, steps_combo, out_dir, start_idx):
    ensure_dirs(out_dir)
    idx = start_idx
    for rx in rx_targets_deg:
        for ry in ry_targets_deg:
            if abs(rx) < 1e-6 and abs(ry) < 1e-6:
                continue
            print(f"\n▶ COMBO: RX={rx:+.1f}°, RY={ry:+.1f}°  (steps={steps_combo})")
            drx = rx / max(1, steps_combo)
            dry = ry / max(1, steps_combo)
            for _ in range(steps_combo):
                idx = mov_and_capture_combo(reader, pipeline, align, mesh, intr,
                                            base_cam_R, R_cam_tcp, cam_roll_rad, roll_mode,
                                            R_tcp_base0,
                                            drx, dry, out_dir, idx)
            print("  ↩︎ BASE…")
            if not go_to_base(reader, a=1.2, v=0.7, r=0.0, settle_s=0.5, tol_deg=2.0, retries=2):
                print("   ❌ No s’ha pogut garantir BASE. S’intenta una darrera vegada més suau…")
                go_to_base(reader, a=0.8, v=0.5, r=0.0, settle_s=0.7, tol_deg=3.0, retries=1)

    return idx

# ───────── Utils escenes (reals) ─────────
def depth_to_points3d(depth_m, intr):
    """Converteix depth (m) → Nx3 amb intrínsecs ideals."""
    H, W = depth_m.shape
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    vs, us = np.where(np.isfinite(depth_m) & (depth_m > 0))
    if vs.size == 0:
        return np.empty((0,3), np.float32)
    z = depth_m[vs, us]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    return pts

def place_mask_no_overlap(scene_mask, inst_mask, max_tries=4000, dilate_px=0):
    """Intenta col·locar una màscara instància dins imatge sense solapament amb scene_mask."""
    H, W = scene_mask.shape
    yy, xx = np.where(inst_mask > 0)
    if yy.size == 0: return None
    inst_h, inst_w = inst_mask.shape

    # marge per no sortir
    y_min = 0
    y_max = H - 1 - (inst_h - 1)
    x_min = 0
    x_max = W - 1 - (inst_w - 1)
    if x_max < x_min or y_max < y_min:
        return None

    # dilatació del scene_mask per “gap”
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
        occ = cv2.dilate(scene_mask.astype(np.uint8), k, iterations=1).astype(bool)
    else:
        occ = scene_mask.astype(bool)

    for _ in range(max_tries):
        ty = np.random.randint(y_min, y_max+1)
        tx = np.random.randint(x_min, x_max+1)
        roi = occ[ty:ty+inst_h, tx:tx+inst_w]
        if roi.shape != inst_mask.shape:
            continue
        overlap = roi & (inst_mask>0)
        if not overlap.any():
            return ty, tx
    return None

def compose_depth_min(scene_depth, scene_mask, inst_depth, inst_mask, ty, tx):
    """Fusió depth: al solapar, tria z mínim (més a prop)."""
    ih, iw = inst_mask.shape
    roi_depth = scene_depth[ty:ty+ih, tx:tx+iw]
    roi_mask  = scene_mask[ty:ty+ih, tx:tx+iw]
    im = inst_mask>0

    # On hi ha instància:
    #  - si roi_mask=0 → posa inst_depth
    #  - si roi_mask=1 → min(scene_depth, inst_depth)
    target = roi_depth.copy()
    # on no hi havia res:
    add_idxs = im & (~roi_mask.astype(bool))
    target[add_idxs] = inst_depth[add_idxs]

    # on ja hi havia profunditat:
    both = im & roi_mask.astype(bool)
    sd = roi_depth[both]
    idp = inst_depth[both]
    # si algun és 0 o NaN, tracta-ho
    sd[~np.isfinite(sd) | (sd<=0)] = np.inf
    idp[~np.isfinite(idp) | (idp<=0)] = np.inf
    target[both] = np.minimum(sd, idp)

    scene_depth[ty:ty+ih, tx:tx+iw] = target
    scene_mask[ty:ty+ih, tx:tx+iw] |= (inst_mask>0)

def generate_scenes_from_run(out_root: Path, Zc_m: float,
                             scenes_per_dist: int, min_objs: int, max_objs: int,
                             place_gap_px: int):
    """Crea scenes/dist_<Zc>m/ combinant depth/mask de la carpeta run i intrínsecs ideals."""
    # Carrega totes les captures d’aquest run
    depth_dir = out_root / "depth_npy"
    mask_dir  = out_root / "mask_npy"
    meta_dir  = out_root / "meta"
    if not depth_dir.exists() or not mask_dir.exists() or not meta_dir.exists():
        print("⚠️ No s’han trobat subcarpetes de captures (depth_npy/mask_npy/meta).")
        return

    files = sorted([p for p in depth_dir.glob("*.npy")])
    if not files:
        print("⚠️ Sense depth .npy per compondre escenes.")
        return

    # Intrínsecs ideals i mida
    # (els assumim constants a totes les captures del run)
    first_meta = next(meta_dir.glob("*.json"), None)
    if first_meta is None:
        print("⚠️ Sense meta per extreure intrínsecs.")
        return
    intr = json.loads(first_meta.read_text(encoding="utf-8")).get("intrinsics_ideal", None)
    if intr is None:
        print("⚠️ Meta sense intrinsics_ideal.")
        return
    W, H = int(intr["W"]), int(intr["H"])

    # Carrega llistes depth/mask
    samples = []
    for f in files:
        stem = f.stem
        mpath = mask_dir / f"{stem}.npy"
        if not mpath.exists(): continue
        d = np.load(f).astype(np.float32)   # depth (m) original (no enmascarat)
        m = np.load(mpath)                  # booleà
        # Ens quedem amb la part enmascarada amb el valor de profunditat original
        # (més robust si profunditat va amb soroll fora màscara)
        dm = d.copy()
        dm[~m] = 0.0
        samples.append((stem, dm, m.astype(np.uint8)))

    if not samples:
        print("⚠️ Sense parelles depth/mask vàlides.")
        return

    # Carpeta sortida scenes/dist_XX.XXm
    dist_tag = f"dist_{Zc_m:.2f}m"
    out_dist = out_root / "scenes" / dist_tag
    for sub in ("points3d", "depth_npy", "mask_npy", "depth_png", "mask_png", "meta"):
        (out_dist/sub).mkdir(parents=True, exist_ok=True)
    gt_lines = []

    print(f"🎬 Generant escenes reals a {out_dist} …")
    for i in range(scenes_per_dist):
        k = random.randint(min_objs, max_objs)
        chosen = random.sample(samples, k=min(k, len(samples)))

        scene_depth = np.zeros((H, W), np.float32)  # 0 = buit
        scene_mask  = np.zeros((H, W), np.uint8)

        placed = 0
        for (stem, d, m) in chosen:
            # retalla bounding box per accelerar col·locació
            ys, xs = np.where(m > 0)
            if ys.size == 0: continue
            y0, y1 = ys.min(), ys.max()+1
            x0, x1 = xs.min(), xs.max()+1
            inst_mask  = m[y0:y1, x0:x1]
            inst_depth = d[y0:y1, x0:x1]

            pos = place_mask_no_overlap(scene_mask, inst_mask, max_tries=2000, dilate_px=place_gap_px)
            if pos is None:
                continue
            ty, tx = pos
            compose_depth_min(scene_depth, scene_mask, inst_depth, inst_mask, ty, tx)
            placed += 1

        # genera fitxers si com a mínim hi ha 1 objecte col·locat
        if placed == 0:
            # repeteix intent amb menys gap
            continue

        # Converteix 0 → NaN per coherència amb dataset sintètic
        depth_out = scene_depth.copy()
        depth_out[~np.isfinite(depth_out) | (depth_out <= 0)] = np.nan

        # PNGs
        finite = np.isfinite(depth_out)
        if finite.any():
            d = depth_out.copy()
            dmin, dmax = np.nanmin(d), np.nanmax(d)
            img = np.zeros_like(d, np.uint8)
            if dmax - dmin > 1e-9:
                norm = (d - dmin) / (dmax - dmin)
                norm = 1.0 - norm
                norm[~finite] = 0.0
                img = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        else:
            img = np.zeros((H, W), np.uint8)

        stem = f"scene_{i:04d}_N{placed}"
        write_png_gray(img, str(out_dist/"depth_png"/f"{stem}.png"))
        write_png_gray((scene_mask*255).astype(np.uint8), str(out_dist/"mask_png"/f"{stem}.png"))

        # NPY
        np.save(out_dist/"depth_npy"/f"{stem}.npy", depth_out.astype(np.float32))
        np.save(out_dist/"mask_npy"/f"{stem}.npy", (scene_mask>0))

        # points3d
        pts = depth_to_points3d(depth_out, intr)
        np.save(out_dist/"points3d"/f"{stem}.npy", pts.astype(np.float32))

        # meta
        meta = {
            "type": "scene_real",
            "distance_m": float(Zc_m),
            "intrinsics": {k: (float(v) if isinstance(v,(int,float)) else v) for k,v in intr.items()},
            "points_N": int(pts.shape[0]),
            "placed": int(placed)
        }
        (out_dist/"meta"/f"meta_{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        gt_lines.append(f"{stem},{placed}")

        if (i+1) % max(1, scenes_per_dist//10) == 0:
            print(f"  scenes {i+1}/{scenes_per_dist}")

    (out_dist/"gt_counts.csv").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    print(f"✔ Escenes reals generades ({len(gt_lines)}) a {out_dist}")

# ───────── Verificació de “anar a BASE” ─────────
def _deg2rad(x): return math.radians(x)

def joints_close(q_now, q_ref, tol_deg=2.0):
    """Comprova |q_now - q_ref| < tol for all joints (en graus)."""
    q_now = np.asarray(q_now, float)
    q_ref = np.asarray(q_ref, float)
    tol = math.radians(tol_deg)
    return np.all(np.abs(q_now - q_ref) <= tol)

def wait_until_reached_base(reader, q_ref_rad, timeout_s=6.0, tol_deg=2.0, poll_hz=50):
    """Espera fins que els joints siguin a prop de q_ref (BASE) o s’acabi el timeout."""
    t0 = time.time()
    dt = 1.0 / max(1, poll_hz)
    while (time.time() - t0) < timeout_s:
        try:
            q_now = reader.rtde.getActualQ()  # 6 joints (rad)
        except Exception:
            time.sleep(dt); continue
        if joints_close(q_now, q_ref_rad, tol_deg=tol_deg):
            return True
        time.sleep(dt)
    return False

def go_to_base(reader, a=1.2, v=0.7, r=0.0, settle_s=0.5, tol_deg=2.0, retries=2):
    """
    Envia movej(J_BASE) i comprova amb RTDE que hi ha arribat (joints).
    Si no, reintenta fins a 'retries' vegades.
    """
    ok = False
    last_err = None
    for attempt in range(retries + 1):
        try:
            movej(J_BASE, a=a, v=v, r=r)
        except Exception as e:
            last_err = e
        # petit temps perquè comenci el moviment
        time.sleep(settle_s)
        # espera activa a que estigui a BASE
        ok = wait_until_reached_base(reader, J_BASE, timeout_s=BASE_EXTRA, tol_deg=tol_deg, poll_hz=50)
        if ok:
            break
        else:
            print(f"   ⚠️  No s’ha assolit BASE (int {attempt+1}/{retries+1}). Reintentant…")
    if not ok and last_err:
        print(f"   ⚠️  Error movej cap a BASE: {last_err}")
    return ok


# ───────── MAIN ─────────
def main():
    ap = argparse.ArgumentParser("UR3 + RealSense + STL-mask (centrada) — captura + ESCENES reals")
    ap.add_argument("--stl", required=True, help="Fitxer STL")
    ap.add_argument("--out_dir", required=True, help="Carpeta arrel de sortida")
    ap.add_argument("--stl_units", choices=["m","cm","mm"], default="m", help="Unitats de l’STL")
    ap.add_argument("--cam_roll_deg", type=float, default=-90.0, help="roll físic càmera (deg)")
    ap.add_argument("--roll_mode", choices=["pre_map","post_world","post_cam"], default="post_world")
    ap.add_argument("--base_y_sign", type=int, choices=[-1,1], default=-1, help="+X_tcp→+Z_cam (±90° sobre Y)")
    ap.add_argument("--steps_x", type=int, default=STEPS_X)
    ap.add_argument("--steps_y", type=int, default=STEPS_Y)
    ap.add_argument("--rx_min", type=float, default=RX_MIN_DEG)
    ap.add_argument("--rx_max", type=float, default=RX_MAX_DEG)
    ap.add_argument("--ry_min", type=float, default=RY_MIN_DEG)
    ap.add_argument("--ry_max", type=float, default=RY_MAX_DEG)
    ap.add_argument("--rx_targets", type=float, nargs="*", default=RX_TARGETS_DEG)
    ap.add_argument("--ry_targets", type=float, nargs="*", default=RY_TARGETS_DEG)
    ap.add_argument("--steps_combo", type=int, default=STEPS_COMBO)

    # Paràmetres de síntesi d’ESCENES reals (post-captura)
    ap.add_argument("--scenes_per_dist", type=int, default=120)
    ap.add_argument("--min_objs", type=int, default=3)
    ap.add_argument("--max_objs", type=int, default=18)
    ap.add_argument("--place_gap_px", type=int, default=3, help="dilatació (px) per evitar solapaments")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    # STL → metres, normals, recentrar
    mesh = o3d.io.read_triangle_mesh(args.stl)
    if mesh.is_empty(): raise RuntimeError("No s'ha pogut llegir l'STL")
    if args.stl_units == "mm": mesh.scale(0.001, center=(0,0,0))
    elif args.stl_units == "cm": mesh.scale(0.01, center=(0,0,0))
    mesh.compute_vertex_normals()
    ctr0 = mesh.get_center()
    mesh.translate((-ctr0[0], -ctr0[1], -ctr0[2]))

    # RealSense
    pipeline = rs.pipeline(); cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(cfg); align = rs.align(rs.stream.color)

    # Robot: BASE
    reader = PoseReader(IP, csv_path="movement_log_simple.csv", verbose=True)
    print("➡️  BASE…"); movej(J_BASE, a=1.2, v=0.7); time.sleep(BASE_EXTRA)

    # Hand–eye base i roll
    base_sign = -1 if args.base_y_sign < 0 else 1
    R_cam_tcp_base = Ry(base_sign * (math.pi/2))  # +X_tcp → +Z_cam
    cam_roll_rad = math.radians(args.cam_roll_deg)
    base_tcp_pose = reader.tcp_pose()
    R_tcp_base0 = rotvec_to_R(base_tcp_pose[3:])
    if args.roll_mode == "pre_map":
        base_cam_R = (Rz(cam_roll_rad) @ R_cam_tcp_base) @ R_tcp_base0
    else:
        base_cam_R = R_cam_tcp_base @ R_tcp_base0

    # Vista de centrament → obté intrínsecs ideals + distància inicial
    intr = preview_center_overlay(reader, pipeline, profile, align, mesh,
                                  R_cam_tcp_base, base_cam_R, cam_roll_rad, args.roll_mode,
                                  init_shrink_px=3)
    if intr is None:
        try: pipeline.stop()
        except: pass
        return

    Zc_m = float(intr["Zc_init"])
    print(f"\n[i] Distància única per a scenes reals: {Zc_m:.3f} m")

    # Carpeta de sortida (run)
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_dir) / f"run_{run_ts}"
    os.makedirs(out_root, exist_ok=True)
    ensure_dirs(str(out_root))

    print("\nCONFIG:")
    print(f"  RX lim: [{args.rx_min:.1f}°, +{args.rx_max:.1f}°]   RY lim: [{args.ry_min:.1f}°, +{args.ry_max:.1f}°]")
    print(f"  STEPS_X={args.steps_x}, STEPS_Y={args.steps_y}   COMBO_STEPS={args.steps_combo}")
    print(f"  ACC={ACC}, VEL={VEL}, DWELL={DWELL}s, BASE_EXTRA={BASE_EXTRA}s")
    print(f"  roll={args.cam_roll_deg:+.0f}°, mode={args.roll_mode}")
    print(f"  ESCENES: {args.scenes_per_dist}  objs:[{args.min_objs},{args.max_objs}]  gap={args.place_gap_px}px\n")

    idx = 0
    # RY bàsics
    idx = sweep_from_zero(reader, pipeline, align, mesh, intr,
                          base_cam_R, R_cam_tcp_base, cam_roll_rad, args.roll_mode,
                          R_tcp_base0,
                          axis="ry", limit_deg=abs(args.ry_max), steps=args.steps_y, out_dir=str(out_root), start_idx=idx)
    idx = sweep_from_zero(reader, pipeline, align, mesh, intr,
                          base_cam_R, R_cam_tcp_base, cam_roll_rad, args.roll_mode,
                          R_tcp_base0,
                          axis="ry", limit_deg=-abs(args.ry_min), steps=args.steps_y, out_dir=str(out_root), start_idx=idx)
    # RX bàsics
    idx = sweep_from_zero(reader, pipeline, align, mesh, intr,
                          base_cam_R, R_cam_tcp_base, cam_roll_rad, args.roll_mode,
                          R_tcp_base0,
                          axis="rx", limit_deg=abs(args.rx_max), steps=args.steps_x, out_dir=str(out_root), start_idx=idx)
    idx = sweep_from_zero(reader, pipeline, align, mesh, intr,
                          base_cam_R, R_cam_tcp_base, cam_roll_rad, args.roll_mode,
                          R_tcp_base0,
                          axis="rx", limit_deg=-abs(args.rx_min), steps=args.steps_x, out_dir=str(out_root), start_idx=idx)

    # COMBINACIONS
    print("\n🎛  Combinacions RX×RY:")
    idx = do_combos_from_base(reader, pipeline, align, mesh, intr,
                              base_cam_R, R_cam_tcp_base, cam_roll_rad, args.roll_mode,
                              R_tcp_base0,
                              rx_targets_deg=args.rx_targets, ry_targets_deg=args.ry_targets,
                              steps_combo=args.steps_combo, out_dir=str(out_root), start_idx=idx)

    # Atura càmera
    try: pipeline.stop()
    except: pass
    dash_stop()

    # ───── Genera ESCENES reals (mateixa estructura que el generador) ─────
    try:
        generate_scenes_from_run(out_root, Zc_m,
                                 scenes_per_dist=args.scenes_per_dist,
                                 min_objs=args.min_objs, max_objs=args.max_objs,
                                 place_gap_px=args.place_gap_px)
    except Exception as e:
        print(f"⚠️ Error generant scenes reals: {e}")

    print(f"\n✅ Dataset creat: {out_root}")

if __name__ == "__main__":
    main()
