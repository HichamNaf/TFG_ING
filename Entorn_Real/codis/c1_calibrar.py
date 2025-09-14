#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UR3 + RealSense + STL-mask (CENTRADA) — Mode calibratge amb màscara solapada a la RGB.

• La màscara STL es renderitza amb R_mesh_in_cam i translació (0,0,Zc) i
  es solapa a la imatge RGB amb opacitat ajustable.
• Sense guardats; pensat per ajustar el TCP “a ull” movent en passos petits.

Tecles:
  q/ESC: sortir
  b    : anar a BASE
  i/k  : +rx / -rx   (graus per pas)
  j/l  : +ry / -ry
  u/o  : +rz / -rz
  [/]  : -/+ erosió màscara (px)
  ,/.  : -/+ pas angular (graus)
  m    : alterna màscara plena / només contorn
  p/P  : -/+ opacitat del solapat
  SPACE: refresc sense moure
"""

import argparse, math, time, socket
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
BASE_EXTRA = 3.5

J_BASE = [math.radians(0.0),
          math.radians(-111.0),
          math.radians(-88.0),
          math.radians(-70.0),
          math.radians( 90.0),
          math.radians(  0.0)]

# ───────── Càmera ─────────
WIDTH, HEIGHT, FPS = 640, 480, 30
MIN_PTS_CROP  = 50   # només per avisos (no bloqueja)

# ───────── Àlgebra rotacions ─────────
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

# ───────── RTDE ─────────
class PoseReader:
    def __init__(self, ip):
        self.rtde = RTDEReceiveInterface(ip)
    def tcp_pose(self):
        return np.array(self.rtde.getActualTCPPose(), float)

# ───────── RealSense ─────────
def intr_from_rs(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return dict(fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy, W=intr.width, H=intr.height)

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

def shrink_mask(mask01, px):
    if px <= 0: return mask01
    k = 2*int(px)+1
    kernel = np.ones((k,k), np.uint8)
    return cv2.erode((mask01*255).astype(np.uint8), kernel, iterations=1) // 255

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

# ───────── Jog ─────────
def jog_rotate(reader: PoseReader, d_rx, d_ry, d_rz, a=ACC, v=VEL):
    before = reader.tcp_pose()
    target_rot = compose_rot_abs(before[3:], np.array([d_rx, d_ry, d_rz], float))
    tgt = np.array([before[0], before[1], before[2], target_rot[0], target_rot[1], target_rot[2]], float)
    movel_abs_pose(tgt, a=a, v=v, r=BLEND)
    time.sleep(0.6)

# ───────── Overlay ─────────
def make_overlay_rgb(color_bgr, mask01, alpha, fill_mode=True):
    """Solapa la màscara a la RGB: ple (alpha) o només contorn."""
    vis = color_bgr.copy()
    if mask01 is None:
        return vis
    if fill_mode:
        overlay = vis.copy()
        overlay[mask01>0] = (0,255,0)
        vis = cv2.addWeighted(overlay, float(alpha), vis, 1.0-float(alpha), 0.0)
    else:
        edges = cv2.Canny((mask01*255).astype(np.uint8), 50, 150)
        vis[edges>0] = (0,255,255)
    return vis

# ───────── MAIN ─────────
def main():
    ap = argparse.ArgumentParser("UR3 + RealSense + STL-mask (calibratge, màscara solapada a RGB)")
    ap.add_argument("--stl", required=True)
    ap.add_argument("--stl_units", choices=["m","cm","mm"], default="m")
    ap.add_argument("--base_y_sign", type=int, choices=[-1,1], default=-1,
                    help="Ry(±90°) base per mapejar +X_tcp → +Z_cam")
    ap.add_argument("--cam_roll_deg", type=float, default=-90.0)
    ap.add_argument("--roll_mode", choices=["pre_map","post_world","post_cam"], default="post_world")
    ap.add_argument("--mask_shrink_px", type=int, default=3)
    ap.add_argument("--step_deg", type=float, default=2.0, help="pas angular del jog (graus)")
    ap.add_argument("--alpha", type=float, default=0.40, help="opacitat de la màscara [0..1]")
    args = ap.parse_args()

    # STL → metres, normals i RECENTRAR al centre geomètric
    mesh = o3d.io.read_triangle_mesh(args.stl)
    if mesh.is_empty():
        raise RuntimeError("No s'ha pogut llegir l'STL")
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
    intr = intr_from_rs(profile)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # Robot base + RTDE
    reader = PoseReader(IP)
    print("➡️  BASE…"); movej(J_BASE, a=1.2, v=0.7); time.sleep(BASE_EXTRA)

    # Hand–eye base i roll
    base_sign = -1 if args.base_y_sign < 0 else 1
    R_cam_tcp_base = Ry(base_sign * (math.pi/2))  # +X_tcp → +Z_cam
    cam_roll_rad = math.radians(args.cam_roll_deg)
    base_tcp = reader.tcp_pose(); R_tcp_base = rotvec_to_R(base_tcp[3:])
    if args.roll_mode == "pre_map":
        base_cam_R = (Rz(cam_roll_rad) @ R_cam_tcp_base) @ R_tcp_base
    else:
        base_cam_R = R_cam_tcp_base @ R_tcp_base

    # Estat UI
    mask_shrink_px = int(args.mask_shrink_px)
    step_deg = float(args.step_deg)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    fill_mode = True  # True: ple, False: contorn

    print("\nControls: q/ESC sortir | b BASE | i/k +/−rx | j/l +/−ry | u/o +/−rz | [/] shrink | ,/. step | m mode | p/P opacitat | SPACE refresc")
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        dfrm = frames.get_depth_frame(); cfrm = frames.get_color_frame()
        if not dfrm or not cfrm:
            cv2.waitKey(1); continue

        z16 = np.asanyarray(dfrm.get_data())
        color = np.asanyarray(cfrm.get_data())
        H,W = z16.shape; cx,cy = W//2, H//2

        # Z central robusta
        patch = z16[max(0,cy-3):min(H,cy+4), max(0,cx-3):min(W,cx+4)]
        valid = patch[patch > 0]
        Zc = float(valid.mean() * depth_scale) if valid.size > 0 else float('nan')

        # Rotació mesh→càmera a partir del TCP actual
        tcp = reader.tcp_pose()
        R_mesh_cam = compute_R_mesh_cam(R_cam_tcp_base, tcp[3:], base_cam_R, cam_roll_rad, args.roll_mode)

        mask01 = None
        if np.isfinite(Zc):
            mask01 = raycast_mask_stl_in_cam(
                mesh, intr['fx'], intr['fy'], intr['cx'], intr['cy'],
                W, H, dist_m=Zc, R_mesh_in_cam=R_mesh_cam
            ).astype(np.uint8)
            if mask_shrink_px > 0:
                mask01 = shrink_mask(mask01, mask_shrink_px)

        # Solapat directe a la RGB
        vis = make_overlay_rgb(color, mask01, alpha, fill_mode=fill_mode)

        # HUD
        cv2.drawMarker(vis, (cx,cy), (0,0,0), cv2.MARKER_CROSS, 20, 2)
        cv2.drawMarker(vis, (cx,cy), (255,255,255), cv2.MARKER_CROSS, 14, 2)
        txt = f"Zc: {Zc:.3f} m  | step: {step_deg:.1f} deg  | shrink: {mask_shrink_px}px  | mode: {'PLE' if fill_mode else 'CONTORN'}  | alpha: {alpha:.2f}"
        cv2.putText(vis, txt, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,255,0), 2, cv2.LINE_AA)

        # Depth en miniatura a la cantonada
        d8 = cv2.convertScaleAbs(z16, alpha=0.03)
        d8 = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
        small = cv2.resize(d8, (W//3, H//3))
        vis[0:small.shape[0], -small.shape[1]:] = small

        # Avís de punts dins màscara
        if mask01 is not None:
            pts_in = int(np.sum((mask01>0) & (z16>0)))
            if pts_in < MIN_PTS_CROP:
                cv2.putText(vis, "ATENCIO: pocs punts dins la mascara", (10,55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow("Calibratge UR3 + RealSense (RGB + màscara)", vis)
        k = cv2.waitKey(1) & 0xFF

        if k in (27, ord('q')):
            break
        elif k == ord('b'):
            print("➡️  BASE…"); movej(J_BASE, a=1.2, v=0.7); time.sleep(BASE_EXTRA)
        elif k == ord('i'):  # +rx
            jog_rotate(reader, math.radians(step_deg), 0.0, 0.0)
        elif k == ord('k'):  # -rx
            jog_rotate(reader, -math.radians(step_deg), 0.0, 0.0)
        elif k == ord('j'):  # +ry
            jog_rotate(reader, 0.0, math.radians(step_deg), 0.0)
        elif k == ord('l'):  # -ry
            jog_rotate(reader, 0.0, -math.radians(step_deg), 0.0)
        elif k == ord('u'):  # +rz
            jog_rotate(reader, 0.0, 0.0, math.radians(step_deg))
        elif k == ord('o'):  # -rz
            jog_rotate(reader, 0.0, 0.0, -math.radians(step_deg))
        elif k == ord('['):
            mask_shrink_px = max(0, mask_shrink_px - 1)
        elif k == ord(']'):
            mask_shrink_px += 1
        elif k == ord(','):
            step_deg = max(0.1, step_deg - 0.5)
        elif k == ord('.'):
            step_deg += 0.5
        elif k == ord('m'):
            fill_mode = not fill_mode
        elif k == ord('p'):
            alpha = max(0.0, alpha - 0.05)
        elif k == ord('P'):
            alpha = min(1.0, alpha + 0.05)
        elif k == ord(' '):
            pass  # refresc sense moure

    try: pipeline.stop()
    except: pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
