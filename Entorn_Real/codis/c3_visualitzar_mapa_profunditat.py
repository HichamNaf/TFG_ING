# viz_npy.py
# Visualitza .npy com a núvol de punts (N×3 o N×6) o com a profunditat (H×W).
# Ús:
#   python viz_npy.py ruta/al/fitxer.npy [--type auto|points|depth]
#                      [--fov 60] [--voxel 0.0] [--save out.png]
# Exemple:
#   python viz_npy.py scenes/dist_0.30m/points3d/scene_0001.npy --voxel 0.003
#   python viz_npy.py scenes/dist_0.30m/depth_npy/scene_0001.npy --type depth --fov 60 --save cap.png

import argparse, numpy as np, open3d as o3d, math
from matplotlib import cm

def depth_to_points(depth, fov_deg=60.0):
    H, W = depth.shape
    cx, cy = (W-1)/2.0, (H-1)/2.0
    fx = (W/2.0) / math.tan(math.radians(fov_deg)/2.0)
    fy = fx
    vs, us = np.where(np.isfinite(depth) & (depth > 0))
    if vs.size == 0: return np.empty((0,3), np.float32)
    z = depth[vs, us].astype(np.float32)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)

def color_by_z(pts):
    if pts.size == 0: return np.zeros_like(pts, dtype=np.float32)
    z = pts[:,2]
    zmin, zmax = np.percentile(z, 1), np.percentile(z, 99)
    z = np.clip((z - zmin) / max(1e-9, (zmax - zmin)), 0, 1)
    return np.array(cm.jet(z))[:,:3].astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy", help=".npy amb Nx3/Nx6 o HxW (profunditat amb NaN)")
    ap.add_argument("--type", choices=["auto","points","depth"], default="auto")
    ap.add_argument("--fov", type=float, default=60.0, help="FOV per a profunditat")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel size per fer downsample")
    ap.add_argument("--save", default=None, help="guarda captura a PNG (offscreen)")
    args = ap.parse_args()

    arr = np.load(args.npy)
    mode = args.type
    if mode == "auto":
        mode = "depth" if arr.ndim == 2 else "points"

    # Construeix point cloud
    if mode == "depth":
        pts = depth_to_points(arr.astype(np.float32), fov_deg=args.fov)
        cols = color_by_z(pts)
    else:
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[1] not in (3,6):
            raise SystemExit("Esperava Nx3 o Nx6 per a points.")
        pts = arr[:,:3].astype(np.float32)
        cols = arr[:,3:6].astype(np.float32) if arr.shape[1] == 6 else color_by_z(pts)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if pts.size and args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    if cols.size:
        # Si hi ha menys colors que punts (després del voxel), recolorim per z
        if np.asarray(pcd.points).shape[0] != cols.shape[0]:
            cols = color_by_z(np.asarray(pcd.points))
        pcd.colors = o3d.utility.Vector3dVector(cols)

    if args.save:
        # Render offscreen i desa captura
        W, H = 1280, 720
        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        m = o3d.visualization.rendering.MaterialRecord()
        m.shader = "defaultUnlit"
        renderer.scene.set_background([1,1,1,1])
        renderer.scene.add_geometry("pcd", pcd, m)
        bounds = renderer.scene.bounding_box
        center = bounds.get_center()
        extent = np.linalg.norm(bounds.get_extent())
        cam = renderer.scene.camera
        cam.look_at(center, center + [0,0,1], [0,-1,0])
        cam.set_projection(60.0, W/H, 0.01, 100.0, o3d.visualization.rendering.Camera.Projection.PERSPECTIVE)
        renderer.scene.camera.set_zoom(1.7/ max(1e-6, extent))
        img = renderer.render_to_image()
        o3d.io.write_image(args.save, img)
        print(f"✔ Captura desada a {args.save}")
    else:
        o3d.visualization.draw_geometries([pcd], window_name="NPY viewer", width=1280, height=720)

if __name__ == "__main__":
    main()
