#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, sys, math
from pathlib import Path
import numpy as np
import open3d as o3d

def depth_to_points3d(depth_m, intr):
    H, W = depth_m.shape
    fx, fy = float(intr["fx"]), float(intr["fy"])
    cx, cy = float(intr["cx"]), float(intr["cy"])
    vs, us = np.where(np.isfinite(depth_m) & (depth_m > 0))
    if vs.size == 0:
        return np.empty((0,3), np.float32)
    z = depth_m[vs, us]
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    return np.stack([x, y, z], axis=1).astype(np.float32)

def try_load_points_npy(path: Path):
    """Intenta interpretar un .npy com (a) Nx3 punts o (b) depth map amb meta al costat."""
    arr = np.load(path)
    # Cas 1: ja són punts Nx3
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.astype(np.float32), None
    # Cas 2: és un depth map (H,W) → requeriria intrinsics al meta germà
    meta_json = (path.parent / "meta" / (path.stem + ".json"))
    if meta_json.exists():
        md = json.loads(meta_json.read_text(encoding="utf-8"))
        intr = md.get("intrinsics_ideal") or md.get("intrinsics")
        if intr is None:
            raise KeyError(f"Meta sense intrinsics a {meta_json}")
        pts = depth_to_points3d(arr.astype(np.float32), intr)
        dist = float(md.get("Zc_m", md.get("distance_m", "nan")))
        if not math.isfinite(dist): dist = None
        return pts, dist
    return None, None

def load_points_from_scene(scene_dir: Path):
    """Carrega punts des de diferents layouts possibles."""
    name = scene_dir.name

    # 1) points3d/<stem>.npy
    p = scene_dir / "points3d" / f"{name}.npy"
    if p.exists():
        pts = np.load(p).astype(np.float32)
        dist = None
        meta_file = scene_dir / "meta" / f"meta_{name}.json"
        if meta_file.exists():
            md = json.loads(meta_file.read_text(encoding="utf-8"))
            dist = float(md.get("distance_m", md.get("Zc_m", "nan")))
            if not math.isfinite(dist): dist = None
        return pts, dist

    # 2) depth_npy/<stem>.npy + meta/<stem>.json
    d = scene_dir / "depth_npy" / f"{name}.npy"
    m = scene_dir / "meta" / f"{name}.json"
    if d.exists() and m.exists():
        md = json.loads(m.read_text(encoding="utf-8"))
        intr = md.get("intrinsics_ideal") or md.get("intrinsics")
        if intr is None:
            raise KeyError("El meta no conté intrinsics.")
        pts = depth_to_points3d(np.load(d).astype(np.float32), intr)
        dist = float(md.get("Zc_m", md.get("distance_m", "nan")))
        if not math.isfinite(dist): dist = None
        return pts, dist

    # 3) .npy DIRECTE a la carpeta (p.ex. shot_0001.npy o points3d.npy)
    direct = [scene_dir / f"{name}.npy", scene_dir / "points3d.npy"]
    for cand in direct:
        if cand.exists():
            pts, dist = try_load_points_npy(cand)
            if pts is not None:
                return pts, dist

    # 4) Agafa el primer .npy que trobi dins la carpeta
    any_npy = sorted(scene_dir.glob("*.npy"))
    if any_npy:
        pts, dist = try_load_points_npy(any_npy[0])
        if pts is not None:
            return pts, dist

    raise FileNotFoundError(
        f"No s'han trobat punts a {scene_dir}. "
        "Espera: points3d/<stem>.npy, o depth_npy/<stem>.npy + meta/<stem>.json, "
        "o un .npy directe (Nx3 o depth)."
    )

def pick_eps_minpts(tuned_dir: Path, dist_m: float | None):
    info = json.loads((tuned_dir / "tuned_info.json").read_text(encoding="utf-8"))
    if not info:
        raise RuntimeError("tuned_info.json buit.")
    if dist_m is None:
        k = sorted(info.keys())[0]
        v = info[k]
        return float(v["eps"]), int(v["min_points"])
    keys = [float(k) for k in info.keys()]
    nearest = min(keys, key=lambda x: abs(x - dist_m))
    v = info[f"{nearest:.2f}"]
    return float(v["eps"]), int(v["min_points"])

def colorize_by_labels(pts, labels):
    rng = np.random.RandomState(42)
    max_lbl = labels.max() if labels.size else -1
    palette = rng.rand(max(0, max_lbl) + 1, 3)
    colors = np.zeros((pts.shape[0], 3), np.float32) + 0.6  # gris soroll
    for lbl in range(0, max_lbl + 1):
        colors[labels == lbl] = palette[lbl]
    return colors

def highest_cluster_label(pts, labels):
    best_lbl, best_z = -1, -np.inf
    for lbl in np.unique(labels):
        if lbl < 0: continue
        z_mean = float(pts[labels == lbl, 2].mean())
        if z_mean > best_z:
            best_z, best_lbl = z_mean, lbl
    return best_lbl

def main():
    ap = argparse.ArgumentParser("Visualitza DBSCAN sobre escena real")
    ap.add_argument("--scene_dir", required=True)
    ap.add_argument("--tuned_dir", required=True)
    ap.add_argument("--override", type=float, nargs=2, metavar=("EPS","MINPTS"))
    ap.add_argument("--dist", type=float, help="Força distància (m) per seleccionar entrada de tuned_info.json")
    args = ap.parse_args()

    scene_dir = Path(args.scene_dir)
    tuned_dir = Path(args.tuned_dir)
    if not (tuned_dir / "tuned_info.json").exists():
        sys.exit(f"❌ Falta {tuned_dir/'tuned_info.json'}")

    pts, dist_m_found = load_points_from_scene(scene_dir)
    dist_m = args.dist if args.dist is not None else dist_m_found
    if pts.shape[0] < 30:
        sys.exit("❌ Núvol massa petit o buit.")

    if args.override:
        eps, minpts = float(args.override[0]), int(round(args.override[1]))
    else:
        eps, minpts = pick_eps_minpts(tuned_dir, dist_m)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.array(pcd.cluster_dbscan(eps=float(eps), min_points=int(minpts), print_progress=False))
    colors = colorize_by_labels(pts, labels)
    top = highest_cluster_label(pts, labels)
    if top >= 0:
        colors[labels == top] = np.array([1.0, 0.0, 0.0], np.float32)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    uniq, cnts = np.unique(labels, return_counts=True)
    print(f"\nEscena: {scene_dir}")
    print(f"Distància usada per a tuned_info: {dist_m}")
    print(f"DBSCAN → eps={eps:.6f}  minPts={minpts}  | clusters={int(labels.max())+1 if labels.size and labels.max()>=0 else 0}")
    for u, c in zip(uniq, cnts):
        print(f"  etiqueta {int(u):>2}: {c} punts")

    o3d.visualization.draw_geometries([pcd], window_name="DBSCAN — clusters (vermell = clúster superior)")

if __name__ == "__main__":
    main()
