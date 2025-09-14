#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenament per trobar (eps, min_points) per distància i (opcionalment) entrenar models supervisats.

Dues utilitats principals:
  1) TUNING per distància (SCENES): per a cada escena (núvol de punts) i GT de #peces,
     busquem (eps*, minPts*) que minimitzi |K_pred - K_gt|.
     - --method binary: cerca binària d'eps dins d'un interval [lo, hi] escalat amb d4_med (dist al 4t veí).
     - --method grid:   cerca discreta d'eps sobre una graella dins [lo, hi].

  2) (OPCIONAL) ENTRENAMENT de models per predir eps i minPts a partir de *features* geomètriques.
     - Model per defecte: RandomForestRegressor (robust i sense necessitat d'escalar).
     - Altres: MLP, Ridge, KNN (amb StandardScaler intern).

Traçabilitat:
  - Es desa un CSV amb resultats per escena (tuned_per_scene.csv).
  - Es desa un tuned_info.json amb mitjanes per distància.
  - Si s'entrenen models: .joblib + feature_names.txt + metrics.txt

Comentaris en català.
"""

import argparse, json, math, re, random
from pathlib import Path
import numpy as np
import open3d as o3d

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
import joblib

# ----------------------- UTILITATS DE FITXERS / LECTURA -----------------------

def parse_distance_from_dirname(dirname: str) -> float:
    """ Extreu el valor en metres d'una ruta del tipus '.../dist_0.50m' """
    m = re.search(r"dist_([0-9.]+)m", dirname)
    return float(m.group(1)) if m else float("nan")

def list_scene_files(dist_dir: Path):
    """ Llista de fitxers .npy dins dist_dir/points3d """
    p = dist_dir / "points3d"
    return sorted(p.glob("*.npy")) if p.exists() else []

def load_gt_counts(dist_dir: Path) -> dict:
    """ Llegeix gt_counts.csv → {stem: count} """
    gt_path = dist_dir / "gt_counts.csv"
    if not gt_path.exists():
        return {}
    mapping = {}
    for line in gt_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        mapping[parts[0].strip()] = int(parts[1].strip())
    return mapping

def discover_distance_dirs(scenes_root: Path):
    """ Troba carpetes 'dist_*.m' dins SCENES root """
    return sorted([p for p in (scenes_root).glob("dist_*m") if p.is_dir()])

# ----------------------- FEATURES GEOMÈTRIQUES -----------------------

def aabb_features(pts: np.ndarray):
    mn = pts.min(0); mx = pts.max(0)
    ext = mx - mn
    vol = float(ext[0]*ext[1]*ext[2] + 1e-12)
    return ext, vol

def knn_stats(pts: np.ndarray, k_list=(1,2,3,4,8,16), maxN=50000):
    """ Estadístiques de distància al k-è veí: (mediana, mitjana) per cada k """
    from scipy.spatial import cKDTree
    N = pts.shape[0]
    if N > maxN:
        idx = np.random.choice(N, maxN, replace=False)
        P = pts[idx]
    else:
        P = pts
    tree = cKDTree(P)
    feats = []
    for k in k_list:
        d, _ = tree.query(P, k=k+1)  # inclou el mateix punt
        dk = d[:, -1]
        feats.extend([float(np.median(dk)), float(np.mean(dk))])
    return feats

def basic_features(pts: np.ndarray):
    """ Vector de *features* base per a cada escena """
    N = pts.shape[0]
    ext, vol = aabb_features(pts)
    dens = N / vol
    mean = pts.mean(0); var = pts.var(0)
    kstats = knn_stats(pts)
    feats = [N, ext[0], ext[1], ext[2], vol, dens, mean[0], mean[1], mean[2], var[0], var[1], var[2]]
    feats.extend(kstats)
    return np.array(feats, dtype=np.float32)

FEATURE_NAMES = [
    "N","aabb_x","aabb_y","aabb_z","aabb_vol","density",
    "mean_x","mean_y","mean_z","var_x","var_y","var_z",
    "k1_med","k1_mean","k2_med","k2_mean","k3_med","k3_mean",
    "k4_med","k4_mean","k8_med","k8_mean","k16_med","k16_mean"
]

# ----------------------- DBSCAN I ESCALAT D'EPS -----------------------

def dbscan_count_clusters(pts: np.ndarray, eps: float, min_pts: int) -> int:
    """ Executa DBSCAN (Open3D) i retorna nombre de clústers (soroll exclòs) """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=int(min_pts)))
    if labels.size == 0:
        return 0
    k = int(labels.max())
    return max(0, k+1)

def d4_median(pts: np.ndarray) -> float:
    """ Escala natural per eps: mediana de la distància al 4t veí (robusta a soroll) """
    from scipy.spatial import cKDTree
    N = pts.shape[0]
    maxN = 50000
    P = pts[np.random.choice(N, maxN, replace=False)] if N > maxN else pts
    tree = cKDTree(P)
    d, _ = tree.query(P, k=5)      # [d0=0, d1, d2, d3, d4]
    return float(np.median(d[:, -1]))

# ----------------------- CERCA D'EPS PER UNA ESCENA -----------------------

def search_eps_binary(pts: np.ndarray, k_gt: int, min_pts: int,
                      eps_lo: float, eps_hi: float, max_iter: int = 18):
    """
    Cerca binària d'eps assumint que, en general, augmentar eps ↓ nombre de clústers.
    No sempre és estrictament monòtona (soroll), però funciona bé en la pràctica.
    Retorna (eps_star, k_pred_star, err_star).
    """
    best = None
    lo, hi = eps_lo, eps_hi
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        k_pred = dbscan_count_clusters(pts, mid, min_pts)
        err = abs(k_pred - k_gt)
        if (best is None) or (err < best[2]) or (err == best[2] and mid < best[0]):
            best = (mid, k_pred, err)
        # política de moviment del bracket
        if k_pred > k_gt:
            # massa clústers → cal fusionar → pujar eps
            lo = mid
        elif k_pred < k_gt:
            # massa pocs → cal separar → baixar eps
            hi = mid
        else:
            # match exacte
            return mid, k_pred, 0
    return best

def search_eps_grid(pts: np.ndarray, k_gt: int, min_pts: int,
                    eps_lo: float, eps_hi: float, grid_steps: int = 21):
    """
    Cerca discreta d'eps en una graella lineal dins [eps_lo, eps_hi].
    Retorna (eps_star, k_pred_star, err_star).
    """
    best = None
    grid = np.linspace(eps_lo, eps_hi, grid_steps)
    for e in grid:
        k_pred = dbscan_count_clusters(pts, float(e), min_pts)
        err = abs(k_pred - k_gt)
        if (best is None) or (err < best[2]) or (err == best[2] and e < best[0]):
            best = (float(e), int(k_pred), int(err))
        if err == 0:
            break
    return best

def tune_scene(pts: np.ndarray, k_gt: int, minpts_list,
               method: str, eps_lo: float, eps_hi: float,
               max_iter_bin: int, grid_steps: int):
    """
    Prova diversos minPts i cerca eps amb el mètode escollit.
    Retorna el millor (eps*, minPts*, k_pred*, err*).
    """
    best = None
    for mp in minpts_list:
        if method == "binary":
            e, kpred, err = search_eps_binary(pts, k_gt, mp, eps_lo, eps_hi, max_iter=max_iter_bin)
        else:
            e, kpred, err = search_eps_grid(pts, k_gt, mp, eps_lo, eps_hi, grid_steps=grid_steps)
        cand = (float(e), int(mp), int(kpred), int(err))
        if (best is None) or (cand[3] < best[3]) or (cand[3] == best[3] and cand[0] < best[0]):
            best = cand
        if best[3] == 0:  # match exacte
            break
    return best  # (eps*, minPts*, k_pred*, err*)

# ----------------------- ENTRENAMENT DE MODELS (OPCIONAL) -----------------------

def build_regressor(kind: str, seed: int):
    """ Construeix un regressor segons el tipus. """
    if kind == "rf":
        return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1)
    if kind == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(128,64),
                                          activation="relu", max_iter=400,
                                          learning_rate_init=1e-3, random_state=seed))
    if kind == "rr":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    if kind == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=8, weights="distance"))
    raise ValueError("model_kind ha de ser rf/mlp/rr/knn")

def train_models(X, y_eps, y_mp, kind_eps: str, kind_mp: str, seed: int):
    mdl_eps = build_regressor(kind_eps, seed)
    mdl_mp  = build_regressor(kind_mp,  seed)
    mdl_eps.fit(X, y_eps)
    mdl_mp.fit(X, y_mp)
    return mdl_eps, mdl_mp

def eval_models(mdl_eps, mdl_mp, Xtr, y_eps_tr, y_mp_tr, Xva, y_eps_va, y_mp_va) -> str:
    def metr(mdl, X, y):
        yp = mdl.predict(X)
        return r2_score(y, yp), mean_absolute_error(y, yp)
    r2e_tr, mae_e_tr = metr(mdl_eps, Xtr, y_eps_tr)
    r2e_va, mae_e_va = metr(mdl_eps, Xva, y_eps_va)
    r2m_tr, mae_m_tr = metr(mdl_mp,  Xtr, y_mp_tr)
    r2m_va, mae_m_va = metr(mdl_mp,  Xva, y_mp_va)
    txt = []
    txt.append("=== VALIDACIÓ MODELS ===")
    txt.append(f"eps: R2={r2e_va:.3f}  MAE={mae_e_va:.6f}  (train R2={r2e_tr:.3f}, MAE={mae_e_tr:.6f})")
    txt.append(f"mp : R2={r2m_va:.3f}  MAE={mae_m_va:.3f}  (train R2={r2m_tr:.3f}, MAE={mae_m_tr:.3f})")
    return "\n".join(txt)

# ----------------------- PIPELINE PRINCIPAL -----------------------

def main():
    ap = argparse.ArgumentParser(
        "Entrenament per trobar eps i min_points per distància (ràpid amb cerca binària d'eps)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--scenes_root", required=True, help="Arrel de SCENES (conté dist_*.m amb points3d i gt_counts.csv)")
    ap.add_argument("--k_per_dist", type=int, default=60, help="#escenes aleatòries per distància (si en tens més, se'n mostregen)")
    ap.add_argument("--minpts_list", type=int, nargs="+", default=[8,10,12,16,20,24,28,32],
                    help="Conjunt discret de minPts a provar")
    ap.add_argument("--method", choices=["binary","grid"], default="binary", help="Mètode de cerca d'eps")
    ap.add_argument("--eps_lo_factor", type=float, default=0.5, help="Factor pel límit inferior: eps_lo = factor * d4_med")
    ap.add_argument("--eps_hi_factor", type=float, default=3.0, help="Factor pel límit superior: eps_hi = factor * d4_med")
    ap.add_argument("--max_iter_bin", type=int, default=18, help="Iteracions màximes per a la cerca binària")
    ap.add_argument("--grid_steps", type=int, default=21, help="# passos d'eps per a la cerca 'grid'")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatòria per mostrar escenes")
    ap.add_argument("--out", default="tuned", help="Carpeta de sortida (resultats i models)")

    # Opcional: entrenar models supervisats
    ap.add_argument("--train_models", action="store_true", help="Entrenar regressors per eps i minPts a partir de les etiquetes trobades")
    ap.add_argument("--model_kind_eps", default="rf", choices=["rf","mlp","rr","knn"], help="Model per a eps")
    ap.add_argument("--model_kind_mp",  default="rf", choices=["rf","mlp","rr","knn"], help="Model per a minPts")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    scenes_root = Path(args.scenes_root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    ddirs = discover_distance_dirs(scenes_root)
    if not ddirs:
        raise SystemExit("❌ No s'han trobat carpetes 'dist_*.m' dins --scenes_root.")

    rows = []      # per al CSV per-escena
    X_all = []     # features per entrenar models (si es demana)
    y_eps = []     # etiquetes eps*
    y_mp  = []     # etiquetes minPts*
    dlab  = []     # etiqueta de distància en metres (float) per cada escena

    per_dist_stats = {}  # mitjanes per distància

    for dist_dir in ddirs:
        dist_m = parse_distance_from_dirname(str(dist_dir))
        gt_map = load_gt_counts(dist_dir)
        files  = list_scene_files(dist_dir)
        if not gt_map or not files:
            print(f"[dist_{dist_m:.2f}m] ⚠️  Sense gt_counts.csv o sense points3d/*.npy")
            continue

        # Mostreig aleatori de k_per_dist escenes
        stems = [f.stem for f in files if f.stem in gt_map]
        random.shuffle(stems)
        stems = stems[:args.k_per_dist]

        eps_vals = []
        mp_vals  = []

        print(f"[dist_{dist_m:.2f}m] mostres: {len(stems)}")
        for stem in stems:
            f = dist_dir / "points3d" / f"{stem}.npy"
            pts = np.load(f).astype(np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 20:
                continue
            k_gt = int(gt_map[stem])

            # Escala natural per eps via d4_med
            d4 = d4_median(pts)
            eps_lo = d4 * args.eps_lo_factor
            eps_hi = d4 * args.eps_hi_factor

            # Cerca de (eps*, minPts*) amb el mètode escollit
            eps_star, mp_star, k_pred, err = tune_scene(
                pts, k_gt, args.minpts_list, args.method, eps_lo, eps_hi,
                max_iter_bin=args.max_iter_bin, grid_steps=args.grid_steps
            )

            # Guardem traçabilitat per escena
            rows.append({
                "dist_m": dist_m,
                "stem": stem,
                "N": int(pts.shape[0]),
                "d4_med": d4,
                "k_gt": k_gt,
                "eps_star": float(eps_star),
                "minPts_star": int(mp_star),
                "k_pred": int(k_pred),
                "err": int(err)
            })

            eps_vals.append(float(eps_star))
            mp_vals.append(int(mp_star))

            # features i etiquetes per entrenar (si es demana)
            if args.train_models:
                X_all.append(basic_features(pts))
                y_eps.append(float(eps_star))
                y_mp.append(int(mp_star))
                dlab.append(float(dist_m))

        if eps_vals:
            per_dist_stats[f"{dist_m:.2f}"] = {
                "eps": float(np.mean(eps_vals)),
                "min_points": int(round(np.mean(mp_vals)))
            }

    # ----- Escriure traçabilitat: per-escena i per-distància -----
    import csv
    csv_path = out_dir / "tuned_per_scene.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "dist_m","stem","N","d4_med","k_gt","eps_star","minPts_star","k_pred","err"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    (out_dir / "tuned_info.json").write_text(json.dumps(per_dist_stats, indent=2), encoding="utf-8")
    print(f"\n✔ Resultats per escena: {csv_path}")
    print(f"✔ Mitjanes per distància: {out_dir/'tuned_info.json'}")

    # ----- Entrenament (opcional) -----
    if args.train_models and X_all:
        X_all = np.stack(X_all, axis=0)
        y_eps = np.array(y_eps, dtype=np.float32)
        y_mp  = np.array(y_mp,  dtype=np.float32)

        Xtr, Xva, ye_tr, ye_va, ym_tr, ym_va = train_test_split(
            X_all, y_eps, y_mp, test_size=0.2, random_state=42
        )

        mdl_eps, mdl_mp = train_models(Xtr, ye_tr, ym_tr,
                                       kind_eps=args.model_kind_eps,
                                       kind_mp=args.model_kind_mp,
                                       seed=42)
        report = eval_models(mdl_eps, mdl_mp, Xtr, ye_tr, ym_tr, Xva, ye_va, ym_va)
        print("\n" + report)

        # Desa models i traçabilitat de *features*
        joblib.dump(mdl_eps, out_dir / f"model_eps_{args.model_kind_eps}.joblib")
        joblib.dump(mdl_mp,  out_dir / f"model_minpts_{args.model_kind_mp}.joblib")
        (out_dir / "feature_names.txt").write_text("\n".join(FEATURE_NAMES), encoding="utf-8")
        (out_dir / "metrics.txt").write_text(report + "\n", encoding="utf-8")
        print(f"✔ Models desats a: {out_dir}")

if __name__ == "__main__":
    main()
