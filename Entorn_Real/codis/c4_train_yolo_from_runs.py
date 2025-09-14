#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenament YOLO (detecció o segmentació) a partir del dataset real generat pels scripts UR3+Realsense.

Què fa:
  1) Recorre una o més carpetes 'run_YYYYmmdd_HHMMSS' dins --dataset_root
     i extreu parelles (RGB, mask) de:
        - rgb_png/*.png  +  mask_png/*.png  (prioritari)
        - rgb_png/*.png  +  mask_npy/*.npy  (alternativa)
     També pot llegir:
        - shots/shot_XXXX/rgb.png  +  shots/shot_XXXX/mask.png (si existeix)
     (Les carpetes 'scenes' tenen màscares “globalitzades” d’escenes; per entrenar single-piece,
      per defecte NO s’inclouen. Pots activar-les amb --use_scenes)

  2) Converteix cada màscara a anotació YOLO:
       * Mode 'seg' (per defecte): polígons (YOLOv8-seg)
       * Mode 'det': bounding boxes (YOLOv8-det)
     Classe única: 'piece'.

  3) Crea la jerarquia YOLO (images/train,val + labels/train,val) i data.yaml.

  4) Entrena amb Ultralytics:
       * 'seg' → yolov8n-seg.pt
       * 'det' → yolov8n.pt

Ús:
  python train_yolo_from_runs.py --dataset_root path\a\run_2025... [--out yolo_out] [--task seg|det]
                                 [--use_scenes] [--val_split 0.2] [--epochs 50] [--imgsz 640]
                                 [--device cpu|0] [--seed 42]

Notes:
  - Si el dataset conté múltiples 'run_*', l'script els ajunta.
  - Si en alguna run tens màscares en .npy però no en .png, també funcionen.
"""

import argparse, os, re, json, random, math, shutil
from pathlib import Path
import numpy as np
import cv2

# YOLO (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("❌ Cal 'ultralytics'. Instal·la-ho amb: pip install ultralytics") from e


# ----------------- Utils de lectura dataset -----------------

def list_runs(root: Path):
    """Torna llista de carpetes 'run_*' dins root; si root ja és un 'run_*', el retorna sol."""
    root = Path(root)
    if root.is_dir() and root.name.startswith("run_"):
        return [root]
    runs = sorted([p for p in root.glob("run_*") if p.is_dir()])
    if runs:
        return runs
    # També admetre “shots-only” (cap carpeta run_*, però té 'shots/shot_XXXX')
    if (root / "shots").exists():
        return [root]
    return []


def collect_samples_from_run(run_dir: Path, use_scenes: bool):
    """
    Extreu parelles (img_path, mask_path) d'una run.
    Prioritza rgb_png + mask_png/mask_npy. També afegeix shots si troba màscara.
    Retorna llista de dicts: {'img': Path, 'mask': Path or None, 'W': int, 'H': int}
    """
    out = []

    # 1) Captures “centrades” (rgb_png + mask_png/mask_npy)
    rgb_dir = run_dir / "rgb_png"
    msk_png_dir = run_dir / "mask_png"
    msk_npy_dir = run_dir / "mask_npy"
    if rgb_dir.exists():
        for img_p in sorted(rgb_dir.glob("*.png")):
            stem = img_p.stem
            m_png = msk_png_dir / f"{stem}.png"
            m_npy = msk_npy_dir / f"{stem}.npy"
            mask_p = None
            if m_png.exists():
                mask_p = m_png
            elif m_npy.exists():
                mask_p = m_npy
            # Si no hi ha màscara, el mostre deixem fora (per YOLO-seg)
            if mask_p is not None:
                out.append({"img": img_p, "mask": mask_p})
    
    # 2) SHOTS (si existeix una màscara al costat)
    shots_dir = run_dir / "shots"
    if shots_dir.exists():
        for shot in sorted(shots_dir.glob("shot_*")):
            img_p = shot / "rgb.png"
            # accepta mask.png (si algú l’ha creat) o mask.npy (si existeix)
            m_png = shot / "mask.png"
            m_npy = shot / "mask.npy"
            if img_p.exists():
                mask_p = None
                if m_png.exists():
                    mask_p = m_png
                elif m_npy.exists():
                    mask_p = m_npy
                if mask_p is not None:
                    out.append({"img": img_p, "mask": mask_p})

    # 3) SCENES (opcionales) → OJO: la màscara és la unió de moltes peces (no instance-level)
    if use_scenes:
        scenes_root = run_dir / "scenes"
        if scenes_root.exists():
            for dist_dir in sorted(scenes_root.glob("dist_*m")):
                rgb_dir2 = dist_dir / "depth_png"  # no tenim rgb real per scenes reals; entrenem amb depth? Millor no.
                # Si vols fer servir depth com a “imatge”, ho podem permetre:
                # En comptes de rgb, fem servir depth_png (grisos) + mask_png
                msk_dir2 = dist_dir / "mask_png"
                if rgb_dir2.exists() and msk_dir2.exists():
                    for m in sorted(msk_dir2.glob("*.png")):
                        stem = m.stem.replace("meta_", "")
                        dpng = rgb_dir2 / f"{stem}.png"
                        if dpng.exists():
                            out.append({"img": dpng, "mask": m})

    return out


def load_mask(mask_path: Path, target_size=None):
    """Carrega màscara en uint8 {0,1}. Accepta .png i .npy (booleà o 0/1)."""
    mask_path = Path(mask_path)
    if mask_path.suffix.lower() == ".png":
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        m = (m > 127).astype(np.uint8)
    elif mask_path.suffix.lower() == ".npy":
        arr = np.load(mask_path)
        if arr.dtype == bool:
            m = arr.astype(np.uint8)
        else:
            # pot venir com {0,1}, o 0/255
            m = (arr > 0).astype(np.uint8)
    else:
        return None

    if target_size is not None:
        W, H = target_size
        h0, w0 = m.shape[:2]
        if (w0 != W) or (h0 != H):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

    return m


def find_image_size(image_path: Path):
    im = cv2.imread(str(image_path))
    if im is None:
        return None
    h, w = im.shape[:2]
    return (w, h)


# ----------------- Conversió YOLO labels -----------------

def mask_to_polygons(mask_u8, max_points=200):
    """
    Converteix màscara binària a un o més polígons per YOLOv8-seg.
    Retorna llista de polígons (cada un és llista [x1,y1,x2,y2,...] en coordenades de píxel).
    """
    # Troba contorns externs
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Aproximació per simplificar
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)  # 1% del perímetre
        pts = approx.reshape(-1, 2)

        # Si encara hi ha molts punts, fem un submostreig uniforme
        if pts.shape[0] > max_points:
            idx = np.linspace(0, pts.shape[0]-1, num=max_points, dtype=int)
            pts = pts[idx]

        # A YOLOv8-seg li passem una sola línia: class x1 y1 x2 y2 ... (normalitzats 0..1)
        poly = pts.reshape(-1).astype(float).tolist()
        polys.append(poly)

    return polys


def polygons_to_yolo_line(polys, W, H, cls=0):
    """
    Converteix polígons (en píxels) a línies YOLOv8-seg (normalitzades).
    Retorna llista de línies (str).
    """
    lines = []
    for poly in polys:
        if len(poly) < 6:
            continue
        norm = []
        for i, v in enumerate(poly):
            if i % 2 == 0:  # x
                norm.append(v / W)
            else:           # y
                norm.append(v / H)
        # format: class x1 y1 x2 y2 ...
        line = str(int(cls)) + " " + " ".join(f"{c:.6f}" for c in norm)
        lines.append(line)
    return lines


def mask_to_bbox_lines(mask_u8, W, H, cls=0):
    """
    Alternativa detecció: genera una o més bboxes (YOLO-det).
    Retorna llista de línies YOLO 'cls cx cy w h' (normalitzades).
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            continue
        cx = (x + w/2) / W
        cy = (y + h/2) / H
        ww = w / W
        hh = h / H
        lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
    return lines


# ----------------- Construcció dataset YOLO + entrenament -----------------

def main():
    ap = argparse.ArgumentParser("Entrenar YOLO (seg/det) des del dataset real (runs/shots)")
    ap.add_argument("--dataset_root", required=True, help="Carpeta arrel (conté run_YYYYmmdd_..., o ella mateixa és una run)")
    ap.add_argument("--out", default="yolo_out_real", help="Carpeta de sortida (dataset YOLO + resultats entrenament)")
    ap.add_argument("--task", choices=["seg","det"], default=None,
                    help="Força 'seg' (segmentació) o 'det' (detecció). Per defecte: 'seg' si hi ha màscares, sinó 'det'.")
    ap.add_argument("--use_scenes", action="store_true",
                    help="Incloure 'scenes' (depth_png+mask_png). ATENCIÓ: màscara és la unió; no és instance-level.")
    ap.add_argument("--val_split", type=float, default=0.2, help="Proporció per a validació")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cpu | 0 | 1 ... (per defecte autodetecta)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out)
    ds_dir = out_root / "dataset_yolo"
    images_tr = ds_dir / "images" / "train"
    images_va = ds_dir / "images" / "val"
    labels_tr = ds_dir / "labels" / "train"
    labels_va = ds_dir / "labels" / "val"

    # Neteja i crea estructura
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    for p in [images_tr, images_va, labels_tr, labels_va]:
        p.mkdir(parents=True, exist_ok=True)

    # Recol·lecta runs
    runs = list_runs(dataset_root)
    if not runs:
        raise SystemExit("❌ No s'han trobat carpetes 'run_*' ni 'shots/' sota --dataset_root.")

    # Recol·lecta mostres (img + mask)
    samples = []
    for run in runs:
        samples += collect_samples_from_run(run, use_scenes=args.use_scenes)

    if not samples:
        raise SystemExit("❌ No s'han trobat (imatge + màscara). Assegura't que hi ha 'rgb_png' i 'mask_png' o 'mask_npy'.")

    # Filtra només mostres amb màscara existent i mida d’imatge vàlida
    cleaned = []
    for it in samples:
        img = it["img"]
        mask = it["mask"]
        size = find_image_size(img)
        if size is None:
            continue
        cleaned.append({"img": img, "mask": mask, "size": size})

    if not cleaned:
        raise SystemExit("❌ Cap imatge vàlida llegida (OpenCV no pot obrir-les?).")

    print(f"✅ Mostres vàlides: {len(cleaned)}")

    # Determina tasca per defecte
    task = args.task
    if task is None:
        task = "seg"  # tenim màscares, millor seg
    print(f"→ Tasca: {task}")

    # Train/Val split
    random.shuffle(cleaned)
    n_total = len(cleaned)
    n_val = max(1, int(round(args.val_split * n_total)))
    val_set = cleaned[:n_val]
    train_set = cleaned[n_val:]

    # Converteix a etiquetes YOLO i copia imatges
    def process_split(subset, images_dir, labels_dir):
        n_kept = 0
        for i, ex in enumerate(subset, 1):
            img_path = ex["img"]
            msk_path = ex["mask"]
            W, H = ex["size"]

            # llegir màscara (i redimensionar si cal)
            mask_u8 = load_mask(msk_path, target_size=(W, H))
            if mask_u8 is None or (mask_u8.max() == 0):
                # sense màscara útil → saltem
                continue

            # Decideix línies YOLO
            if task == "seg":
                polys = mask_to_polygons(mask_u8, max_points=200)
                lines = polygons_to_yolo_line(polys, W, H, cls=0)
            else:  # "det"
                lines = mask_to_bbox_lines(mask_u8, W, H, cls=0)

            if not lines:
                continue

            # Copia imatge i escriu label
            # Posem noms consecutius per simplicitat
            stem = f"sample_{n_kept:06d}"
            out_img = images_dir / f"{stem}.png"
            out_lbl = labels_dir / f"{stem}.txt"

            # copia imatge
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            cv2.imwrite(str(out_img), im)

            # escriu label
            with open(out_lbl, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

            n_kept += 1

        return n_kept

    kept_tr = process_split(train_set, images_tr, labels_tr)
    kept_va = process_split(val_set, images_va, labels_va)
    print(f"→ Train: {kept_tr}  |  Val: {kept_va}")
    if kept_tr == 0 or kept_va == 0:
        raise SystemExit("❌ No s’han generat etiquetes per train/val. Revisa les màscares i el paràmetre --task.")

    # Crea data.yaml
    data_yaml = {
        "path": str(ds_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "names": {0: "piece"}
    }
    if task == "seg":
        data_yaml["task"] = "segment"  # informatiu
    (ds_dir / "data.yaml").write_text(json.dumps(data_yaml, indent=2), encoding="utf-8")

    # Model base
    if task == "seg":
        base_weights = "yolov8n-seg.pt"
    else:
        base_weights = "yolov8n.pt"

    # Entrenament
    print("\n🚀 Entrenant YOLO...")
    model = YOLO(base_weights)
    train_args = dict(
        data=str(ds_dir / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(out_root / "runs"),
        name="train",
        seed=args.seed,
        verbose=True
    )
    if args.device is not None:
        train_args["device"] = args.device

    results = model.train(**train_args)

    print("\n✔ Entrenament finalitzat.")
    print(f"Pesos: {(out_root / 'runs' / 'train' / 'weights' / 'best.pt').resolve()}")
    print(f"Dataset YOLO: {ds_dir.resolve()}")
    print(" Pots provar inferència així (exemple):")
    if task == "seg":
        print(f"   yolo task=segment mode=predict model={(out_root / 'runs' / 'train' / 'weights' / 'best.pt').resolve()} source={str(images_va)}")
    else:
        print(f"   yolo task=detect mode=predict model={(out_root / 'runs' / 'train' / 'weights' / 'best.pt').resolve()} source={str(images_va)}")


if __name__ == "__main__":
    main()
