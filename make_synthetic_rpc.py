#!/usr/bin/env python3
"""
Build synthetic multi-item checkout scenes from the single-item RPC train images,
then write a new dataset yaml that mixes:
    - synthetic multi-item scenes          (--n-synth)
    - real single-item train images         (--n-single)
    - a slice of REAL checkout (val) images (--n-real), removed from validation

Everything is referenced through txt image lists, so no files are moved or copied
(except the generated scenes and cutouts).

Usage:
    python make_synthetic_rpc.py                       # defaults
    python make_synthetic_rpc.py --n-synth 6000 --n-real 2000
    python make_synthetic_rpc.py --rebuild-bank        # redo the cutouts
Then train with:  python train_yolo_rpc_exp5.py --stage all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
IMG_EXT = {".jpg", ".jpeg", ".png"}
S = 640  # scene size (matches imgsz)


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def read_first_box(img: Path):
    txt = img.with_suffix(".txt")
    if not txt.exists():
        return None
    for line in txt.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            c, xc, yc, w, h = int(parts[0]), *map(float, parts[1:5])
            return c, xc, yc, w, h
    return None


# --------------------------------------------------------------------------- #
# Cutout bank (GrabCut on the single-item images)
# --------------------------------------------------------------------------- #
def extract_cutout(img_path: Path, box) -> np.ndarray | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    H, W = img.shape[:2]
    _, xc, yc, bw, bh = box
    x1, y1 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
    x2, y2 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None

    pad = int(0.08 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
    cx2, cy2 = min(x2 + pad, W), min(y2 + pad, H)
    crop = img[cy1:cy2, cx1:cx2]
    rect = [x1 - cx1, y1 - cy1, x2 - x1, y2 - y1]

    scale = 256 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rect = [max(int(v * scale), 1) for v in rect]
    ch, cw = crop.shape[:2]
    rect[2] = min(rect[2], cw - rect[0] - 1)
    rect[3] = min(rect[3], ch - rect[1] - 1)
    if rect[2] < 8 or rect[3] < 8:
        return None

    mask = np.zeros((ch, cw), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, tuple(rect), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None

    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg = (lab == k).astype(np.uint8)

    ratio = fg.sum() / float(rect[2] * rect[3])
    if ratio < 0.3 or ratio > 0.98:   # failed segmentation (nothing, or whole box)
        return None

    alpha = cv2.GaussianBlur(fg * 255, (3, 3), 0)
    ys, xs = np.nonzero(fg)
    rgba = np.dstack([crop, alpha])
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_bank(train_imgs: list[Path], out_dir: Path, per_class: int,
               rng: np.random.Generator, rebuild: bool) -> dict[int, list[np.ndarray]]:
    bank_dir = out_dir / "cutouts"
    existing = {}
    if bank_dir.exists() and not rebuild:
        for cdir in sorted(bank_dir.iterdir()):
            files = sorted(cdir.glob("*.png"))
            if files:
                existing[int(cdir.name)] = [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in files]
        if existing and min(len(v) for v in existing.values()) >= min(per_class, 30):
            print(f"Loaded cutout bank: {len(existing)} classes, "
                  f"{sum(len(v) for v in existing.values())} cutouts")
            return existing

    print("Reading labels of train images (one pass)...")
    by_class: dict[int, list[tuple[Path, tuple]]] = {}
    for i, p in enumerate(train_imgs):
        box = read_first_box(p)
        if box:
            by_class.setdefault(box[0], []).append((p, box))
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(train_imgs)}")
    if not by_class:
        sys.exit("ERROR: no labels found next to train images.")
    print(f"Found {len(by_class)} classes. Building cutouts ({per_class} per class)...")

    bank: dict[int, list[np.ndarray]] = {}
    for cls in sorted(by_class):
        items = by_class[cls]
        order = rng.permutation(len(items))
        cdir = bank_dir / str(cls)
        cdir.mkdir(parents=True, exist_ok=True)
        for old in cdir.glob("*.png"):
            old.unlink()
        got = []
        for idx in order:
            if len(got) >= per_class:
                break
            path, box = items[int(idx)]
            rgba = extract_cutout(path, box)
            if rgba is None:
                continue
            cv2.imwrite(str(cdir / f"{len(got):04d}.png"), rgba)
            got.append(rgba)
        bank[cls] = got
        print(f"  class {cls}: {len(got)} cutouts")
    bank = {c: v for c, v in bank.items() if v}
    if not bank:
        sys.exit("ERROR: could not extract any cutouts.")
    return bank


# --------------------------------------------------------------------------- #
# Scene composition
# --------------------------------------------------------------------------- #
def make_background(rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(165, 245) + rng.uniform(-8, 8, 3)
    bg = np.empty((S, S, 3), np.float32)
    bg[:] = base
    low = rng.normal(0, 1, (8, 8, 3)).astype(np.float32) * rng.uniform(2, 8)
    bg += cv2.resize(low, (S, S), interpolation=cv2.INTER_CUBIC)
    ramp = np.linspace(-1, 1, S, dtype=np.float32)
    bg += ramp[None, :, None] * rng.uniform(-12, 12)
    bg += ramp[:, None, None] * rng.uniform(-12, 12)
    bg += rng.normal(0, rng.uniform(1, 4), bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255)


def prepare_item(rgba: np.ndarray, rng: np.random.Generator):
    h, w = rgba.shape[:2]
    s = rng.uniform(55, 150) / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    img = cv2.resize(rgba, (max(1, int(w * s)), max(1, int(h * s))), interpolation=interp)
    if rng.random() < 0.5:
        img = img[:, ::-1]
    angle = rng.uniform(-180, 180) if rng.random() < 0.6 else rng.uniform(-25, 25)

    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    img = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    rgb = img[..., :3].astype(np.float32)
    rgb = rgb * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)
    return np.clip(rgb, 0, 255), img[..., 3].astype(np.float32) / 255.0


def make_scene(bank: dict[int, list[np.ndarray]], rng: np.random.Generator):
    classes = list(bank)
    canvas = make_background(rng)
    idmap = np.full((S, S), -1, np.int16)
    items: list[tuple[int, float]] = []

    for _ in range(int(rng.integers(6, 19))):
        cls = classes[int(rng.integers(len(classes)))]
        rgba = bank[cls][int(rng.integers(len(bank[cls])))]
        rgb, a = prepare_item(rgba, rng)
        h, w = a.shape
        if h >= S or w >= S:
            continue
        x, y = int(rng.integers(0, S - w + 1)), int(rng.integers(0, S - h + 1))
        am = a[..., None]
        canvas[y:y + h, x:x + w] = canvas[y:y + h, x:x + w] * (1 - am) + rgb * am
        idmap[y:y + h, x:x + w][a > 0.5] = len(items)
        items.append((cls, float((a > 0.5).sum())))

    lines = []
    for k, (cls, area) in enumerate(items):
        ys, xs = np.nonzero(idmap == k)
        if len(xs) == 0 or len(xs) / max(area, 1.0) < 0.4:   # mostly hidden -> skip
            continue
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if x2 - x1 < 12 or y2 - y1 < 12:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / S:.6f} {(y1 + y2) / 2 / S:.6f} "
                     f"{(x2 - x1) / S:.6f} {(y2 - y1) / S:.6f}")

    canvas += rng.normal(0, rng.uniform(0, 3), canvas.shape).astype(np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8), lines


def generate_scenes(bank, out_dir: Path, n: int, rng: np.random.Generator) -> list[Path]:
    img_dir, lbl_dir, prev_dir = out_dir / "images", out_dir / "labels", out_dir / "preview"
    for d in (img_dir, lbl_dir, prev_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths = []
    for i in range(n):
        for _ in range(5):                       # retry if everything got occluded
            img, lines = make_scene(bank, rng)
            if len(lines) >= 3:
                break
        p = img_dir / f"syn_{i:05d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(70, 96))])
        (lbl_dir / f"syn_{i:05d}.txt").write_text("\n".join(lines) + "\n")
        paths.append(p)

        if i < 8:                                # preview with boxes drawn
            vis = img.copy()
            for ln in lines:
                c, xc, yc, w, h = ln.split()
                xc, yc, w, h = (float(v) * S for v in (xc, yc, w, h))
                cv2.rectangle(vis, (int(xc - w / 2), int(yc - h / 2)),
                              (int(xc + w / 2), int(yc + h / 2)), (0, 200, 0), 1)
                cv2.putText(vis, c, (int(xc - w / 2), int(yc - h / 2) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.imwrite(str(prev_dir / f"preview_{i}.jpg"), vis)
        if (i + 1) % 500 == 0:
            print(f"  scenes: {i + 1}/{n}")
    return paths


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic checkout scenes for RPC.")
    ap.add_argument("--dataset", type=Path, default=ROOT / "archive" / "retail_product_checkout")
    ap.add_argument("--train-dir", default="train2019")
    ap.add_argument("--val-dir", default="val2019")
    ap.add_argument("--out", type=Path, default=None, help="default: <dataset>/synthetic")
    ap.add_argument("--base-yaml", type=Path, default=ROOT / "retail_checkout.yaml")
    ap.add_argument("--yaml-out", type=Path, default=ROOT / "retail_checkout_synth.yaml")
    ap.add_argument("--n-synth", type=int, default=4000)
    ap.add_argument("--n-single", type=int, default=4000, help="real single-item images to keep")
    ap.add_argument("--n-real", type=int, default=2000, help="real checkout images moved to train")
    ap.add_argument("--cutouts-per-class", type=int, default=120)
    ap.add_argument("--rebuild-bank", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ds = args.dataset
    out = args.out or ds / "synthetic"
    train_imgs = list_images(ds / args.train_dir)
    val_imgs = list_images(ds / args.val_dir)
    if not train_imgs or not val_imgs:
        sys.exit(f"ERROR: no images in {ds / args.train_dir} or {ds / args.val_dir}")
    if not args.base_yaml.exists():
        sys.exit(f"ERROR: {args.base_yaml} not found")
    print(f"Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")

    bank = build_bank(train_imgs, out, args.cutouts_per_class, rng, args.rebuild_bank)
    print(f"Generating {args.n_synth} synthetic scenes...")
    synth = generate_scenes(bank, out, args.n_synth, rng)

    singles = [train_imgs[i] for i in rng.permutation(len(train_imgs))[:args.n_single]]
    order = rng.permutation(len(val_imgs))
    n_real = min(args.n_real, len(val_imgs) - 1000)       # always keep >=1000 for validation
    real_train = [val_imgs[i] for i in order[:n_real]]
    real_val = [val_imgs[i] for i in order[n_real:]]

    train_list, val_list = out / "train_list.txt", out / "val_list.txt"
    train_list.write_text("\n".join(str(p) for p in (*synth, *singles, *real_train)) + "\n")
    val_list.write_text("\n".join(str(p) for p in real_val) + "\n")

    base = yaml.safe_load(args.base_yaml.read_text())
    args.yaml_out.write_text(yaml.safe_dump({
        "path": str(ds), "train": str(train_list), "val": str(val_list), "names": base["names"],
    }, sort_keys=False))

    print("\n=== Done ===")
    print(f"Train: {len(synth)} synthetic + {len(singles)} single + {len(real_train)} real checkout")
    print(f"Val  : {len(real_val)} real checkout images (held out)")
    print(f"Yaml : {args.yaml_out}")
    print(f"Check the boxes in: {out / 'preview'}")


if __name__ == "__main__":
    main()
