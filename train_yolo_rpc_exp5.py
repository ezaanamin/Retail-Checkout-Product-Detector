#!/usr/bin/env python3
"""
YOLO11 on Retail Product Checkout (RPC) - Experiment 5 (two-stage training).

Stage 1: frozen backbone, strong augmentation  -> keeps pretrained features,
         learns the detection head on the train domain.
Stage 2: unfrozen, lower LR, lighter aug       -> fine-tunes everything,
         starting from stage 1 best.pt.

Usage examples:
    python train_yolo_rpc_exp5.py --diagnose                 # check train vs val gap first
    python train_yolo_rpc_exp5.py --stage 1                  # stage 1 only
    python train_yolo_rpc_exp5.py --stage 2                  # stage 2 only (needs stage 1)
    python train_yolo_rpc_exp5.py --stage all                # both, back to back
    python train_yolo_rpc_exp5.py --stage 1 --resume         # resume interrupted stage 1
    python train_yolo_rpc_exp5.py --stage all --batch 4 --model yolo11s.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parent

SETTINGS.update({
    "datasets_dir": str(ROOT),
    "runs_dir": str(ROOT / "runs"),
})

_SYNTH_YAML = ROOT / "retail_checkout_synth.yaml"
# Use the synthetic-mix dataset if make_synthetic_rpc.py has been run.
DATA_YAML = _SYNTH_YAML if _SYNTH_YAML.exists() else ROOT / "retail_checkout.yaml"
PROJECT = ROOT / "runs" / "rpc"

EXP = "sku100_exp5"
STAGE1_NAME = f"{EXP}_stage1_frozen"
STAGE2_NAME = f"{EXP}_stage2_finetune"


# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
def common_kwargs(args: argparse.Namespace, device: str) -> dict:
    return dict(
        data=str(DATA_YAML),
        imgsz=args.imgsz,
        batch=args.batch,
        amp=args.amp,                 # GTX 1650 + AMP can give NaNs; off by default
        workers=args.workers,
        cache=args.cache,             # "disk" / "ram" / False
        device=device,
        optimizer="AdamW",
        cos_lr=True,
        seed=42,
        deterministic=False,          # faster than deterministic=True
        project=str(PROJECT),
        exist_ok=False,
        save=True,
        save_period=5,
        pretrained=True,
        plots=True,
    )


def stage1_kwargs(args: argparse.Namespace, device: str) -> dict:
    """Frozen backbone + strong augmentation to bridge the single-item -> checkout gap."""
    return dict(
        **common_kwargs(args, device),
        name=STAGE1_NAME,
        epochs=args.epochs1,
        patience=10,
        lr0=3e-4,
        lrf=0.05,
        weight_decay=1e-3,
        warmup_epochs=2,
        freeze=10,                    # freeze first 10 layers (backbone)
        # Augmentation (strong)
        mosaic=1.0,
        close_mosaic=5,
        mixup=0.1,
        scale=0.6,
        degrees=10.0,
        translate=0.2,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.1,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
    )


def stage2_kwargs(args: argparse.Namespace, device: str) -> dict:
    """Unfrozen fine-tune, lower LR, slightly gentler augmentation."""
    return dict(
        **common_kwargs(args, device),
        name=STAGE2_NAME,
        epochs=args.epochs2,
        patience=10,
        lr0=1e-4,
        lrf=0.05,
        weight_decay=1e-3,
        warmup_epochs=1,
        freeze=None,
        mosaic=1.0,
        close_mosaic=5,
        mixup=0.05,
        scale=0.5,
        degrees=8.0,
        translate=0.15,
        perspective=0.0003,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.35,
        erasing=0.3,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def resolve_device() -> str:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1024**3:.2f} GiB)")
        return "0"
    print("WARNING: CUDA unavailable; training will run on CPU.", file=sys.stderr)
    return "cpu"


def run_dir(name: str) -> Path:
    return PROJECT / name


def last_ckpt(name: str) -> Path:
    return run_dir(name) / "weights" / "last.pt"


def best_ckpt(name: str) -> Path:
    return run_dir(name) / "weights" / "best.pt"


def print_summary(title: str, kwargs: dict, weights: Path | str) -> None:
    print(f"\n=== {title} ===")
    print(f"Weights : {weights}")
    for key in ("data", "epochs", "imgsz", "batch", "amp", "workers", "cache",
                "lr0", "lrf", "freeze", "mosaic", "mixup", "scale", "name"):
        print(f"{key:<9}: {kwargs.get(key)}")
    print()


def train_stage(name: str, weights: Path | str, kwargs: dict, resume: bool) -> None:
    if resume:
        ckpt = last_ckpt(name)
        if not ckpt.exists():
            sys.exit(f"ERROR: no checkpoint at {ckpt}")
        print(f"Resuming {name} from {ckpt}")
        YOLO(str(ckpt)).train(resume=True)
        return

    if run_dir(name).exists():
        sys.exit(
            f"ERROR: {run_dir(name)} already exists.\n"
            "Use --resume, delete the folder, or change EXP at the top of the file."
        )

    print_summary(name, kwargs, weights)
    YOLO(str(weights)).train(**kwargs)


def diagnose(weights: Path | str, device: str, args: argparse.Namespace) -> None:
    """Compare mAP on train vs val. High train + near-zero val => domain gap."""
    model = YOLO(str(weights))
    results = {}
    for split in ("train", "val"):
        print(f"\n--- Validating on '{split}' split ---")
        m = model.val(
            data=str(DATA_YAML),
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            plots=False,
            fraction=args.fraction,
            project=str(PROJECT),
            name=f"{EXP}_diag_{split}",
            exist_ok=True,
        )
        results[split] = (m.box.map50, m.box.map)

    print("\n=== Diagnostic summary ===")
    for split, (m50, m5095) in results.items():
        print(f"{split:<6} mAP50={m50:.4f}  mAP50-95={m5095:.4f}")
    gap = results["train"][0] - results["val"][0]
    if gap > 0.3:
        print("-> Large train/val gap: domain gap. Add synthetic multi-item scenes "
              "or move real checkout images into train.")
    else:
        print("-> Gap is small: look at labels, class mapping or model capacity instead.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage YOLO11 training on RPC.")
    p.add_argument("--stage", choices=["1", "2", "all"], default="all")
    p.add_argument("--resume", action="store_true", help="Resume the selected stage.")
    p.add_argument("--diagnose", action="store_true",
                   help="Only run the train-vs-val diagnostic and exit.")
    p.add_argument("--model", default="yolo11n.pt", help="Initial weights (e.g. yolo11s.pt).")
    p.add_argument("--diag-weights", default=None,
                   help="Weights for --diagnose (default: stage 2 best, stage 1 best, or --model).")
    p.add_argument("--fraction", type=float, default=0.1,
                   help="Fraction of each split used by --diagnose (default 0.1).")
    p.add_argument("--epochs1", type=int, default=25)
    p.add_argument("--epochs2", type=int, default=35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4, help="Use 2-4 on a GTX 1650 (4 GB).")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cache", default="disk", choices=["disk", "ram", "false"])
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    args = p.parse_args()
    args.cache = False if args.cache == "false" else args.cache
    return args


def main() -> None:
    args = parse_args()

    if not DATA_YAML.exists():
        sys.exit(f"ERROR: dataset config not found: {DATA_YAML}")

    device = resolve_device()
    init_weights = ROOT / args.model if (ROOT / args.model).exists() else args.model

    try:
        if args.diagnose:
            if args.diag_weights:
                w = args.diag_weights
            elif best_ckpt(STAGE2_NAME).exists():
                w = best_ckpt(STAGE2_NAME)
            elif best_ckpt(STAGE1_NAME).exists():
                w = best_ckpt(STAGE1_NAME)
            else:
                sys.exit(
                    "ERROR: no trained weights found. Diagnosing the pretrained COCO model "
                    "is meaningless.\nPass --diag-weights runs/rpc/sku100_experiment_4/weights/best.pt"
                )
            diagnose(w, device, args)
            return

        if args.stage in ("1", "all"):
            train_stage(STAGE1_NAME, init_weights, stage1_kwargs(args, device),
                        resume=args.resume and args.stage == "1")

        if args.stage in ("2", "all"):
            s1_best = best_ckpt(STAGE1_NAME)
            if not (args.resume and args.stage == "2") and not s1_best.exists():
                sys.exit(f"ERROR: stage 1 weights missing: {s1_best}\nRun --stage 1 first.")
            train_stage(STAGE2_NAME, s1_best, stage2_kwargs(args, device),
                        resume=args.resume and args.stage == "2")

        print("\nDone. Watch val/cls_loss: it should fall together with train/cls_loss.")
        print(f"Final weights: {best_ckpt(STAGE2_NAME)}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Re-run with --resume and the same --stage.")
        sys.exit(130)


if __name__ == "__main__":
    main()
