#!/usr/bin/env python3
"""Convert RPC COCO instances JSON to YOLO .txt labels (17 meta-categories)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "archive" / "retail_product_checkout"

# Must match retail_checkout.yaml
CLASS_NAMES = [
    "puffed_food",
    "dried_fruit",
    "dried_food",
    "instant_drink",
    "instant_noodles",
    "dessert",
    "drink",
    "alcohol",
    "milk",
    "canned_food",
    "chocolate",
    "gum",
    "candy",
    "seasoner",
    "personal_hygiene",
    "tissue",
    "stationery",
]
SUPER_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

SPLITS = {
    "train2019": "instances_train2019.json",
    "val2019": "instances_val2019.json",
    "test2019": "instances_test2019.json",
}


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = bbox
    x_c = (x + w / 2) / width
    y_c = (y + h / 2) / height
    return x_c, y_c, w / width, h / height


def convert_split(split_dir: str, json_name: str) -> dict[str, int]:
    json_path = DATASET / json_name
    image_dir = DATASET / split_dir

    with json_path.open(encoding="utf-8") as f:
        coco = json.load(f)

    cat_to_super: dict[int, int] = {}
    for cat in coco["categories"]:
        super_name = cat["supercategory"]
        if super_name not in SUPER_TO_ID:
            raise KeyError(f"Unknown supercategory {super_name!r} in {json_path.name}")
        cat_to_super[cat["id"]] = SUPER_TO_ID[super_name]

    images = {img["id"]: img for img in coco["images"]}
    by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    stats = {"labels_written": 0, "images_with_ann": 0, "missing_image_file": 0, "skipped_bbox": 0}

    for image_id, anns in by_image.items():
        img = images[image_id]
        w, h = img["width"], img["height"]
        stem = Path(img["file_name"]).stem
        label_path = image_dir / f"{stem}.txt"

        lines: list[str] = []
        for ann in anns:
            cls_id = cat_to_super[ann["category_id"]]
            xc, yc, bw, bh = coco_bbox_to_yolo(ann["bbox"], w, h)
            if bw <= 0 or bh <= 0:
                stats["skipped_bbox"] += 1
                continue
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            bw = min(max(bw, 0.0), 1.0)
            bh = min(max(bh, 0.0), 1.0)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            continue

        img_path = image_dir / img["file_name"]
        if not img_path.exists():
            stats["missing_image_file"] += 1

        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats["labels_written"] += 1
        stats["images_with_ann"] += 1

    return stats


def main() -> None:
    for split_dir, json_name in SPLITS.items():
        stats = convert_split(split_dir, json_name)
        print(f"{split_dir}: {stats}")


if __name__ == "__main__":
    main()
