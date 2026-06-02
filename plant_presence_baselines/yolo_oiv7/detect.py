"""
YOLOv8x-oiv7 plant-presence detection.

Single class only: "Plant" (OIV7 index 393). Bbox-union coverage.
Per-image processing time recorded in ProcessSeconds.

Output columns:
    PictureId, Folder, PlantPresence, NumPlantDetections,
    PlantCoverageFraction, ProcessSeconds
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

PLANT_CLASS_NAME: str = "Plant"


@dataclass
class ImageRecord:
    path: Path
    folder: str
    picture_id: str


@dataclass
class RunConfig:
    folderpath: Path
    output_dir: Path
    model: str
    conf: float
    iou: float
    imgsz: int
    device: str


def load_config() -> RunConfig:
    load_dotenv()
    fp = os.getenv("FOLDERPATH")
    if not fp:
        print("FOLDERPATH env var not set.")
        sys.exit(2)
    return RunConfig(
        folderpath=Path(fp),
        output_dir=Path(os.getenv("OUTPUT_DIR", "./runs")),
        model=os.getenv("MODEL", "yolov8x-oiv7.pt"),
        conf=float(os.getenv("CONF", "0.10")),
        iou=float(os.getenv("IOU", "0.50")),
        imgsz=int(os.getenv("IMGSZ", "1280")),
        device=os.getenv("DEVICE", "cuda:0"),
    )


def init_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("yolo_oiv7")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(sh)
    fh = logging.FileHandler(out_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def discover_images(root: Path) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = chain(
            sub.glob("*.[jJ][pP][gG]"),
            sub.glob("*.[jJ][pP][eE][gG]"),
            sub.glob("*.[pP][nN][gG]"),
        )
        for f in sorted(files):
            records.append(ImageRecord(path=f, folder=sub.name, picture_id=f.name))
    return records


def union_area(boxes_xyxy: np.ndarray, img_w: int, img_h: int) -> float:
    if boxes_xyxy.size == 0:
        return 0.0
    GRID = 256
    mask = np.zeros((GRID, GRID), dtype=bool)
    sx = GRID / float(img_w); sy = GRID / float(img_h)
    for x1, y1, x2, y2 in boxes_xyxy:
        gx1 = max(0, min(GRID, int(np.floor(x1 * sx))))
        gx2 = max(0, min(GRID, int(np.ceil (x2 * sx))))
        gy1 = max(0, min(GRID, int(np.floor(y1 * sy))))
        gy2 = max(0, min(GRID, int(np.ceil (y2 * sy))))
        if gx2 > gx1 and gy2 > gy1:
            mask[gy1:gy2, gx1:gx2] = True
    return float(mask.mean())


def main():
    cfg = load_config()
    stamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = cfg.output_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = init_logging(out_dir)
    logger.info(f"Run output: {out_dir}")
    logger.info(
        f"Config: model={cfg.model} class={PLANT_CLASS_NAME!r} conf={cfg.conf} "
        f"iou={cfg.iou} imgsz={cfg.imgsz} device={cfg.device} (batch=1 for per-image timing)"
    )

    from ultralytics import YOLO

    model = YOLO(cfg.model)
    names: Dict[int, str] = (
        model.names if isinstance(model.names, dict)
        else {i: n for i, n in enumerate(model.names)}
    )
    target_lower = PLANT_CLASS_NAME.lower()
    plant_idxs: Set[int] = {i for i, n in names.items() if str(n).lower() == target_lower}
    if not plant_idxs:
        logger.error(f"Class {PLANT_CLASS_NAME!r} not found in model.names")
        sys.exit(3)
    plant_idx_list = list(plant_idxs)
    logger.info(f"Plant class indices: {sorted(plant_idxs)}")

    records = discover_images(cfg.folderpath)
    logger.info(f"Discovered {len(records)} images")
    if not records:
        return

    csv_path = out_dir / "plant_presence.csv"
    columns = ["PictureId", "Folder", "PlantPresence",
               "NumPlantDetections", "PlantCoverageFraction", "ProcessSeconds"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)

    failures: List[dict] = []

    # Warmup so the first row's time isn't dominated by CUDA init.
    try:
        _ = model.predict(source=str(records[0].path), conf=cfg.conf, iou=cfg.iou,
                          imgsz=cfg.imgsz, device=cfg.device, verbose=False)
        logger.info("Warmup pass complete")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")

    t_run = time.time()
    pbar = tqdm(total=len(records), desc="Detecting", unit="img")

    for rec in records:
        t0 = time.perf_counter()
        try:
            res = model.predict(
                source=str(rec.path),
                conf=cfg.conf, iou=cfg.iou, imgsz=cfg.imgsz,
                device=cfg.device, verbose=False, stream=False,
            )[0]

            boxes = res.boxes
            img_h, img_w = res.orig_shape
            if boxes is None or boxes.cls is None or len(boxes) == 0:
                plant_boxes = np.zeros((0, 4), dtype=np.float32)
            else:
                cls = boxes.cls.detach().cpu().numpy().astype(np.int64)
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                mask = np.isin(cls, plant_idx_list)
                plant_boxes = xyxy[mask]

            n_det = int(plant_boxes.shape[0])
            coverage = union_area(plant_boxes, img_w, img_h)
            presence = round(coverage * 10.0, 2)

            elapsed = time.perf_counter() - t0
            row = [rec.picture_id, rec.folder, presence, n_det,
                   round(coverage, 6), round(elapsed, 4)]
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.exception(f"Failed on {rec.path}: {e}")
            failures.append({"PictureId": rec.picture_id, "Folder": rec.folder,
                             "Stage": "predict", "Error": str(e),
                             "ElapsedSeconds": round(elapsed, 4)})
        pbar.update(1)

    pbar.close()
    total = time.time() - t_run
    logger.info(f"Processed {len(records)} images in {total:.1f}s "
                f"({len(records)/max(1e-6, total):.2f} img/s)")

    if failures:
        fp = out_dir / "failures.csv"
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["PictureId","Folder","Stage","Error","ElapsedSeconds"])
            w.writeheader(); w.writerows(failures)
        logger.info(f"Failures: {len(failures)} -> {fp}")
    else:
        logger.info("No failures.")

    meta = {
        "model": cfg.model, "class": PLANT_CLASS_NAME,
        "conf": cfg.conf, "iou": cfg.iou, "imgsz": cfg.imgsz, "device": cfg.device,
        "batch": 1,  # per-image timing
        "folderpath": str(cfg.folderpath),
        "output_dir": str(out_dir),
        "num_images": len(records), "num_failures": len(failures),
        "elapsed_sec": round(total, 2), "timestamp": stamp,
    }
    (out_dir / "experiment.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
