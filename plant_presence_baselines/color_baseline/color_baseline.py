#!/usr/bin/env python3
"""
Simple, cheap color-based "greenness" baseline.

Purpose
-------
This script implements the inexpensive baseline requested by Reviewer 2
(Comment 6): instead of a multi-billion-parameter vision-language model,
estimate the "greenness" of each photograph using only classic, cheap
color statistics (color histograms / green-pixel counting). The resulting
per-image and per-participant scores can then be correlated against the
VLM greenness ratings to test whether the VLM provides a real advantage
over a trivial method.

It deliberately uses NO learned model and NO object detector -- only
arithmetic on RGB / HSV pixel values, so it is fast and fully reproducible.

Metrics computed per image
--------------------------
mean_r, mean_g, mean_b
    Mean of each RGB channel (0-255).

green_pixel_frac_hsv
    Fraction of pixels classified as "green" in HSV space
    (hue in a green band, with a minimum saturation and value).
    This is the most direct analogue of "how much green is in the picture".

green_dominant_frac
    Fraction of pixels where the green channel is the strongest
    (G > R and G > B). A threshold-free, illumination-robust variant.

mean_norm_green
    Mean of G / (R + G + B): chromatic greenness independent of brightness.

exg_mean
    Mean Excess Green Index, ExG = 2*g - r - b on normalized channels
    (r,g,b each divided by R+G+B). A standard vegetation index.

hist_g_*  (optional, --histograms)
    Coarse 8-bin normalized histogram of the green channel.

Outputs
-------
color_baseline_per_image.csv        one row per photograph
color_baseline_per_participant.csv  one row per participant (means + n_images)
greenness.csv                       DROP-IN replacement for the VLM's
                                    greenness.csv (identical schema and join
                                    keys: PictureId, Folder, TimeCreated,
                                    TimeDigitized, TimeModified, Greenness,
                                    GreennessConfidence, ImageDescription).
                                    Feed this to the SAME R statistical
                                    pipeline used for the VLM so Table 1,
                                    Table 2 and the stress correlation are
                                    computed identically and are directly
                                    comparable. The `Greenness` column holds
                                    the chosen cheap metric (default
                                    green_pixel_frac_hsv), optionally
                                    rescaled to the VLM's 1-10 range.

Usage
-----
With NO arguments it uses paths relative to this script's location:
    --input  = ../filtered_prepped   (the JPEGs from prep_images.py)
    --outdir = .                     (write CSVs next to this script)

So on the server you can just run:
    python color_baseline.py

Override either with --input / --outdir when needed. See README.md.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

# PIL safety: these photos are user phone photos; allow large images and
# do not crash on slightly truncated files.
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# HSV green band. OpenCV-free: we use PIL's HSV where H,S,V are all 0-255.
# Green hue in 0-360 deg is ~70-160 deg; scaled to 0-255 that is ~50-114.
HSV_H_LO = 50
HSV_H_HI = 114
HSV_S_MIN = 40   # ignore near-grey pixels
HSV_V_MIN = 30   # ignore near-black pixels

HIST_BINS = 8


def list_images(input_dir: str) -> list[tuple[str, str]]:
    """Return list of (participant_id, absolute_image_path)."""
    items: list[tuple[str, str]] = []
    for participant in sorted(os.listdir(input_dir)):
        pdir = os.path.join(input_dir, participant)
        if not os.path.isdir(pdir):
            continue
        for fname in sorted(os.listdir(pdir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in VALID_EXTS:
                items.append((participant, os.path.join(pdir, fname)))
    return items


def read_image_time(im) -> dict:
    """EXIF timestamps, replicating Step1/Imager.py:readImageTime EXACTLY so
    the join to the esmira self-reports is byte-for-byte identical to the VLM
    pipeline (same tags, same getexif() scope, same '-' default)."""
    data = {"TO": "-", "TD": "-", "TM": "-"}
    try:
        exifdata = im.getexif()
        for tag_id in exifdata:
            if tag_id == 36867:        # DateTimeOriginal
                data["TO"] = exifdata[tag_id]
            elif tag_id == 36868:      # DateTimeDigitized
                data["TD"] = exifdata[tag_id]
            elif tag_id == 306:        # DateTime
                data["TM"] = exifdata[tag_id]
    except Exception:
        pass
    return data


def analyze_image(path: str, max_size: int, want_hist: bool) -> dict:
    """Compute cheap color-greenness metrics for one image."""
    with Image.open(path) as im:
        times = read_image_time(im)        # read BEFORE convert/resize
        im = im.convert("RGB")
        w, h = im.size
        if max_size and max(w, h) > max_size:
            scale = max_size / float(max(w, h))
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        rgb = np.asarray(im, dtype=np.float64)          # H x W x 3
        hsv = np.asarray(im.convert("HSV"), dtype=np.uint8)

    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Hh, Ss, Vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    total = R.size
    s = R + G + B
    s_safe = np.where(s == 0, 1.0, s)  # avoid divide-by-zero

    green_mask = (
        (Hh >= HSV_H_LO) & (Hh <= HSV_H_HI)
        & (Ss >= HSV_S_MIN) & (Vv >= HSV_V_MIN)
    )
    green_dom = (G > R) & (G > B)

    r_n, g_n, b_n = R / s_safe, G / s_safe, B / s_safe
    exg = 2.0 * g_n - r_n - b_n

    out = {
        "width": w,
        "height": h,
        "mean_r": float(R.mean()),
        "mean_g": float(G.mean()),
        "mean_b": float(B.mean()),
        "green_pixel_frac_hsv": float(green_mask.mean()),
        "green_dominant_frac": float(green_dom.mean()),
        "mean_norm_green": float(g_n.mean()),
        "exg_mean": float(exg.mean()),
        "n_pixels": int(total),
        "TimeCreated": times["TO"],
        "TimeDigitized": times["TD"],
        "TimeModified": times["TM"],
    }

    if want_hist:
        hist, _ = np.histogram(G, bins=HIST_BINS, range=(0, 255))
        hist = hist / max(1, hist.sum())
        for i, v in enumerate(hist):
            out[f"hist_g_{i}"] = float(v)

    return out


def _worker(args):
    participant, path, max_size, want_hist = args
    try:
        m = analyze_image(path, max_size, want_hist)
        m["participant"] = participant
        m["image"] = os.path.basename(path)
        m["error"] = ""
        return m
    except Exception as e:  # never let one bad file kill the run
        return {
            "participant": participant,
            "image": os.path.basename(path),
            "error": f"{type(e).__name__}: {e}",
        }


def main() -> int:
    # Defaults relative to THIS script's location so it works no-args on the
    # server: prep_images.py writes to ../filtered_prepped, and CSVs land
    # next to the script (i.e. in ambuv/color_baseline/).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input  = os.path.normpath(os.path.join(script_dir, "..",
                                                   "filtered_prepped"))
    default_outdir = script_dir

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=default_input,
                    help=f"Directory with one subfolder per participant "
                         f"(default: {default_input})")
    ap.add_argument("--outdir", default=default_outdir,
                    help=f"Where to write the CSV outputs "
                         f"(default: {default_outdir})")
    ap.add_argument("--max-size", type=int, default=0,
                    help="Downscale longest image side to this many px "
                         "before analysis (0 = no resize, the default; "
                         "assumes --input is already prepped by "
                         "prep_images.py to 1280 px). Set to 1280 if "
                         "you're running on the raw filtered/ tree. "
                         "Color fractions are scale-invariant, so this "
                         "only affects speed, not the metric values.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                    help="Parallel worker processes")
    ap.add_argument("--histograms", action="store_true",
                    help="Also emit an 8-bin green-channel histogram per image")
    ap.add_argument("--vlm-greenness-metric", default="green_pixel_frac_hsv",
                    choices=["green_pixel_frac_hsv", "green_dominant_frac",
                             "mean_norm_green", "exg_mean"],
                    help="Which cheap metric becomes the `Greenness` column in "
                         "the VLM-compatible greenness.csv (default: "
                         "green_pixel_frac_hsv = literal %% green pixels, the "
                         "metric Reviewer 2 named).")
    ap.add_argument("--vlm-rescale", default="x10",
                    choices=["x10", "raw"],
                    help="'x10' (default): Greenness = 10 * raw_fraction. "
                         "Purely absolute, no offset, no dataset peeking. "
                         "0%% green pixels -> 0, 100%% green pixels -> 10. "
                         "'raw' keeps the native [0, 1] fraction untouched.")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: input dir not found: {args.input}", file=sys.stderr)
        return 1
    os.makedirs(args.outdir, exist_ok=True)

    images = list_images(args.input)
    if not images:
        print(f"ERROR: no images found under {args.input}", file=sys.stderr)
        return 1
    print(f"Found {len(images)} images across "
          f"{len({p for p, _ in images})} participants.", flush=True)

    tasks = [(p, path, args.max_size, args.histograms) for p, path in images]
    rows: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 200 == 0 or done == len(futs):
                print(f"  processed {done}/{len(futs)}", flush=True)

    # Stable ordering for reproducible CSVs.
    rows.sort(key=lambda r: (r["participant"], r["image"]))

    metric_cols = [
        "width", "height", "n_pixels",
        "mean_r", "mean_g", "mean_b",
        "green_pixel_frac_hsv", "green_dominant_frac",
        "mean_norm_green", "exg_mean",
    ]
    hist_cols = [f"hist_g_{i}" for i in range(HIST_BINS)] if args.histograms else []
    per_image_cols = ["participant", "image"] + metric_cols + hist_cols + ["error"]

    per_image_path = os.path.join(args.outdir, "color_baseline_per_image.csv")
    with open(per_image_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=per_image_cols, extrasaction="ignore")
        wri.writeheader()
        for r in rows:
            wri.writerow(r)

    # Per-participant aggregation (mean of successfully analyzed images).
    ok = [r for r in rows if not r.get("error")]
    failed = [r for r in rows if r.get("error")]
    agg_cols = ["green_pixel_frac_hsv", "green_dominant_frac",
                "mean_norm_green", "exg_mean",
                "mean_r", "mean_g", "mean_b"]
    by_p: dict[str, list[dict]] = {}
    for r in ok:
        by_p.setdefault(r["participant"], []).append(r)

    per_p_path = os.path.join(args.outdir, "color_baseline_per_participant.csv")
    with open(per_p_path, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["participant", "n_images"] + [f"{c}_mean" for c in agg_cols])
        for p in sorted(by_p):
            recs = by_p[p]
            means = [np.mean([rec[c] for rec in recs]) for c in agg_cols]
            wri.writerow([p, len(recs)] + [f"{m:.6f}" for m in means])

    # ---- VLM-compatible greenness.csv (drop-in for the VLM rater) ----------
    # Identical schema to Step1_VLM_Greenness/.../greenness.csv so the SAME R
    # statistical pipeline (Table 1, Table 2, stress correlation) can be run
    # on this file exactly as on the VLM output, making the comparison fair.
    metric = args.vlm_greenness_metric
    if args.vlm_rescale == "x10":
        # Purely absolute: Greenness = 10 * raw_fraction. No offset, no
        # stretching, no dataset min/max. For metrics that may go outside
        # [0,1] (e.g. exg_mean) we clip the fraction to [0,1] first so
        # Greenness stays in [0, 10].
        def to_green(v):
            return 10.0 * max(0.0, min(1.0, float(v)))
    else:
        def to_green(v):
            return float(v)

    # Minimal greenness.csv: just the three columns the stats pipeline
    # actually uses (two join keys + the score). Column NAMES kept as
    # PictureId / Folder so the merge against the VLM file still works.
    green_path = os.path.join(args.outdir, "greenness.csv")
    with open(green_path, "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["PictureId", "Folder", "Greenness"])
        for r in ok:
            wri.writerow([
                r["image"],                       # PictureId (filename incl. ext)
                r["participant"],                 # Folder (participant id)
                f"{to_green(r[metric]):.6f}",     # Greenness
            ])

    print(f"\nWrote {per_image_path}  ({len(rows)} rows)")
    print(f"Wrote {per_p_path}  ({len(by_p)} participants)")
    print(f"Wrote {green_path}  ({len(ok)} rows; VLM-compatible, "
          f"Greenness={metric}, rescale={args.vlm_rescale})")
    if failed:
        print(f"WARNING: {len(failed)} image(s) failed to process; "
              f"see the 'error' column.", file=sys.stderr)
        for r in failed[:10]:
            print(f"  {r['participant']}/{r['image']}: {r['error']}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
