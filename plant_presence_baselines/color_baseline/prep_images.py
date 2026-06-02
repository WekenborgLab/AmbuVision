#!/usr/bin/env python3
"""
Server-side image prep — downsizing IDENTICAL to the AmbuVision VLM pipeline.

This reproduces, byte-for-byte where it matters, the in-memory downscale that
`Step1_VLM_Greenness/Imager.py` applies before sending an image to the VLM,
but PERSISTS the result to disk so the same prepped images can be reused by
the VLM run and the color baseline (no per-run re-decoding of 4K photos).

Faithful to Imager.py:
  * MAX_IMAGE_SIZE  : longest side, default 1280  (env-overridable)
  * JPEG_QUALITY    : default 85                   (env-overridable)
  * RGB convert, only downscale when longest > MAX (never upscale)
  * scale = MAX / longest ; new = max(1, round(dim * scale))
  * resampling = Image.LANCZOS
  * JPEG save with quality=Q, optimize=True
  * tolerant to truncated/huge files (LOAD_TRUNCATED_IMAGES, MAX_IMAGE_PIXELS)
  * optional pyvips streaming fast-path, PIL fallback (same as _downscale_to_jpeg)
  * stable byte read with retries (safe on networked workspace filesystems)

Output mirrors the input tree:
    <input>/<participant>/Foto-XXXX.png  ->  <output>/<participant>/Foto-XXXX.jpg

Usage:
    python prep_images.py --input /path/filtered --output /path/filtered_prepped
    MAX_IMAGE_SIZE=1280 JPEG_QUALITY=85 python prep_images.py -i ... -o ...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFile

# Match Imager.py tolerance to malformed / very large phone photos.
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    import pyvips  # optional ultra-low-memory streaming path
    _HAS_VIPS = True
except Exception:
    _HAS_VIPS = False

try:
    import imageio.v3 as iio  # optional forgiving fallback decoder
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# Defaults exactly as main.py: os.getenv("MAX_IMAGE_SIZE", 1280) / 85
DEF_MAX_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 1280))
DEF_QUALITY = int(os.getenv("JPEG_QUALITY", 85))


def _read_file_bytes_stable(path: Path, attempts: int = 4,
                            sleep_s: float = 0.25) -> bytes:
    """Tolerate transient short reads on networked FS (mirrors Imager.py)."""
    last = b""
    for _ in range(attempts):
        data = path.read_bytes()
        if data and data == last:
            return data
        last = data
        time.sleep(sleep_s)
    return last


def _open_image_anyhow(path: Path) -> Image.Image:
    """PIL first, imageio fallback — same strategy as Imager._open_image_anyhow."""
    data = _read_file_bytes_stable(path)
    try:
        im = Image.open(BytesIO(data))
        im.load()
        return im
    except Exception as e:
        last_err = e
    if _HAS_IMAGEIO:
        try:
            arr = iio.imread(data, extension=path.suffix.lower() or None)
            return Image.fromarray(arr).convert("RGB")
        except Exception as e:
            last_err = e
    raise last_err


def _downscale_pil(path: Path, max_size: int, quality: int) -> bytes:
    img = _open_image_anyhow(path)
    try:
        img = img.convert("RGB")
        if max_size > 0:
            w, h = img.size
            longest = max(w, h)
            if longest > max_size:
                scale = max_size / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    finally:
        img.close()


def _downscale_vips(path: Path, max_size: int, quality: int) -> bytes:
    img = pyvips.Image.new_from_file(str(path), access="sequential")
    if max_size > 0:
        longest = max(img.width, img.height)
        if longest > max_size:
            img = img.resize(max_size / float(longest))  # streaming resize
    return bytes(img.jpegsave_buffer(Q=quality))


def downscale_to_jpeg(path: Path, max_size: int, quality: int) -> bytes:
    """Identical dispatch to Imager._downscale_to_jpeg (vips fast-path, PIL fallback)."""
    if _HAS_VIPS:
        try:
            return _downscale_vips(path, max_size, quality)
        except Exception:
            return _downscale_pil(path, max_size, quality)
    return _downscale_pil(path, max_size, quality)


def _worker(args) -> tuple[str, str]:
    src, dst, max_size, quality, skip_existing = args
    try:
        if skip_existing and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return ("skip", src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data = downscale_to_jpeg(Path(src), max_size, quality)
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)  # atomic — safe if interrupted
        return ("ok", src)
    except Exception as e:
        return ("fail", f"{src} :: {type(e).__name__}: {e}")


def collect(input_dir: str) -> list[tuple[str, str]]:
    """(src, dst) preserving <participant>/<file>, output extension forced to .jpg."""
    pairs = []
    for participant in sorted(os.listdir(input_dir)):
        pdir = os.path.join(input_dir, participant)
        if not os.path.isdir(pdir):
            continue
        for fname in sorted(os.listdir(pdir)):
            if os.path.splitext(fname)[1].lower() in VALID_EXTS:
                pairs.append((participant, fname, os.path.join(pdir, fname)))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True,
                    help="Root with one subfolder per participant")
    ap.add_argument("-o", "--output", required=True,
                    help="Destination root (structure mirrored)")
    ap.add_argument("--max-size", type=int, default=DEF_MAX_SIZE,
                    help=f"Longest side in px, 0=no resize (default {DEF_MAX_SIZE})")
    ap.add_argument("--quality", type=int, default=DEF_QUALITY,
                    help=f"JPEG quality (default {DEF_QUALITY})")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) // 2),
                    help="Parallel processes")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip outputs that already exist (resumable)")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 1

    triples = collect(args.input)
    if not triples:
        print(f"ERROR: no images under {args.input}", file=sys.stderr)
        return 1
    print(f"{len(triples)} images across "
          f"{len({p for p, _, _ in triples})} participants; "
          f"max_size={args.max_size} quality={args.quality} "
          f"vips={'yes' if _HAS_VIPS else 'no'}", flush=True)

    tasks = []
    for participant, fname, src in triples:
        dst = os.path.join(args.output, participant,
                           os.path.splitext(fname)[0] + ".jpg")
        tasks.append((src, dst, args.max_size, args.quality,
                      args.skip_existing))

    ok = skip = fail = 0
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            status, info = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                failures.append(info)
            if i % 200 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  ok={ok} skip={skip} fail={fail}",
                      flush=True)

    print(f"\nDone. ok={ok} skip={skip} fail={fail} "
          f"-> {args.output}")
    if failures:
        print(f"WARNING: {fail} failed:", file=sys.stderr)
        for f in failures[:20]:
            print("  " + f, file=sys.stderr)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
