#!/usr/bin/env python3
"""Download the Edge TPU face-detection model and label map.

This fetches Google's SSD MobileNet V2 face model that has been compiled for the
Coral Edge TPU, plus its label file, into the ``models`` directory next to this
script. Run once before running ``detect_face.py``.

    python download_models.py

The files are cached locally; re-running will skip files that already exist
unless you pass ``--force``.
"""

import argparse
import os
import sys
import urllib.request

# Edge TPU compiled SSD MobileNet V2 face model (quantized, postprocess built in).
MODEL_URL = (
    "https://github.com/google-coral/edgetpu/raw/master/"
    "test_data/ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
)

# CPU-compatible twin of the Edge TPU model (no custom ops). detect_face.py uses
# this as a fallback when the Coral accelerator / Edge TPU runtime is absent.
CPU_MODEL_URL = (
    "https://github.com/google-coral/edgetpu/raw/master/"
    "test_data/ssd_mobilenet_v2_face_quant_postprocess.tflite"
)

# Label map for the face model (single class: "Face").
LABELS_CONTENT = "Face\\n"

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download(url: str, dest: str, force: bool = False) -> None:
    if os.path.exists(dest) and not force:
        print(f"[skip] {dest} already exists")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[get ] {url}")
    print(f"[save] {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # noqa: BLE001 - surface any network error clearly
        print(f"[error] failed to download {url}: {exc}", file=sys.stderr)
        if os.path.exists(dest):
            os.remove(dest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    model_path = os.path.join(MODELS_DIR, "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite")
    labels_path = os.path.join(MODELS_DIR, "face_labelmap.txt")

    try:
        download(MODEL_URL, model_path, force=args.force)
    except Exception:
        return 1

    # CPU-compatible fallback model (used by detect_face.py when no Edge TPU).
    cpu_model_path = os.path.join(
        MODELS_DIR, "ssd_mobilenet_v2_face_quant_postprocess.tflite"
    )
    try:
        download(CPU_MODEL_URL, cpu_model_path, force=args.force)
    except Exception:
        # Non-fatal: the Edge TPU model above is the primary one.
        print("[warn] CPU fallback model download failed; Edge TPU model still OK.")

    if not os.path.exists(labels_path) or args.force:
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        with open(labels_path, "w", encoding="utf-8") as fh:
            fh.write(LABELS_CONTENT)
        print(f"[save] {labels_path}")
    else:
        print(f"[skip] {labels_path} already exists")

    print("\nDone. Model files are in:", MODELS_DIR)
    print("Now run:  python detect_face.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())