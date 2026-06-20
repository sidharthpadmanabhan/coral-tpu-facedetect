# Coral Edge TPU - Face Detection

Real-time face detection running on the **Google Coral USB Accelerator** (Edge TPU),
using Google's `ai-edge-litert` (LiteRT) interpreter + the Coral `edgetpu` delegate,
with an Edge TPU–compiled SSD MobileNet V2 face model and OpenCV for webcam capture.

## Files

| File                 | Purpose                                                        |
| -------------------- | ------------------------------------------------------------- |
| `detect_face.py`     | Main program: live webcam face detection (or single image).   |
| `download_models.py` | One-time downloader for the Edge TPU face model + label map.  |
| `requirements.txt`   | Python dependencies.                                          |
| `models/`            | Created by `download_models.py` to cache the `.tflite` files. |

## 1. One-time setup

### a) Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate            # Windows (use `source venv/bin/activate` on macOS/Linux)
python -m pip install --upgrade pip
pip install -r requirements.txt
```
(You can re-activate later with `venv\Scripts\activate` from the project folder.)

### b) Install the Windows Edge TPU runtime (required for the Coral device)
`ai-edge-litert` alone does **not** include the native `edgetpu.dll`. On Windows:

1. Plug in the Coral USB Accelerator. Windows should install the WinUSB driver
   automatically the first time.
2. Download the Edge TPU runtime from Coral's site:
   `https://coral.ai/docs/accelerator/get-started/#edgetpu-runtime-on-windows`
   (the `edgetpu_runtime_…zip` package) and run `install.bat` it contains.
3. After install, `edgetpu.dll` is placed on the system PATH so the
   `ai-edge-litert` delegate loader (`load_delegate("edgetpu.dll")`) can find it.

If you skip this step, `detect_face.py` will print a clear error at startup
telling you the Edge TPU interpreter could not be created.

### c) Download the model
```bash
python download_models.py
```
This places in `models/`:
- `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite` — the detector.
- `face_labelmap.txt` — the label map (single class: `Face`).

> Requires internet on first run. Files are cached; re-run with `--force` to refresh.

## 2. Run

### Live webcam
```bash
python detect_face.py
```
- Press `q` or `ESC` to quit.
- If the wrong camera opens, try `--device 1` (etc.).

### Tune confidence
```bash
python detect_face.py --threshold 0.6
```

### Run on a photo instead of the webcam
```bash
python detect_face.py --image photo.jpg --output faces.jpg
```

### Run on a UI
```bash
python detect_face_ui.py
```
If `--output` is omitted, the annotated image is shown in a window.

## 3. CLI options
```
--model      Path to .tflite Edge TPU model (default: models/...face...edgetpu.tflite)
--labels     Path to label map .txt
--threshold  Confidence threshold (default 0.5)
--device     Webcam index (default 0)
--image      Run once on this image file instead of the webcam
--output     When using --image, write annotated result to this path
--max-fps    Cap webcam polling rate (default 30, 0 = uncapped)
```

## Troubleshooting

> First step: run the diagnostic. It pinpoints exactly which of the layers
> below is failing (Python package, delegate loader, `edgetpu.dll`, or the USB
> device itself).
> ```bash
> python check_tpu.py
> ```
> It exits `0` only when the Edge TPU is actually usable.

The Edge TPU stack has **three independent layers** that must all be present;
the diagnostic checks each one. Fix them top-to-bottom:

### Layer 1 — Windows USB driver / device enumeration
The Coral USB Accelerator identifies as **Google Vendor ID `18D1`**. If
`check_tpu.py` says *"No USB device with Google's vendor ID (VID_18D1) is
present"*, Windows hasn't recognized the stick:
  1. Use a **data-capable** USB cable (some cables are charge-only).
  2. Plug directly into a **USB 3.0 (blue) port** on the PC, not an unpowered hub.
  3. Open **Device Manager**. Look for an *Unknown device* / device with a
     yellow warning triangle. If found, install the WinUSB driver for it using
     [Zadig](https://zadig.akeo.ie/) (Options → List All Devices, select the
     Coral device, set the driver to **WinUSB**, click Replace Driver).
  4. If Device Manager shows nothing at all when you plug/unplug the stick, the
     USB controller may need a reset — reboot, or try a different port.

### Layer 2 — Windows Edge TPU runtime (`edgetpu.dll`)
`ai-edge-litert` does **not** ship `edgetpu.dll`; Coral's native runtime must be
installed. If `check_tpu.py` reports *"edgetpu.dll is NOT on the PATH"*:
  1. Download the Edge TPU runtime for Windows from Coral's site:
     `https://coral.ai/docs/accelerator/get-started/#edgetpu-runtime-on-windows`
     (the `edgetpu_runtime_…zip` package).
  2. Run the `install.bat` it contains. This places `edgetpu.dll` (and its
     dependencies) under `%USERPROFILE%\AppData\Local\EdgeTPU\runtime` and adds
     it to the user PATH.
  3. **Restart your terminal / VS Code** so the new PATH takes effect, then
     re-run `python check_tpu.py` — the *"Found edgetpu.dll on PATH"* line
     should appear.

### Layer 3 — Python delegate loader (`load_delegate`)
The `ai-edge-litert` package moved `load_delegate` between modules across
versions. `detect_face.py`'s `_try_load_edgetpu_delegate()` now tries every
known location (`ai_edge_litert.interpreter.load_delegate`,
`ai_edge_litert.delegate.load_delegate`, and the `tflite_runtime` fallback),
so this layer should "just work" once Layers 1 and 2 are fixed. If
`check_tpu.py` reports *"No load_delegate entry point found"*, reinstall the
package: `pip install --upgrade ai-edge-litert`.

### Other errors
- **`No Edge TPU devices detected` / interpreter error:** repeat steps 1b and
  the layers above; run `python check_tpu.py`.
- **`model file not found`:** run `python download_models.py`.
- **Black/empty webcam window:** wrong camera index — try `--device 1`.
- **`pycoral` import error on Windows:** Google's `pycoral` isn't on PyPI for
  Windows; this project deliberately uses `ai-edge-litert` instead, which ships
  cross-platform wheels. Ensure Python is 3.9–3.11.
