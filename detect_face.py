#!/usr/bin/env python3
"""Coral Edge TPU face detection.

Runs Google's SSD MobileNet V2 face model (Edge TPU compiled) via the
``ai-edge-litert`` interpreter + the Coral ``edgetpu`` delegate. Supports
live webcam, screen capture, or single-image inference.

    python detect_face.py                       # live webcam (press q/ESC to quit)
    python detect_face.py --source screen       # capture the primary monitor
    python detect_face.py --image photo.jpg     # one image, shown in a window
    python detect_face.py --image photo.jpg --output faces.jpg

See README.md for setup (Edge TPU runtime, model download) and CLI options.
"""

import argparse
import os
import sys
import time

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - import guard
    print(f"[error] missing dependency: {exc}", file=sys.stderr)
    print("        install requirements:  pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1)

# ai-edge-litert exposes the interpreter + delegate loader. Import lazily so a
# clear, actionable error is printed when the package (or the Edge TPU runtime)
# is missing instead of a raw ImportError buried deep in a traceback.
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        # Fallback for older/newer package layouts that re-export from litert.
        from litert.interpreter import Interpreter  # type: ignore
    except ImportError as exc:
        print("[error] could not import the LiteRT interpreter.", file=sys.stderr)
        print("        Make sure `ai-edge-litert` is installed:", file=sys.stderr)
        print("            pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(
    HERE, "models", "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
)
# CPU-compatible twin of the Edge TPU model (no custom ops). Used as a fallback
# when the Coral accelerator / edgetpu.dll is unavailable.
CPU_FALLBACK_MODEL = os.path.join(
    HERE, "models", "ssd_mobilenet_v2_face_quant_postprocess.tflite"
)
DEFAULT_LABELS = os.path.join(HERE, "models", "face_labelmap.txt")

# SSD MobileNet V2 detector I/O. The Edge TPU compiled model has the
# post-processing built in, so the interpreter returns four tensors:
#   0: boxes      shape [1, N, 4]   (ymin, xmin, ymax, xmax) in normalized coords
#   1: classes    shape [1, N]      (class id, 0-indexed here)
#   2: scores     shape [1, N]      (confidence 0..1)
#   3: count      shape [1]         (number of valid detections)
OUTPUT_TENSOR_ORDER = ("boxes", "classes", "scores", "count")


def load_labels(path):
    """Load a label map file (one label per line). Returns a 0-indexed list."""
    labels = []
    if not path or not os.path.exists(path):
        return labels
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def _try_load_edgetpu_delegate():
    """Attempt to load the Coral Edge TPU delegate. Returns the delegate or None.

    The ``ai-edge-litert`` package has moved ``load_delegate`` around across
    versions: older builds exposed ``ai_edge_litert.delegate.load_delegate``,
    while recent ones (>=2.x) expose it at ``ai_edge_litert.interpreter.load_delegate``.
    We try each known location and also accept a bare ``delegate`` submodule.
    """
    # (module_path, attr, is_submodule): if is_submodule, attr is a module whose
    # ``load_delegate`` function we call; otherwise attr is the function itself.
    candidates = (
        ("ai_edge_litert.interpreter", "load_delegate", False),
        ("ai_edge_litert.delegate", "load_delegate", False),
        ("ai_edge_litert", "delegate", True),
        ("tflite_runtime.interpreter", "load_delegate", False),
    )
    last_exc = None
    for module_path, attr, is_submodule in candidates:
        try:
            mod = __import__(module_path, fromlist=[attr])
            obj = getattr(mod, attr)
            if is_submodule:
                # `ai_edge_litert.delegate` is a module; call its load_delegate.
                return obj.load_delegate("edgetpu.dll")
            return obj("edgetpu.dll")
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        print(f"[warn] Edge TPU delegate could not be loaded: {last_exc}",
              file=sys.stderr)
    return None


def make_interpreter(model_path):
    """Create a LiteRT interpreter, preferring the Coral Edge TPU delegate.

    Returns the interpreter. If the Edge TPU runtime (``edgetpu.dll`` on
    Windows) is not installed, or the provided model is the Edge TPU variant
    (which contains custom ops the CPU can't run), we transparently fall back
    to the CPU-compatible model (``CPU_FALLBACK_MODEL``) if it exists, so the
    script still runs for testing, after printing a clear warning.
    """
    if not os.path.exists(model_path):
        print(f"[error] model file not found: {model_path}", file=sys.stderr)
        print("        run:  python download_models.py", file=sys.stderr)
        raise SystemExit(1)

    delegate = _try_load_edgetpu_delegate()

    if delegate is not None:
        try:
            return Interpreter(model_path=model_path, experimental_delegates=[delegate])
        except Exception as exc:
            print(f"[warn] Edge TPU interpreter creation failed: {exc}", file=sys.stderr)

    # No delegate (or it failed): use the CPU-compatible model if available.
    cpu_model = CPU_FALLBACK_MODEL
    if model_path != cpu_model and os.path.exists(cpu_model):
        print(
            "[warn] Edge TPU unavailable. Using the CPU-compatible model:\n"
            f"        {cpu_model}",
            file=sys.stderr,
        )
        return Interpreter(model_path=cpu_model)

    # No CPU fallback available; try the original model on CPU (may fail later).
    print(
        "[warn] Edge TPU delegate could not be loaded (edgetpu.dll missing?). "
        "Falling back to CPU inference.",
        file=sys.stderr,
    )
    return Interpreter(model_path=model_path)


def get_output_tensors(interpreter):
    """Read the four SSD output tensors into a dict keyed by OUTPUT_TENSOR_ORDER.

    The Edge TPU SSD model with built-in postprocessing exposes four output
    tensors in a fixed order: boxes, classes, scores, count. We read them via
    ``get_output_details()`` (which is robust across interpreter builds) and
    map by position.
    """
    out_details = interpreter.get_output_details()
    tensors = {}
    for i, name in enumerate(OUTPUT_TENSOR_ORDER):
        if i >= len(out_details):
            break
        idx = out_details[i]["index"]
        tensors[name] = interpreter.get_tensor(idx)
    return tensors


def detect(interpreter, image_bgr, threshold):
    """Run a single detection pass on a BGR image.

    Returns a list of detections (each with box in pixel coords + label/score),
    filtered by ``threshold``. Handles uint8/float32 input tensors and resizes
    the image to the model's expected input dimensions while preserving aspect
    ratio with black padding.
    """
    input_details = interpreter.get_input_details()
    input_detail = input_details[0]
    input_shape = input_detail["shape"]  # [1, H, W, C]
    model_h, model_w = int(input_shape[1]), int(input_shape[2])
    model_c = int(input_shape[3])
    is_floating = input_detail["dtype"] == np.float32

    # Resize with aspect-ratio-preserving padding (letterbox).
    ih, iw = image_bgr.shape[:2]
    scale = min(model_w / iw, model_h / ih)
    new_w, new_h = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(image_bgr, (new_w, new_h))
    canvas = np.zeros((model_h, model_w, model_c), dtype=np.uint8)
    top = (model_h - new_h) // 2
    left = (model_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    # SSD models expect RGB input.
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(canvas_rgb, axis=0)

    if is_floating:
        input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    interpreter.set_tensor(input_detail["index"], input_data)
    interpreter.invoke()
    out = get_output_tensors(interpreter)

    count = int(np.array(out["count"]).reshape(-1)[0])
    boxes = np.array(out["boxes"]).reshape(-1, 4)[:count]
    classes = np.array(out["classes"]).reshape(-1)[:count].astype(int)
    scores = np.array(out["scores"]).reshape(-1)[:count]

    pad_l = left
    pad_t = top

    detections = []
    for box, cls, score in zip(boxes, classes, scores):
        if score < threshold:
            continue
        ymin, xmin, ymax, xmax = box
        # Convert from letterboxed model coords back into original image coords.
        xmin = max(0.0, (xmin * model_w - pad_l) / new_w * iw)
        xmax = min(float(iw), (xmax * model_w - pad_l) / new_w * iw)
        ymin = max(0.0, (ymin * model_h - pad_t) / new_h * ih)
        ymax = min(float(ih), (ymax * model_h - pad_t) / new_h * ih)
        detections.append(
            {
                "box": [int(xmin), int(ymin), int(xmax), int(ymax)],
                "class": int(cls),
                "score": float(score),
            }
        )
    return detections


def draw_detections(image_bgr, detections, labels):
    """Annotate an image copy with bounding boxes + labels."""
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = labels[det["class"]] if det["class"] < len(labels) else "id:" + str(det["class"])
        text = label + ": " + ("%.2f" % det["score"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(0, y1 - baseline - 2)
        cv2.rectangle(
            annotated,
            (x1, ty),
            (x1 + tw + 4, ty + th + baseline),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 2, ty + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def run_screen(interpreter, labels, threshold, monitor_index, max_fps):
    """Run live detection on a screen (monitor) capture using mss.

    ``monitor_index`` 0 is the primary monitor; higher indices are additional
    monitors. Press q/ESC to quit.
    """
    try:
        import mss
    except ImportError:
        print("[error] screen capture needs the `mss` package:", file=sys.stderr)
        print("        pip install mss", file=sys.stderr)
        return 1

    print(f"[info] capturing screen (monitor {monitor_index}). Press q or ESC to quit.")
    frame_delay = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
    last = time.monotonic()
    with mss.mss() as sct:
        monitors = sct.monitors
        if monitor_index < 0 or monitor_index >= len(monitors):
            print(f"[error] monitor index {monitor_index} out of range "
                  f"(found {len(monitors)} monitors)", file=sys.stderr)
            return 1
        mon = monitors[monitor_index]
        try:
            while True:
                shot = sct.grab(mon)
                # mss returns BGRA; convert to BGR for OpenCV.
                frame = np.array(shot)[:, :, :3]
                detections = detect(interpreter, frame, threshold)
                annotated = draw_detections(frame, detections, labels)
                cv2.imshow("detect_face - screen (q/ESC to quit)", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                if frame_delay:
                    dt = time.monotonic() - last
                    if dt < frame_delay:
                        time.sleep(frame_delay - dt)
                last = time.monotonic()
        finally:
            cv2.destroyAllWindows()
    return 0


def run_image(interpreter, labels, threshold, image_path, output_path):
    if not os.path.exists(image_path):
        print(f"[error] image not found: {image_path}", file=sys.stderr)
        return 1
    image = cv2.imread(image_path)
    if image is None:
        print(f"[error] could not read image: {image_path}", file=sys.stderr)
        return 1

    detections = detect(interpreter, image, threshold)
    print(f"[{len(detections)} face(s)] detected in {image_path}")
    annotated = draw_detections(image, detections, labels)

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"[save] {output_path}")
    else:
        cv2.imshow("detect_face", annotated)
        print("[info] press any key to close the window")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


def run_webcam(interpreter, labels, threshold, device, max_fps):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[error] could not open webcam device {device}", file=sys.stderr)
        return 1

    print("[info] webcam running. Press q or ESC to quit.")
    frame_delay = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
    last = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] empty frame from webcam; retrying...", file=sys.stderr)
                continue

            detections = detect(interpreter, frame, threshold)
            annotated = draw_detections(frame, detections, labels)
            cv2.imshow("detect_face (q/ESC to quit)", annotated)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            if frame_delay:
                dt = time.monotonic() - last
                if dt < frame_delay:
                    time.sleep(frame_delay - dt)
            last = time.monotonic()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Path to .tflite Edge TPU model")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="Path to label map .txt")
    p.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold (default 0.5)")
    p.add_argument("--source", choices=("webcam", "screen"), default="webcam",
                   help="Live source: webcam (default) or screen capture")
    p.add_argument("--device", type=int, default=0, help="Webcam index (default 0)")
    p.add_argument("--monitor", type=int, default=0,
                   help="Screen/monitor index for --source screen (default 0 = primary)")
    p.add_argument("--image", default=None, help="Run once on this image file instead of the webcam")
    p.add_argument("--output", default=None, help="When using --image, write annotated result to this path")
    p.add_argument("--max-fps", type=float, default=30, help="Cap webcam polling rate (default 30, 0 = uncapped)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    labels = load_labels(args.labels)
    interpreter = make_interpreter(args.model)

    # Warm up the interpreter once so the first real frame isn't slow.
    try:
        interpreter.allocate_tensors()
    except Exception as exc:
        print(f"[error] failed to allocate tensors: {exc}", file=sys.stderr)
        return 1

    if args.image:
        return run_image(
            interpreter, labels, args.threshold, args.image, args.output
        )
    if args.source == "screen":
        return run_screen(interpreter, labels, args.threshold, args.monitor, args.max_fps)
    return run_webcam(interpreter, labels, args.threshold, args.device, args.max_fps)


if __name__ == "__main__":
    raise SystemExit(main())