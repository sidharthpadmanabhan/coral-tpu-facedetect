#!/usr/bin/env python3
"""Tkinter launcher + live viewer for Coral Edge TPU face detection.

Lists available camera sources (probed via OpenCV) and the screen/monitors
(discovered via mss), lets you pick one, adjust the confidence threshold, and
runs detection **in-process**, drawing the annotated camera output directly
into a Tkinter region. This avoids ``cv2.imshow`` entirely, which fails on
headless / non-GUI OpenCV builds on Windows.

    python detect_face_ui.py

Controls: press q or ESC in the Tkinter window (or click Stop) to end the
current detection stream and return to the launcher.
"""

import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_SCRIPT = os.path.join(HERE, "detect_face.py")


def probe_cameras(max_indices=6):
    """Return a list of integer camera indices that OpenCV can open.

    We try indices 0..max_indices-1, open each briefly, and keep the ones that
    successfully read a frame. Done in a background thread so the UI stays
    responsive.
    """
    import cv2

    found = []
    for idx in range(max_indices):
        cap = cv2.VideoCapture(idx)
        ok = False
        if cap.isOpened():
            ok, _ = cap.read()
        cap.release()
        if ok:
            found.append(idx)
    return found


def list_monitors():
    """Return a list of monitor index -> description using mss."""
    try:
        import mss
    except ImportError:
        return []
    with mss.mss() as sct:
        out = []
        for i, m in enumerate(sct.monitors):
            out.append((i, f"Monitor {i}  ({m['width']}x{m['height']})"))
        return out


def build_sources(on_done):
    """Probe cameras + monitors in a background thread; call on_done(list)."""
    def work():
        cams = probe_cameras()
        mons = list_monitors()
        items = []
        for c in cams:
            items.append(("webcam", c, f"Camera {c}"))
        for idx, desc in mons:
            items.append(("screen", idx, f"Screen - {desc}"))
        if not items:
            items.append(("webcam", 0, "Camera 0 (default, unverified)"))
        on_done(items)

    threading.Thread(target=work, daemon=True).start()


class DetectionRunner:
    """Runs face detection on a background thread and pushes frames to Tk.

    The runner imports ``detect_face`` (same package) and reuses its
    ``make_interpreter`` / ``detect`` / ``draw_detections`` helpers. Frames are
    converted to PIL images and handed to Tk via ``root.after`` so all widget
    updates happen on the main thread.
    """

    def __init__(self, root, canvas_label, status_var):
        self.root = root
        self.canvas_label = canvas_label
        self.status_var = status_var

        self._thread = None
        self._stop_event = threading.Event()
        self._interp_lock = threading.Lock()
        self.interpreter = None
        self.labels = []
        self._last_fps = 0.0
        self._frame_times = []

    def _ensure_interpreter(self):
        """Lazily build the interpreter once (slow first call)."""
        with self._interp_lock:
            if self.interpreter is not None:
                return
            import detect_face as df

            self.labels = df.load_labels(df.DEFAULT_LABELS)
            self.interpreter = df.make_interpreter(df.DEFAULT_MODEL)
            self.interpreter.allocate_tensors()

    def start(self, source, index, threshold, max_fps):
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(source, index, float(threshold), float(max_fps)),
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self._stop_event.set()

    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self, source, index, threshold, max_fps):
        import cv2
        from PIL import Image, ImageTk
        import numpy as np
        import detect_face as df

        try:
            self._ensure_interpreter()
        except Exception as exc:
            self._post_status(f"[error] interpreter init failed: {exc}")
            return

        frame_delay = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
        last = time.monotonic()

        if source == "screen":
            try:
                import mss
            except ImportError:
                self._post_status("[error] screen capture needs the `mss` package.")
                return
            sct = mss.mss()
            monitors = sct.monitors
            if index < 0 or index >= len(monitors):
                self._post_status(f"[error] monitor {index} not found ({len(monitors)} available).")
                sct.close()
                return
            mon = monitors[index]
        else:
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                self._post_status(f"[error] could not open camera {index}.")
                return

        self._post_status(f"Running on {source} {index}. Press q/ESC or Stop to end.")
        try:
            while not self._stop_event.is_set():
                if source == "screen":
                    shot = sct.grab(mon)
                    frame = np.array(shot)[:, :, :3]
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        time.sleep(0.01)
                        continue

                detections = df.detect(self.interpreter, frame, threshold)
                annotated = df.draw_detections(frame, detections, self.labels)

                # FPS calc (rolling 20-frame window).
                now = time.monotonic()
                self._frame_times.append(now)
                self._frame_times = self._frame_times[-20:]
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    if span > 0:
                        self._last_fps = (len(self._frame_times) - 1) / span

                # Convert BGR -> RGB -> PIL -> PhotoImage on this thread; only
                # the widget assignment is done on the main thread.
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                # Scale to a reasonable display width preserving aspect ratio.
                max_w = 640
                if img.width > max_w:
                    scale = max_w / img.width
                    img = img.resize((max_w, int(img.height * scale)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                def update_widget(p=photo, n=len(detections)):
                    try:
                        self.canvas_label.config(image=p)
                        self.canvas_label.image = p  # keep reference
                        self.status_var.set(
                            f"{n} face(s) | {self._last_fps:4.1f} FPS | {source} {index}"
                        )
                    except Exception:
                        pass

                self.root.after(0, update_widget)

                if frame_delay:
                    dt = time.monotonic() - last
                    if dt < frame_delay:
                        time.sleep(frame_delay - dt)
                last = time.monotonic()
        except Exception as exc:
            self._post_status(f"[error] {exc}")
        finally:
            if source == "screen":
                try:
                    sct.close()
                except Exception:
                    pass
            else:
                try:
                    cap.release()
                except Exception:
                    pass
            self._post_status("Stopped.")

    def _post_status(self, msg):
        def setit(m=msg):
            try:
                self.status_var.set(m)
            except Exception:
                pass
        try:
            self.root.after(0, setit)
        except Exception:
            pass


def main():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError as exc:
        print(f"[error] Tkinter is required for the UI: {exc}", file=sys.stderr)
        print("        On Windows Tkinter ships with Python; on Linux install python3-tk.",
              file=sys.stderr)
        raise SystemExit(1)

    import cv2  # noqa: F401  - ensure import errors surface early

    root = tk.Tk()
    root.title("Coral Edge TPU - Face Detection Launcher")
    root.geometry("900x560")
    root.minsize(760, 480)

    # ---- Layout: left = controls, right = camera/overlay region ----------
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    left = ttk.Frame(paned, width=320)
    paned.add(left, weight=0)

    ttk.Label(left, text="Select a source:", font=("Segoe UI", 11, "bold")).pack(
        pady=(8, 4), anchor=tk.W
    )

    listbox = tk.Listbox(left, height=10, activestyle="dotbox")
    listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    sources_var = tk.StringVar(value="Probing sources...")
    status = ttk.Label(left, textvariable=sources_var, anchor=tk.W, wraplength=300)
    status.pack(fill=tk.X, padx=4, pady=(0, 6))

    ctrl = ttk.Frame(left)
    ctrl.pack(fill=tk.X, padx=4, pady=4)

    ttk.Label(ctrl, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
    thresh_var = tk.DoubleVar(value=0.5)
    thresh_scale = ttk.Scale(ctrl, from_=0.1, to=0.95, variable=thresh_var,
                             orient=tk.HORIZONTAL, length=200)
    thresh_scale.grid(row=0, column=1, sticky=tk.EW, padx=8)
    thresh_val = ttk.Label(ctrl, text="0.50", width=5)
    thresh_val.grid(row=0, column=2)

    def on_thresh(_=None):
        thresh_val.config(text=f"{thresh_var.get():.2f}")
    thresh_scale.configure(command=on_thresh)

    ttk.Label(ctrl, text="Max FPS:").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    fps_var = tk.DoubleVar(value=30.0)
    ttk.Spinbox(ctrl, from_=0, to=120, increment=1, textvariable=fps_var, width=6).grid(
        row=1, column=1, sticky=tk.W, pady=(6, 0))

    ctrl.columnconfigure(1, weight=1)

    btns = ttk.Frame(left)
    btns.pack(fill=tk.X, padx=4, pady=(6, 8))
    start_btn = ttk.Button(btns, text="Start detection")
    start_btn.pack(side=tk.LEFT)
    stop_btn = ttk.Button(btns, text="Stop")
    stop_btn.pack(side=tk.LEFT, padx=8)
    ttk.Button(btns, text="Quit", command=root.quit).pack(side=tk.RIGHT)

    # ---- Right region: camera output + detection overlay ----------------
    right = ttk.Frame(paned)
    paned.add(right, weight=1)

    ttk.Label(right, text="Camera output + detection overlay",
              font=("Segoe UI", 10, "bold")).pack(pady=(8, 2))

    view_frame = ttk.Frame(right, relief=tk.SUNKEN, borderwidth=1)
    view_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

    overlay_label = tk.Label(view_frame, text="No source running.\nSelect a source and click Start.",
                             bg="black", fg="#cfcfcf",
                             font=("Segoe UI", 10))
    overlay_label.pack(fill=tk.BOTH, expand=True)

    runner = DetectionRunner(root, overlay_label, sources_var)

    items = []  # list of (source, index, label)

    def populate(new_items):
        nonlocal items
        items = new_items
        listbox.delete(0, tk.END)
        if not items:
            listbox.insert(tk.END, "No sources found")
            sources_var.set("No sources found - you can still try the default camera.")
            return
        for _, _, label in items:
            listbox.insert(tk.END, label)
        listbox.selection_set(0)
        sources_var.set(f"Found {len(items)} source(s). Select one and click Start.")

    build_sources(populate)

    def on_start():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a source first.")
            return
        if not items:
            source, index = "webcam", 0
        else:
            source, index, _ = items[sel[0]]
        if runner.running():
            messagebox.showinfo("Already running", "Detection is already running. Stop it first.")
            return
        runner.start(source, index, thresh_var.get(), fps_var.get())

    def on_stop():
        runner.stop()

    start_btn.configure(command=on_start)
    stop_btn.configure(command=on_stop)

    # Keyboard shortcuts: q / ESC stop detection.
    def on_key(event):
        if event.keysym in ("q", "Escape"):
            runner.stop()
    root.bind("<KeyPress>", on_key)

    def on_close():
        runner.stop()
        root.quit()
    root.protocol("WM_DELETE_WINDOW", on_close)

    root.mainloop()


if __name__ == "__main__":
    main()