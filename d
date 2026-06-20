#!/usr/bin/env python3
"""Tkinter launcher for Coral Edge TPU face detection.

Lists available camera sources (probed via OpenCV) and the screen/monitors
(discovered via mss), lets you pick one, adjust the confidence threshold, and
launches detection on it. Detection runs in a background process that calls
``detect_face.py`` with the right flags, so the heavy inference + OpenCV window
don't block the UI.

    python detect_face_ui.py

Controls in the launched window: press q or ESC to stop detection and return
to this launcher.
"""

import os
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DETECT_SCRIPT = os.path.join(HERE, "detect_face.py")
PYTHON = sys.executable


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
            items.append(("screen", idx, f"Screen — {desc}"))
        if not items:
            items.append(("webcam", 0, "Camera 0 (default, unverified)"))
        on_done(items)

    threading.Thread(target=work, daemon=True).start()


def launch(source, index, threshold, max_fps):
    """Start detect_face.py as a subprocess for the chosen source."""
    cmd = [PYTHON, DETECT_SCRIPT, "--threshold", f"{float(threshold):.3f}",
           "--max-fps", f"{float(max_fps):.1f}"]
    if source == "screen":
        cmd += ["--source", "screen", "--monitor", str(index)]
    else:
        cmd += ["--source", "webcam", "--device", str(index)]
    # Run in its own process so the UI thread isn't blocked and the OpenCV
    # window has its own event loop. Inherits this terminal so output is visible.
    return subprocess.Popen(cmd, cwd=HERE)


def main():
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError as exc:
        print(f"[error] Tkinter is required for the UI: {exc}", file=sys.stderr)
        print("        On Windows Tkinter ships with Python; on Linux install python3-tk.",
              file=sys.stderr)
        raise SystemExit(1)

    root = tk.Tk()
    root.title("Coral Edge TPU — Face Detection Launcher")
    root.geometry("460x420")
    root.minsize(420, 380)

    ttk.Label(root, text="Select a source:", font=("Segoe UI", 11, "bold")).pack(pady=(12, 4))

    sources_var = tk.StringVar(value="Probing sources…")
    listbox = tk.Listbox(root, height=10, activestyle="dotbox")
    listbox.pack(fill=tk.BOTH, expand=True, padx=16, pady=4)

    status = ttk.Label(root, textvariable=sources_var, anchor=tk.W)
    status.pack(fill=tk.X, padx=16, pady=(0, 6))

    ctrl = ttk.Frame(root)
    ctrl.pack(fill=tk.X, padx=16, pady=4)

    ttk.Label(ctrl, text="Confidence:").grid(row=0, column=0, sticky=tk.W)
    thresh_var = tk.DoubleVar(value=0.5)
    thresh_scale = ttk.Scale(ctrl, from_=0.1, to=0.95, variable=thresh_var,
                             orient=tk.HORIZONTAL, length=200, command=lambda v: thresh_var.set(round(float(v), 2)))
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

    items = []  # list of (source, index, label)

    def populate(new_items):
        nonlocal items
        items = new_items
        listbox.delete(0, tk.END)
        if not items:
            listbox.insert(tk.END, "No sources found")
            sources_var.set("No sources found — you can still try the default camera.")
            return
        for _, _, label in items:
            listbox.insert(tk.END, label)
        listbox.selection_set(0)
        sources_var.set(f"Found {len(items)} source(s). Select one and click Start.")

    build_sources(populate)

    proc_holder = {"p": None}

    def on_start():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a source first.")
            return
        if not items:
            # Fallback: default camera 0
            source, index = "webcam", 0
        else:
            source, index, _ = items[sel[0]]
        if proc_holder["p"] is not None and proc_holder["p"].poll() is None:
            messagebox.showinfo("Already running", "Detection is already running. Stop it first.")
            return
        proc_holder["p"] = launch(source, index, thresh_var.get(), fps_var.get())
        sources_var.set(f"Started detection on {('camera '+str(index)) if source=='webcam' else ('screen '+str(index))}. Close its window (q/ESC) when done.")

    def on_stop():
        p = proc_holder.get("p")
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
            except Exception:
                pass
        proc_holder["p"] = None
        sources_var.set("Stopped.")

    btns = ttk.Frame(root)
    btns.pack(fill=tk.X, padx=16, pady=(6, 12))
    ttk.Button(btns, text="Start detection", command=on_start).pack(side=tk.LEFT)
    ttk.Button(btns, text="Stop", command=on_stop).pack(side=tk.LEFT, padx=8)
    ttk.Button(btns, text="Quit", command=root.quit).pack(side=tk.RIGHT)

    root.mainloop()


if __name__ == "__main__":
    main()