#!/usr/bin/env python3
"""Diagnostic helper for the Coral Edge TPU on Windows.

Reports, in order:
  1. Whether `ai-edge-litert` is importable and where its interpreter lives.
  2. Which `load_delegate` entry points are available (the current installed
     version exposes it at `ai_edge_litert.interpreter.load_delegate`, not as
     a separate `ai_edge_litert.delegate` module).
  3. Whether `edgetpu.dll` is on the PATH / can be loaded, and where it is.
  4. Whether the Coral USB device is visible to Windows (USB VID 18D1).
  5. Whether an Edge TPU delegate can actually be created.

Run it with:  python check_tpu.py
It exits non-zero if the TPU is not usable so it can be used in CI/scripts.
"""

import ctypes
import os
import subprocess
import sys
import warnings

# Silence a harmless AttributeError raised by ai_edge_litert's Delegate.__del__
# when the delegate constructor itself failed (no _library attribute set).
warnings.filterwarnings("ignore", message=".*Delegate.*")


def _ok(msg):
    print(f"[ OK ] {msg}")


def _warn(msg):
    print(f"[WARN] {msg}")


def _err(msg):
    print(f"[ERR ] {msg}")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_litert():
    section("1) ai-edge-litert package")
    try:
        import ai_edge_litert
        _ok(f"ai_edge_litert {getattr(ai_edge_litert, '__version__', '?')} "
            f"at {os.path.dirname(ai_edge_litert.__file__)}")
    except Exception as exc:
        _err(f"ai_edge_litert not importable: {exc}")
        _err("Install with: pip install -r requirements.txt")
        return None
    try:
        from ai_edge_litert.interpreter import Interpreter
        _ok("Interpreter import: ai_edge_litert.interpreter.Interpreter")
    except Exception as exc:
        _err(f"Could not import Interpreter: {exc}")
        return None
    return ai_edge_litert


def check_load_delegate():
    section("2) load_delegate entry points")
    candidates = [
        ("ai_edge_litert.interpreter", "load_delegate"),
        ("ai_edge_litert.delegate", "load_delegate"),
        ("ai_edge_litert", "delegate"),
        ("tflite_runtime.interpreter", "load_delegate"),
        ("tensorflow.lite", "load_delegate"),
    ]
    found = None
    for module_path, attr in candidates:
        try:
            mod = __import__(module_path, fromlist=[attr])
            candidate = getattr(mod, attr)
            if attr == "delegate":
                _ok(f"Found delegate module at {module_path} (call .load_delegate)")
                found = ("module", candidate)
            else:
                _ok(f"Found {attr} at {module_path}")
                found = ("func", candidate)
        except Exception:
            pass
    if not found:
        _err("No load_delegate entry point found in any known module.")
        _err("This is the most likely reason detection falls back to CPU.")
    return found


def _where_on_path(dll):
    """Search PATH-style dirs for a file (mimics DLL search basics)."""
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if not d:
            continue
        p = os.path.join(d, dll)
        if os.path.exists(p):
            return p
    return None


def check_edgetpu_dll():
    section("3) edgetpu.dll (Windows Edge TPU runtime)")
    dll_name = "edgetpu.dll"
    p = _where_on_path(dll_name)
    if p:
        _ok(f"Found {dll_name} on PATH: {p}")
    else:
        _warn(f"{dll_name} is NOT on the PATH.")
        # Try common install locations.
        guesses = [
            r"C:\Program Files\EdgeTPU\runtime\lib\edgetpu.dll",
            r"C:\Program Files\EdgeTPU\lib\edgetpu.dll",
            r"C:\Program Files (x86)\EdgeTPU\runtime\lib\edgetpu.dll",
            r"C:\Program Files (x86)\EdgeTPU\lib\edgetpu.dll",
            os.path.expanduser(r"~\edgetpu_runtime\lib\edgetpu.dll"),
            os.path.expanduser(r"~\AppData\Local\EdgeTPU\runtime\lib\edgetpu.dll"),
        ]
        for g in guesses:
            if os.path.exists(g):
                _ok(f"Found {dll_name} (not on PATH) at: {g}")
                _warn("Add its directory to PATH so the delegate loader can find it.")
                return g
        _err(f"{dll_name} not found in common locations.")
        _err("Install the Windows Edge TPU runtime from "
             "https://coral.ai/docs/accelerator/get-started/#edgetpu-runtime-on-windows")
    # Try loading it via ctypes to surface the real OS error.
    try:
        ctypes.CDLL(dll_name if not p else p)
        _ok(f"ctypes.CDLL('{dll_name}') loaded successfully.")
    except OSError as exc:
        _err(f"ctypes.CDLL('{dll_name}') failed: {exc}")
    return p


def check_usb_device():
    section("4) Coral USB device (VID 18D1)")
    if sys.platform != "win32":
        _warn("USB scan is Windows-only in this script; skipping.")
        return
    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_PnPEntity | "
             "Where-Object { $_.DeviceID -like '*VID_18D1*' } | "
             "Select-Object Name,DeviceID,Status,ConfigManagerErrorCode | "
             "Format-List"],
            capture_output=True, text=True, timeout=20,
        )
    except Exception as exc:
        _err(f"PowerShell USB scan failed: {exc}")
        return
    out = ps.stdout.strip()
    if not out:
        _err("No USB device with Google's vendor ID (VID_18D1) is present.")
        _err("The Coral accelerator is NOT enumerated by Windows. Check:")
        _err("  - The USB cable is data-capable and firmly plugged in.")
        _err("  - Try a different USB port (prefer USB 3.0).")
        _err("  - In Device Manager, look for an unknown/failed device and")
        _err("    update its driver to the WinUSB (libusb) driver using Zadig.")
        return
    _ok("Coral USB device found:")
    print(out)
    if "ConfigManagerErrorCode : 0" in out:
        _ok("Device reports no configuration errors.")


def try_build_delegate(found):
    section("5) Build Edge TPU delegate")
    if not found:
        _warn("Skipping (no load_delegate available).")
        return None
    kind, obj = found
    try:
        if kind == "module":
            delegate = obj.load_delegate("edgetpu.dll")
        else:
            delegate = obj("edgetpu.dll")
        _ok("Edge TPU delegate created successfully.")
        return delegate
    except Exception as exc:
        _err(f"Failed to create Edge TPU delegate: {exc}")
        _err("This usually means edgetpu.dll is missing or the device is not "
             "visible to the runtime.")
        return None


def main():
    print("Coral Edge TPU diagnostic — "
          f"Python {sys.version.split()[0]} on {sys.platform}")
    ai = check_litert()
    found = check_load_delegate()
    dll_path = check_edgetpu_dll()
    check_usb_device()
    delegate = try_build_delegate(found)

    section("SUMMARY")
    tpu_usable = bool(ai and found and dll_path and delegate)
    if tpu_usable:
        print("RESULT: Edge TPU is READY. detect_face.py should use it.")
        return 0
    print("RESULT: Edge TPU is NOT usable. See the ERR/WARN lines above.")
    print("After fixing, re-run: python check_tpu.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())