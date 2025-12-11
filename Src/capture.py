import os
import ctypes
from ctypes import wintypes
from PIL import Image

try:
    import mss
except Exception:
    mss = None


def _get_primary_monitor_rect_windows():
    """Return primary monitor rect as (left, top, right, bottom) on Windows, or None."""
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", wintypes.LONG),
            ("top", wintypes.LONG),
            ("right", wintypes.LONG),
            ("bottom", wintypes.LONG),
        ]

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    MonitorEnumProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(RECT), wintypes.LPARAM
    )

    primary_rect = {}

    def _callback(hMonitor, hdcMonitor, lprcMonitor, dwData):
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(mi)
        res = ctypes.windll.user32.GetMonitorInfoW(hMonitor, ctypes.byref(mi))
        if not res:
            return True
        MONITORINFOF_PRIMARY = 1
        if mi.dwFlags & MONITORINFOF_PRIMARY:
            primary_rect['left'] = mi.rcMonitor.left
            primary_rect['top'] = mi.rcMonitor.top
            primary_rect['right'] = mi.rcMonitor.right
            primary_rect['bottom'] = mi.rcMonitor.bottom
            return False
        return True

    enum_proc = MonitorEnumProc(_callback)
    user32.EnumDisplayMonitors(0, 0, enum_proc, 0)

    if primary_rect:
        return (
            primary_rect['left'],
            primary_rect['top'],
            primary_rect['right'],
            primary_rect['bottom'],
        )
    return None


def capture_primary_monitor() -> Image.Image:
    """Capture the primary monitor and return a PIL Image.

    Raises RuntimeError if `mss` is not available.
"""
    if mss is None:
        raise RuntimeError("Missing dependency 'mss'. Install with: pip install mss")

    with mss.mss() as sct:
        monitors = sct.monitors

        monitor_choice = None
        if os.name == 'nt':
            try:
                rect = _get_primary_monitor_rect_windows()
                if rect:
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top
                    for m in monitors[1:]:
                        if (
                            m.get('left') == left
                            and m.get('top') == top
                            and m.get('width') == width
                            and m.get('height') == height
                        ):  # pragma: no cover - platform-specific
                            monitor_choice = m
                            break
            except Exception:
                monitor_choice = None

        if monitor_choice is None:
            monitor_choice = monitors[1] if len(monitors) >= 2 else monitors[0]

        sct_img = sct.grab(monitor_choice)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        return img
