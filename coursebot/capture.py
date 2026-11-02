from __future__ import annotations

import time
import ctypes
import sys
import threading
from contextlib import contextmanager
from functools import lru_cache
from dataclasses import dataclass

import cv2
import mss
import numpy as np

from .windows import WindowInfo


_overlay_windows: set[int] = set()
_overlay_lock = threading.RLock()


@lru_cache(maxsize=1)
def _display_affinity():
    if sys.platform != "win32" or sys.getwindowsversion().build < 19041:
        return None
    from ctypes import wintypes
    api = ctypes.WinDLL("user32", use_last_error=True).SetWindowDisplayAffinity
    api.argtypes = [wintypes.HWND, wintypes.DWORD]
    api.restype = wintypes.BOOL
    return api


def register_capture_overlay(hwnd: int) -> bool:
    """只登记本进程悬浮窗；截图排除仅在本程序抓帧时启用。"""
    api = _display_affinity()
    if api is None or not hwnd:
        return False
    with _overlay_lock:
        if not api(hwnd, 0x11):  # WDA_EXCLUDEFROMCAPTURE, Windows 10 2004+
            return False
        api(hwnd, 0)
        _overlay_windows.add(hwnd)
    return True


def unregister_capture_overlay(hwnd: int) -> None:
    with _overlay_lock:
        _overlay_windows.discard(hwnd)


@contextmanager
def exclude_capture_overlays():
    # 同一进程的预览与自动化抓帧共用此锁，避免一次抓帧过早恢复另一帧的排除状态。
    with _overlay_lock:
        api = _display_affinity()
        protected = []
        try:
            if api is not None:
                for hwnd in _overlay_windows:
                    if api(hwnd, 0x11):
                        protected.append(hwnd)
            yield
        finally:
            for hwnd in protected:
                api(hwnd, 0)


@dataclass
class Frame:
    image: np.ndarray  # BGR
    left: int
    top: int
    timestamp: float

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


class ScreenCapture:
    """抓取目标窗口客户区。坐标均为屏幕物理像素。"""

    def __init__(self, window: WindowInfo):
        self.window = window
        self._sct = None
        self._mss_factory = getattr(mss, "MSS", None)
        if self._mss_factory is None:  # 兼容 mss 9.x
            self._mss_factory = mss.mss

    def open(self) -> None:
        if self._sct is None:
            self._sct = self._mss_factory()

    def close(self) -> None:
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None

    def grab(self) -> Frame:
        self.open()
        region = {
            "left": int(self.window.left),
            "top": int(self.window.top),
            "width": int(self.window.width),
            "height": int(self.window.height),
        }
        with exclude_capture_overlays():
            shot = self._sct.grab(region)
        arr = np.asarray(shot, dtype=np.uint8)
        bgr = np.ascontiguousarray(arr[:, :, :3])
        return Frame(bgr, self.window.left, self.window.top, time.time())

    def grab_screen(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        self.open()
        left, top, width, height = rect
        with exclude_capture_overlays():
            shot = self._sct.grab(
                {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
            )
        arr = np.asarray(shot, dtype=np.uint8)
        return np.ascontiguousarray(arr[:, :, :3])

    def save(self, path: str) -> None:
        frame = self.grab()
        cv2.imwrite(path, frame.image)
