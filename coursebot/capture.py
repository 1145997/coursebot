from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import mss
import numpy as np

from .windows import WindowInfo


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
        shot = self._sct.grab(region)
        arr = np.asarray(shot, dtype=np.uint8)
        bgr = np.ascontiguousarray(arr[:, :, :3])
        return Frame(bgr, self.window.left, self.window.top, time.time())

    def grab_screen(self, rect: tuple[int, int, int, int]) -> np.ndarray:
        self.open()
        left, top, width, height = rect
        shot = self._sct.grab(
            {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        )
        arr = np.asarray(shot, dtype=np.uint8)
        return np.ascontiguousarray(arr[:, :, :3])

    def save(self, path: str) -> None:
        frame = self.grab()
        cv2.imwrite(path, frame.image)
