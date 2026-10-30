from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception as exc:  # pragma: no cover - 在缺少依赖时给出友好错误
    RapidOCR = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class OcrLine:
    text: str
    score: float
    x0: int
    y0: int
    x1: int
    y1: int
    poly: tuple[tuple[int, int], ...] = ()

    @property
    def width(self) -> int:
        return max(1, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def cx(self) -> int:
        return (self.x0 + self.x1) // 2

    @property
    def cy(self) -> int:
        return (self.y0 + self.y1) // 2

    def contains(self, x: int, y: int, pad: int = 0) -> bool:
        return (
            self.x0 - pad <= x <= self.x1 + pad
            and self.y0 - pad <= y <= self.y1 + pad
        )

    def expanded(self, pad: int) -> tuple[int, int, int, int]:
        return self.x0 - pad, self.y0 - pad, self.x1 + pad, self.y1 + pad


class OcrError(RuntimeError):
    pass


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200b", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[\s\u3000·•,，。.;；:：!！?？、\"'“”‘’()（）\[\]【】<>《》_\-—/\\|]+", "", text)


def similarity(a: str, b: str) -> float:
    import difflib

    a, b = compact_text(a), compact_text(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()


class TextOCR:
    """RapidOCR 封装：全文检测 + 单行快速识别。"""

    def __init__(self, max_width: int = 2400):
        if RapidOCR is None:  # pragma: no cover
            raise OcrError(
                "未安装 OCR 依赖 rapidocr_onnxruntime，请先执行 pip install -r requirements.txt"
            ) from _IMPORT_ERROR
        self.max_width = max(800, int(max_width))
        self._lock = threading.RLock()
        self._full = None
        self._line = None

    def _ensure(self) -> None:
        if self._full is not None:
            return
        with self._lock:
            if self._full is None:
                self._full = RapidOCR(use_angle_cls=False)
            if self._line is None:
                self._line = RapidOCR(use_text_det=False, use_angle_cls=False)

    def warmup(self) -> None:
        self._ensure()
        dummy = np.full((64, 320, 3), 255, dtype=np.uint8)
        cv2.putText(dummy, "00:00 / 00:00", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        try:
            self.line(dummy)
        except Exception:
            pass

    @staticmethod
    def _to_line(item, scale: float) -> OcrLine | None:
        try:
            box, text, score = item[0], str(item[1]), float(item[2])
        except (TypeError, ValueError, IndexError):
            return None
        text = normalize_text(text)
        if not text:
            return None
        pts: list[tuple[int, int]] = []
        for point in box:
            try:
                pts.append((int(round(float(point[0]) / scale)), int(round(float(point[1]) / scale))))
            except (TypeError, ValueError, IndexError):
                continue
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return OcrLine(text, score, min(xs), min(ys), max(xs), max(ys), tuple(pts))

    def full(self, image: np.ndarray) -> list[OcrLine]:
        """识别整张页面。返回按从上到下、从左到右排序的文本行。"""
        self._ensure()
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        scale = min(1.0, self.max_width / max(1, w))
        if scale < 0.999:
            resized = cv2.resize(
                image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
            )
        else:
            resized = image
        try:
            with self._lock:
                result, _ = self._full(resized)  # type: ignore[misc]
        except Exception as exc:
            raise OcrError(f"OCR 识别失败：{exc}") from exc
        lines: list[OcrLine] = []
        for item in result or []:
            line = self._to_line(item, scale)
            if line and line.score >= 0.35:
                lines.append(line)
        lines.sort(key=lambda x: (x.y0, x.x0))
        return lines

    def line(self, crop: np.ndarray) -> str:
        """识别一行文本（不跑检测模型，约 30-60ms）。"""
        self._ensure()
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return ""
        h, w = crop.shape[:2]
        if h < 36:
            factor = min(3.0, 48.0 / max(1, h))
            crop = cv2.resize(crop, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_CUBIC)
        try:
            with self._lock:
                result, _ = self._line(crop)  # type: ignore[misc]
        except Exception:
            return ""
        if not result:
            return ""
        texts = [normalize_text(str(item[1])) for item in result if len(item) >= 2]
        return " ".join(t for t in texts if t).strip()
