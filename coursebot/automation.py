from __future__ import annotations

import logging
import queue
import re
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .capture import Frame, ScreenCapture
from .config import LOG_DIR, Settings, masked_key
from .llm import LLMClient, LLMError
from .ocr import OcrError, OcrLine, TextOCR
from .vision import (
    ExamQuestion,
    PageView,
    QuizView,
    TimeReading,
    analyze_page,
    annotate_frame,
    compact_text,
    find_button,
    find_next_button,
    find_quiz_tab,
    ocr_debug_text,
    parse_time_text,
    pick_next_section,
    preview_jpeg,
)
from .windows import (
    Clicker,
    MouseInputError,
    WindowInfo,
    cursor_position,
    focus_window,
    get_window_info,
    is_foreground,
    maximize_window,
    virtual_screen,
)


class StopAutomation(Exception):
    pass


class NavigationFailed(Exception):
    pass


class AutomationWorker(threading.Thread):
    def __init__(
        self,
        settings: Settings,
        window: WindowInfo,
        out_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True, name="coursebot-automation")
        self.settings = settings
        self.window = window
        self.queue = out_queue
        self.stop_event = stop_event
        self.capture: ScreenCapture | None = None
        self.ocr: TextOCR | None = None
        self.llm = LLMClient(settings)
        self.clicker = Clicker(dry_run=settings.dry_run)
        self.logger = logging.getLogger(f"coursebot.{uuid.uuid4().hex[:8]}")
        self.log_file: Path | None = None
        self._handler: logging.Handler | None = None
        self.view: PageView | None = None
        self.sections = []
        self.current_idx: int | None = None
        self.current_key: str | None = None
        self.time_box: OcrLine | None = None
        self.time_reading: TimeReading | None = None
        self.duration = 0.0
        self.last_current: float | None = None
        self.last_progress_wall = time.time()
        self.last_stall_attempt = 0.0
        self.resume_attempts = 0
        self.end_streak = 0
        self.fast_fail = 0
        self.last_full_scan = 0.0
        self.last_relocate = 0.0
        self.last_preview = 0.0
        self.last_status = 0.0
        self.last_logged_bucket = -1
        self.completion_handled = False
        self.navigating = False
        self.quiz_active = False
        self.exam_active = False
        self._chapter_quiz_done = False
        self._backfill_done = False
        self._backfill_active = False
        self.video_playing_unknown_time = False
        self._last_unknown_scan = 0.0
        self._unknown_since = 0.0
        self._last_icon_resume = 0.0
        self._last_video_retry = 0.0
        self._video_retry_count = 0
        self._last_outline_audit = 0.0
        self._last_video_tab_click = 0.0
        self.last_time_read_at = 0.0
        self._same_time_value: float | None = None
        self._same_time_since = 0.0
        self._page_video_key: str | None = None
        self._page_video_index = 0
        self._page_video_total = 0
        self.sections_finished = 0
        self.quizzes_solved = 0
        self._force_full = True
        self._debug_files: list[Path] = []
        self._stop_reason = "用户停止"

    # ------------------------------------------------------------------ 基础
    def _setup_logging(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
        self._handler = logging.FileHandler(self.log_file, encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(self._handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

    def _log(self, text: str, level: str = "info") -> None:
        getattr(self.logger, level if level in {"debug", "info", "warning", "error"} else "info")(text)
        self.queue.put({"type": "log", "level": level, "text": text})

    def _emit(self, kind: str | None = None, **payload: Any) -> None:
        if kind:
            payload["type"] = kind
        self.queue.put(payload)

    def _status(
        self,
        state: str,
        text: str,
        current: float | None = None,
        duration: float | None = None,
    ) -> None:
        cur = self.time_reading.current if self.time_reading else current
        dur = self.duration if duration is None else duration
        percent = 0.0
        if cur is not None and dur and dur > 0:
            percent = max(0.0, min(100.0, cur / dur * 100.0))
        self._emit(
            "status",
            state=state,
            text=text,
            current=cur,
            duration=dur,
            percent=percent,
        )

    def _sleep(self, seconds: float) -> None:
        end = time.time() + max(0.0, seconds)
        while time.time() < end:
            if self.stop_event.is_set():
                raise StopAutomation()
            time.sleep(min(0.15, max(0.02, end - time.time())))

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise StopAutomation()

    def _grab(self) -> Frame:
        if not self.capture:
            raise StopAutomation()
        last: Exception | None = None
        for _ in range(3):
            try:
                return self.capture.grab()
            except Exception as exc:
                last = exc
                self._sleep(0.3)
        raise RuntimeError(f"无法截取目标窗口：{last}")

    def _emit_preview(self, frame: Frame, force: bool = False) -> None:
        if not self.settings.preview_enabled:
            return
        now = time.time()
        if not force and now - self.last_preview < 1.0:
            return
        self.last_preview = now
        status = "OCR READY"
        if self.view:
            t = self.view.time
            if t:
                status = f"PLAY {t.current_text()} / {t.duration_text()}"
            if self.view.quiz.detected:
                status = "QUIZ DETECTED"
            elif self.settings.dry_run:
                status += "  [DRY RUN]"
        annotated = annotate_frame(frame.image, self.view, status, self.current_key)
        data = preview_jpeg(annotated, 1000, 82)
        if data:
            self._emit("preview", jpeg=data)

    def _save_debug(self, frame: Frame, view: PageView | None, tag: str) -> None:
        if not self.settings.save_debug_frames:
            return
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%H%M%S")
            base = LOG_DIR / f"{datetime.now():%Y%m%d}_{stamp}_{tag}"
            cv2.imwrite(str(base.with_suffix(".jpg")), frame.image)
            if view:
                base.with_suffix(".txt").write_text(ocr_debug_text(view), encoding="utf-8")
            self._debug_files.append(base.with_suffix(".jpg"))
            while len(self._debug_files) > 24:
                old = self._debug_files.pop(0)
                try:
                    old.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            pass

    # ------------------------------------------------------------------ 启动
    def run(self) -> None:
        try:
            self._setup_logging()
            self._log(f"自动化启动；窗口：{self.window.label()}")
            if getattr(self.settings, "llm_mode", "cloud") == "local":
                self._log(
                    f"LLM：本地模式 {self.settings.local_model} @ {self.settings.local_api_url}"
                )
            else:
                self._log(
                    f"LLM：{self.settings.model} / Key {masked_key(self.settings.api_key)}"
                )
            if self.settings.dry_run:
                self._log("试运行模式：只识别、不点击", "warning")
            self._preflight()
            self._start()
            self._main_loop()
        except StopAutomation:
            self._log("自动化已停止")
        except NavigationFailed as exc:
            self._stop_reason = str(exc)
            self._log(str(exc), "error")
        except Exception as exc:
            self._stop_reason = f"运行出错：{exc}"
            self._log(f"运行出错：{exc}", "error")
            self.logger.error(traceback.format_exc())
        finally:
            self._cleanup()
            self._emit("stopped", reason=self._stop_reason)

    def _cleanup(self) -> None:
        if self.capture:
            self.capture.close()
            self.capture = None
        if self._handler:
            try:
                self._handler.close()
            except Exception:
                pass
            self._handler = None

    def _preflight(self) -> None:
        self._status("starting", "正在加载 OCR 模型…")
        self._log("正在加载 OCR 模型（首次启动约需 2~5 秒）…")
        try:
            self.ocr = TextOCR(self.settings.ocr_max_width)
            self.ocr.warmup()
        except Exception as exc:
            raise RuntimeError(f"OCR 初始化失败：{exc}") from exc
        if not self.settings.auto_quiz:
            self._log("自动答题已关闭", "warning")
        self._log(f"识别引擎就绪，日志：{self.log_file}")

    def _start(self) -> None:
        countdown = max(0, int(self.settings.start_countdown))
        if countdown:
            for n in range(countdown, 0, -1):
                self._status("starting", f"{n} 秒后开始…")
                self._sleep(1.0)
        if not focus_window(self.window.hwnd):
            self._log("无法将目标窗口切到前台，请手动切换后重试", "warning")
            self._sleep(1.0)
        if not self._ensure_foreground():
            raise StopAutomation()
        # 浏览器窗口太小/被还原时，视频控制栏会跑出可视区，导致读不到时间和暂停按钮。
        maximize_window(self.window.hwnd)
        refreshed = get_window_info(self.window.hwnd)
        if refreshed is not None:
            self.window = refreshed
        self.capture = ScreenCapture(self.window)
        x, y, w, h = virtual_screen()
        if (
            self.window.left < x
            or self.window.top < y
            or self.window.right > x + w
            or self.window.bottom > y + h
        ):
            self._log("目标窗口未完全显示在屏幕内，部分识别/点击可能失败", "warning")
        self._log(f"窗口客户区：{self.window.width}×{self.window.height} @ ({self.window.left},{self.window.top})")
        self._status("scanning", "正在扫描页面…")
        self._wake_controls()
        self._sleep(0.15)
        frame = self._grab()
        view = self._scan(frame)
        # 先处理当前页的视频/测验；不要一启动就去批量补测，把学习目标页当成测验页。
        self._backfill_done = True
        if self.settings.auto_quiz and self.settings.dry_run:
            pending = [r for r in view.sections if r.pending_tasks == 1 and not r.completed]
            if pending:
                self._log(
                    f"[试运行] 发现 {len(pending)} 个“视频已看、测验未做”的小节，"
                    "真实运行时会先补做测验",
                    "dry",
                )
        self._apply_initial(frame, view)

    @staticmethod
    def _find_next_section_line(view: PageView) -> OcrLine | None:
        """找“下一节”，兼容 OCR 把“上一节 下一节”合成一行的情况。"""
        merged: OcrLine | None = None
        for line in view.lines:
            text = line.text.replace(" ", "").replace("\u3000", "")
            if "下一节" not in text:
                continue
            if "上一节" not in text:
                return line
            merged = line
        if merged is not None:
            # “上一节 下一节”合成一行：只点右半部分的“下一节”。
            x0 = merged.x0 + int(merged.width * 0.55)
            return OcrLine(
                "下一节",
                merged.score,
                x0,
                merged.y0,
                merged.x1,
                merged.y1,
                merged.poly,
            )
        return None

    def _looks_like_intro_page(self, view: PageView) -> bool:
        """识别“学习目标/本章导读”这类没有播放器的中间页。"""
        if view.time or view.quiz.detected or view.exam.detected:
            return False
        text = " ".join(line.text.strip() for line in view.lines)
        strong_cues = ("本章学习目标", "通过本章学习", "你需要掌握")
        if not any(cue in text for cue in strong_cues):
            return False
        return bool(self._find_next_section_line(view))

    def _retry_video_error(self, view: PageView) -> bool:
        """识别“网络错误/下载中途失败/尝试其他线路”，点线路重新加载。"""
        now = time.time()
        if now - self._last_video_retry < 2.5:
            return False
        text = " ".join(line.text for line in view.lines)
        if not any(
            key in text
            for key in ("下载中途失败", "尝试其他线路", "视频加载失败")
        ):
            return False
        self._video_retry_count += 1
        order = (
            ("公网1", "公网 1", "公网2", "公网 2", "重新加载")
            if self._video_retry_count % 2
            else ("公网2", "公网 2", "公网1", "公网 1", "重新加载")
        )
        for name in order:
            line = find_button(view.lines, [name], exact=False)
            if not line:
                continue
            self._last_video_retry = now
            self._log(f"视频加载失败，点击“{line.text}”重试", "warning")
            self._status("playing", "视频加载失败，正在重试…")
            self._click_line(line, f"视频错误重试-{line.text}")
            self._sleep(1.2)
            return True
        # 没有线路文字时，点视频中央尝试重新加载
        self._last_video_retry = now
        self._log("视频加载失败，点击视频区域重试", "warning")
        self._click_video_center()
        self._sleep(1.2)
        return True

    def _click_intro_next(self, view: PageView) -> bool:
        button = self._find_next_section_line(view)
        if not button:
            return False
        self._log("检测到学习目标/导读页，点击“下一节”继续", "warning")
        self._status("navigating", "学习目标页：进入下一节")
        self._click_line(button, "学习目标页-下一节")
        self._no_time_since = None
        self._force_full = True
        self._sleep(2.5)
        return True

    def _video_needs_play(self, view: PageView) -> bool:
        if view.task_completed is False:
            return True
        if view.current_idx is not None and view.sections:
            row = view.sections[view.current_idx]
            if row.pending_tasks == 2 and not row.completed:
                return True
        return False

    def _looks_static(self) -> bool:
        """比较视频区域两帧，判断画面是否完全静止（多半是暂停/待播放）。"""
        first = self._grab()
        self._sleep(0.8)
        second = self._grab()
        h, w = first.image.shape[:2]
        x0, x1 = int(w * 0.06), int(w * 0.68)
        y0, y1 = int(h * 0.12), int(h * 0.62)
        if x1 <= x0 or y1 <= y0:
            return False
        a = cv2.cvtColor(first.image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(second.image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        a = cv2.resize(a, (240, 140), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (240, 140), interpolation=cv2.INTER_AREA)
        diff = float(np.mean(cv2.absdiff(a, b)))
        self._log(f"视频区域两帧差异：{diff:.2f}")
        return diff < 0.8

    def _mark_unknown_playing(self, reason: str = "") -> None:
        if not self.video_playing_unknown_time:
            self.video_playing_unknown_time = True
            self._unknown_since = time.time()
            self._last_unknown_scan = 0.0
            self._log(
                "视频画面在变化，判定为播放中，但控制栏/时间暂时不可见"
                + (f"（{reason}）" if reason else "")
                + "；继续监控并定期尝试同步时间",
                "warning",
            )
        self.last_progress_wall = time.time()
        self._status("playing", "播放中（等待控制栏/时间出现）")

    def _find_quiz_tab_zoomed(self, frame: Frame) -> OcrLine | None:
        """顶部标签很小、整页 OCR 漏掉时，放大页面顶部再找“章节测验”。"""
        if not self.ocr:
            return None
        try:
            h, w = frame.image.shape[:2]
            ox, oy = int(w * 0.02), int(h * 0.06)
            crop = frame.image[oy : int(h * 0.30), ox : int(w * 0.36)]
            if crop.size == 0:
                return None
            crop = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            lines = self.ocr.full(crop)
            for line in lines:
                if "章节测验" not in line.text:
                    continue
                return OcrLine(
                    line.text,
                    line.score,
                    ox + line.x0 // 2,
                    oy + line.y0 // 2,
                    ox + line.x1 // 2,
                    oy + line.y1 // 2,
                )
        except Exception:
            return None
        return None

    def _click_video_tab(self, view: PageView, frame: Frame | None = None) -> bool:
        """页面有“视频”标签但当前停在学习目标/练习页时，先切到视频。"""
        if view.time:
            return False
        if time.time() - self._last_video_tab_click < 2.5:
            return False
        best = None
        for line in view.lines:
            text = compact_text(line.text)
            # 只接受“视频 / 1视频 / 2视频”这种标签，绝不能把
            # “观看时长需≥总时长90%…当前视频…”这种说明文字当标签。
            if not re.fullmatch(r"\d{0,2}视频", text):
                continue
            if line.y0 > view.height * 0.30:
                continue
            if line.x0 > view.width * 0.72:
                continue
            if best is None or line.y0 < best.y0:
                best = line
        if not best:
            return False
        if frame is not None:
            try:
                x0 = max(0, best.x0 - 12)
                y0 = max(0, best.y0 - 10)
                x1 = min(frame.image.shape[1], best.x1 + 12)
                y1 = min(frame.image.shape[0], best.y1 + 10)
                crop = frame.image[y0:y1, x0:x1]
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                blue = cv2.inRange(hsv, (90, 70, 80), (135, 255, 255))
                if float((blue > 0).mean()) > 0.18:
                    # 标签已经是蓝色选中态，再点也不会切页；交给播放按钮处理。
                    return False
            except Exception:
                pass
        self._last_video_tab_click = time.time()
        self._log(f"优先切换到视频标签：{best.text}", "warning")
        self._status("playing", "切换到视频…")
        self._click_line(best, "视频标签")
        self._sleep(1.2)
        return True

    @staticmethod
    def _is_empty_content_page(view: PageView) -> bool:
        text = "".join(compact_text(line.text) for line in view.lines)
        return any(
            key in text
            for key in ("暂无内容", "暂无课程内容", "暂无学习内容", "暂无资源")
        )

    def _click_empty_next(self, view: PageView) -> bool:
        """“暂无内容”页单帧判断，直接点下一节。"""
        if not self._is_empty_content_page(view):
            return False
        button = view.next_button or find_next_button(view.lines)
        if not button:
            self._log("页面显示暂无内容，但没找到“下一节”按钮", "warning")
            return False
        self._log("页面显示“暂无内容”，直接进入下一节", "warning")
        self._status("navigating", "暂无内容，进入下一节")
        self._click_line(button, "暂无内容-下一节")
        self._sleep(1.2)
        return True

    def _maybe_enter_chapter_quiz(self, view: PageView, frame: Frame | None = None) -> bool:
        """任务点已完成但章节测验未做（橙色1）时，优先点“章节测验”标签。"""
        tab = view.exam.quiz_tab or find_quiz_tab(view.lines, view.width, view.height)
        if not tab and frame is not None:
            tab = self._find_quiz_tab_zoomed(frame)
        if not tab:
            return False
        view.exam.quiz_tab = tab
        row = None
        if view.current_idx is not None and view.sections:
            row = view.sections[view.current_idx]
        elif self.current_key:
            for item in view.sections:
                if item.key == self.current_key:
                    row = item
                    break
        if row is not None and row.completed:
            return False
        ready = bool(row and row.pending_tasks == 1)
        if not ready and row is None and view.task_completed is True:
            ready = True
        if not ready:
            return False
        self._log(
            "任务点已完成、章节测验未做，点击“章节测验”标签",
            "warning",
        )
        self._enter_chapter_quiz(view)
        self._chapter_quiz_done = True
        return True

    def _try_start_pending_video(self) -> bool:
        """黄色任务点/未完成章节：必要时点击视频中央开始播放。"""
        try:
            if not self._looks_static():
                # 已经在播放，只是控制栏/时间还没出现
                self._mark_unknown_playing("画面有变化")
                return True
            self._wake_controls()
            self._sleep(0.2)
            frame = self._grab()
            if self.time_box:
                self._log(
                    "任务点未完成且已有时间，等待统一恢复逻辑判断",
                    "warning",
                )
                self._mark_unknown_playing("任务点未完成，等待播放按钮确认")
                return True
            else:
                x, y, found = self._find_center_play_button(frame)
                if not found:
                    self._log(
                        "未确认到中央播放按钮，先不点击，避免把正在播放的视频点暂停",
                        "warning",
                    )
                    self._mark_unknown_playing("静态画面但未确认暂停")
                    return True
                self._log("确认中央暂停按钮，点击开始播放", "warning")
                self._status("playing", "任务点未完成，正在开始播放…")
                self._click_screen(x, y, "视频中央播放按钮")
            self._sleep(2.4)
            self._wake_controls()
            self._sleep(0.3)
            frame = self._grab()
            view = self._scan(frame)
            if view.time:
                self._adopt_time(view.time)
                self.last_progress_wall = time.time()
                self.last_current = view.time.current
                if self._is_near_end(view.time):
                    self._advance("视频已结束")
                else:
                    self._ensure_playing()
                return True
            if view.exam.detected:
                self._handle_exam(view, frame)
                return True
            if view.quiz.detected:
                self._handle_quiz(view, frame)
                return True
            if not self._looks_static():
                self._mark_unknown_playing("点击后画面已变化")
                return True
            self._log("点击视频中央后仍未识别到播放器时间，继续观察", "warning")
            return False
        except StopAutomation:
            raise
        except Exception as exc:
            self._log(f"尝试开始播放失败：{exc}", "warning")
            return False

    def _apply_initial(self, frame: Frame, view: PageView) -> None:
        self._note_page_videos(view)
        if view.exam.detected:
            self._handle_exam(view, frame)
            frame = self._grab()
            view = self._scan(frame)
            if view.exam.result:
                self._advance("启动时完成章节测验")
            return
        if view.quiz.detected:
            self._handle_quiz(view, frame)
            frame = self._grab()
            view = self._scan(frame)
        if self._click_video_tab(view, frame):
            return
        if self._click_empty_next(view):
            return
        if self._maybe_enter_chapter_quiz(view, frame):
            return
        if view.time:
            if self._is_near_end(view.time):
                self._status("completed", "视频已结束，准备进入下一节")
                self._advance("开始检测到视频已结束")
                return
            self.last_progress_wall = time.time()
            self.last_current = view.time.current
            if view.task_completed is False:
                self._log(
                    f"任务点未完成（黄色），视频停在 {view.time.current_text()}，自动继续播放",
                    "warning",
                )
            self._status("playing", "已开始监控播放")
            self._ensure_playing()
            return
        if view.quiz_entry:
            self._click_line(view.quiz_entry, "开始答题")
            self._sleep(2.5)
            frame = self._grab()
            view = self._scan(frame)
            if view.quiz.detected:
                self._handle_quiz(view, frame)
                return
        # 控制栏可能自动隐藏。先移动鼠标唤出控制栏，再重新识别，不能贸然点“下一节”。
        self._log("未检测到播放时间，尝试移动鼠标唤出播放器控制栏…", "warning")
        self._wake_controls()
        self._sleep(0.25)
        frame = self._grab()
        view = self._scan(frame)
        if view.time:
            self.last_progress_wall = time.time()
            self.last_current = view.time.current
            if self._is_near_end(view.time):
                self._advance("视频已结束")
            else:
                self._ensure_playing()
            return
        if view.quiz.detected:
            self._handle_quiz(view, frame)
            return
        if view.start_button:
            self._click_line(view.start_button, "开始/继续学习")
            self._sleep(2.0)
            return
        if self._retry_video_error(view):
            return
        if self._looks_like_intro_page(view) and self._click_intro_next(view):
            return
        if self._video_needs_play(view):
            if self._try_start_pending_video():
                return
        # 给 SPA 一点加载时间
        self._log("仍未检测到播放器，等待页面加载…", "warning")
        self._sleep(4.0)
        self._wake_controls()
        self._sleep(0.2)
        frame = self._grab()
        view = self._scan(frame)
        if view.time:
            self.last_progress_wall = time.time()
            self.last_current = view.time.current
            if self._is_near_end(view.time):
                self._advance("视频已结束")
            else:
                self._ensure_playing()
            return
        if view.quiz.detected:
            self._handle_quiz(view, frame)
            return
        if view.quiz_entry:
            self._click_line(view.quiz_entry, "开始答题")
            self._sleep(2.0)
            return
        if self._retry_video_error(view):
            return
        if self._looks_like_intro_page(view) and self._click_intro_next(view):
            return
        if view.task_completed is False and not self._looks_static():
            self._mark_unknown_playing("页面无时间但画面在变化")
            return
        # 启动/首次扫描没有播放时间：先上下大幅翻目录，再前进，不要直接停。
        if self._audit_outline_large(frame, view):
            return
        if view.task_completed is True or self._should_audit_outline(view):
            if self._advance("任务点已完成，页面无视频"):
                return
        self._stop_reason = "未检测到视频播放器。可能是非视频章节，请手动进入视频小节后重新启动。"
        raise NavigationFailed(self._stop_reason)

    def _handle_unknown_playing(self, frame: Frame) -> None:
        """视频在播放但控制栏/时间一直不出现时的兜底监控。"""
        self._keep_controls_awake(force=True)
        self._sleep(0.25)
        now = time.time()
        if now - self._last_unknown_scan >= 6:
            self._last_unknown_scan = now
            frame = self._grab()
            view = self._scan(frame)
            if self._handle_scan_interrupts(view, frame):
                self.video_playing_unknown_time = False
                return
            if view.time:
                self.video_playing_unknown_time = False
                self._status("playing", f"时间已同步 {view.time.current_text()} / {view.time.duration_text()}")
                self._log("控制栏时间已恢复，继续监控播放进度", "success")
                return
            if self._looks_like_intro_page(view) and self._click_intro_next(view):
                self.video_playing_unknown_time = False
                return
            if self._retry_video_error(view):
                self.video_playing_unknown_time = False
                return
            if self._looks_static():
                self._log(
                    "控制栏时间不可见且画面静止，无法确认是否暂停；不自动点击",
                    "warning",
                )
                return
            else:
                if now - self._unknown_since > 20:
                    self._log("视频仍在播放，但控制栏/时间尚未出现；继续监控", "warning")
                    self._unknown_since = now
        self._status("playing", "播放中（控制栏隐藏，等待同步时间）")
        self._emit_preview(frame)
        self._sleep(max(0.1, self.settings.poll_interval))

    # ------------------------------------------------------------------ 主循环
    def _main_loop(self) -> None:
        while not self.stop_event.is_set():
            self._check_stop()
            now = time.time()
            if self._refresh_geometry():
                self._force_full = True
            if self.time_box and not self.quiz_active and not self.navigating:
                self._keep_controls_awake(force=True)
            frame = self._grab()
            now = time.time()

            if self.video_playing_unknown_time and self.time_box is None:
                self._handle_unknown_playing(frame)
                continue

            if self.time_box is None or self._force_full:
                self._force_full = False
                view = self._scan(frame)
                if self._handle_scan_interrupts(view, frame):
                    continue
                if view.time is None:
                    self._handle_no_time(frame, view)
                    continue
            else:
                reading = self._read_time_fast(frame)
                if reading:
                    self.fast_fail = 0
                    self._on_reading(reading)
                else:
                    self.fast_fail += 1
                    if self.fast_fail == 2 and now - getattr(self, "_last_wake", 0) >= 6:
                        self._last_wake = now
                        self._log("播放时间识别丢失，移动鼠标唤出控制栏后重试")
                        self._wake_controls()
                        self._sleep(0.35)
                        frame = self._grab()
                        reading = self._read_time_fast(frame)
                        if reading:
                            self.fast_fail = 0
                            self._on_reading(reading)
                            self._emit_preview(frame)
                            self._sleep(max(0.1, self.settings.poll_interval))
                            continue
                    if self.fast_fail >= 3 and now - self.last_relocate >= 8:
                        self._log("播放时间识别丢失，重新定位…", "warning")
                        self.last_relocate = now
                        view = self._scan(frame)
                        if self._handle_scan_interrupts(view, frame):
                            continue
                        if view.time is None:
                            self._handle_no_time(frame, view)
                            continue
                        self.fast_fail = 0

            now = time.time()
            self._check_stop()
            if self._maybe_resume_paused(frame):
                continue
            if self.time_reading and self.duration > 0:
                if self._is_near_end(self.time_reading):
                    self.end_streak += 1
                    pct = self.time_reading.percent
                    self._status("completed", f"视频接近结束（{pct:.0%}）")
                    if self.end_streak >= max(1, self.settings.completion_confirmations):
                        if self.completion_handled:
                            self._sleep(3.0)
                            continue
                        self.completion_handled = True
                        if self._try_next_video_on_page(frame):
                            continue
                        self._status("navigating", "视频已看完，准备进入下一节")
                        if not self._advance("视频播放完毕"):
                            return
                        continue
                else:
                    self.end_streak = 0
                    stalled = now - self.last_progress_wall
                    if stalled >= self.settings.stall_seconds:
                        if now - self.last_stall_attempt >= max(3.0, self.settings.resume_cooldown):
                            self._handle_stall(frame)
                            continue
                    elif stalled >= 3:
                        self._status("playing", f"播放中 {self._time_status_text()}")
            else:
                if now - self.last_full_scan >= 8:
                    view = self._scan(frame)
                    if self._handle_scan_interrupts(view, frame):
                        continue
                    if view.time is None:
                        self._handle_no_time(frame, view)
                        continue

            if now - self.last_full_scan >= self.settings.full_scan_interval and not self.quiz_active:
                view = self._scan(frame)
                if self._handle_scan_interrupts(view, frame):
                    continue

            self._emit_preview(frame)
            elapsed = time.time() - now
            self._sleep(max(0.1, self.settings.poll_interval - elapsed))

    def _refresh_geometry(self) -> bool:
        now = time.time()
        if now - getattr(self, "_last_geom_check", 0) < 4:
            return False
        self._last_geom_check = now
        info = get_window_info(self.window.hwnd)
        if not info:
            self._stop_reason = "目标窗口已关闭或最小化，自动化停止。"
            raise NavigationFailed(self._stop_reason)
        if (info.width, info.height) != (self.window.width, self.window.height) or (
            info.left,
            info.top,
        ) != (self.window.left, self.window.top):
            self._log(
                f"窗口位置/尺寸变化：{self.window.width}×{self.window.height} → {info.width}×{info.height}，重新定位",
                "warning",
            )
            self.window = info
            if self.capture:
                self.capture.window = info
            self.time_box = None
            self.time_reading = None
            self.duration = 0.0
            return True
        return False

    def _handle_scan_interrupts(self, view: PageView, frame: Frame) -> bool:
        self._note_page_videos(view)
        if view.exam.detected:
            if view.exam.result and self._chapter_quiz_done:
                self._advance("章节测验已完成")
                return True
            self._handle_exam(view, frame)
            if view.exam.result and self._chapter_quiz_done:
                self._advance("章节测验已完成")
            return True
        if view.quiz.detected:
            self._handle_quiz(view, frame)
            return True
        if self._click_video_tab(view, frame):
            return True
        if self._click_empty_next(view):
            return True
        if view.quiz_entry:
            self._log("检测到测验入口，点击进入")
            self._click_line(view.quiz_entry, "开始答题")
            self._sleep(2.5)
            self._force_full = True
            return True
        return False

    def _handle_no_time(self, frame: Frame, view: PageView) -> None:
        if view.exam.detected:
            self._handle_exam(view, frame)
            return
        if view.quiz.detected:
            self._handle_quiz(view, frame)
            return
        if self._click_video_tab(view, frame):
            return
        if self._click_empty_next(view):
            return
        if view.quiz_entry:
            self._click_line(view.quiz_entry, "开始答题")
            self._sleep(2.5)
            return
        if view.resume_button:
            self._click_line(view.resume_button, "继续学习")
            self._sleep(2.0)
            return
        if view.start_button:
            self._click_line(view.start_button, "开始播放")
            self._sleep(2.0)
            return
        if self._retry_video_error(view):
            return
        if self._maybe_enter_chapter_quiz(view, frame):
            return
        if self._looks_like_intro_page(view) and self._click_intro_next(view):
            return
        if view.task_completed is True and not view.quiz.detected and not view.exam.detected:
            self._log(
                "任务点已完成、页面没有播放时间；先大幅滚动目录查漏，再进入下一节",
                "warning",
            )
            if self._audit_outline_large(frame, view):
                return
            if self._advance("任务点已完成，页面无视频"):
                return
        if self._video_needs_play(view):
            if self._try_start_pending_video():
                return
        # 连续多次没时间，先唤出控制栏复检，再停止；不要因为页脚有“下一节”就跳章
        now = time.time()
        if not hasattr(self, "_no_time_since") or self._no_time_since is None:
            self._no_time_since = now
            self._log("没有识别到播放时间，移动鼠标唤出控制栏后复检…", "warning")
            self._wake_controls()
            self._sleep(0.25)
            new_frame = self._grab()
            new_view = self._scan(new_frame)
            if new_view.time:
                self._adopt_time(new_view.time)
                self.last_progress_wall = time.time()
                self.last_current = new_view.time.current
                return
            if new_view.quiz.detected:
                self._handle_quiz(new_view, new_frame)
                return
            if self._video_needs_play(new_view) and not self._looks_static():
                self._mark_unknown_playing("无时间但画面在变化")
                return
            return
        if now - self._no_time_since > 25:
            if self._video_needs_play(view) and not self._looks_static():
                self._mark_unknown_playing("长时间无时间但画面在变化")
                self._no_time_since = None
                return
            if self._should_audit_outline(view):
                if self._audit_outline_large(frame, view):
                    return
            if view.task_completed is True:
                if self._advance("任务点已完成，页面无视频"):
                    return
            self._stop_reason = "未检测到视频播放器，可能本节没有视频。请手动切换后重新启动。"
            raise NavigationFailed(self._stop_reason)
        if self._video_needs_play(view) and not self._looks_static():
            self._mark_unknown_playing("无时间但画面在变化")
            return
        self._sleep(2.0)
        self._force_full = True

    def _is_near_end(self, reading: TimeReading) -> bool:
        if self.duration <= 0:
            return False
        margin = max(0.5, float(self.settings.completion_margin))
        return reading.current >= self.duration - margin

    def _time_status_text(self) -> str:
        if not self.time_reading:
            return ""
        return f"{self.time_reading.current_text()} / {self.time_reading.duration_text()}（{self.time_reading.percent:.0%}）"

    def _adopt_time(self, reading: TimeReading) -> None:
        self.last_time_read_at = time.time()
        old_duration = self.duration
        if abs(reading.duration - old_duration) > max(5.0, old_duration * 0.08):
            self._log(f"视频时长：{reading.duration_text()}", "info")
        self.duration = reading.duration
        self.time_reading = reading
        self.time_box = reading.line
        self._no_time_since = None
        if self.last_current is None:
            self.last_current = reading.current
            self.last_progress_wall = time.time()
            self._same_time_value = reading.current
            self._same_time_since = time.time()
        elif reading.current > self.last_current + 0.5:
            self.last_progress_wall = time.time()
            self.last_current = reading.current
            self._same_time_value = reading.current
            self._same_time_since = time.time()

    def _on_reading(self, reading: TimeReading) -> None:
        now = time.time()
        self.last_time_read_at = now
        if self.last_current is None:
            self.last_current = reading.current
            self.last_progress_wall = now
            self._same_time_value = reading.current
            self._same_time_since = now
        else:
            delta = reading.current - self.last_current
            if delta > 45:
                # 明显的 OCR 误读，忽略这一帧
                self.fast_fail += 1
                return
            if delta < -5:
                # 同一页面内时间不应明显倒退，按 OCR 误读处理
                self.fast_fail += 1
                return
            if delta > 0.2:
                self.last_current = reading.current
                self.last_progress_wall = now
                self.resume_attempts = 0
                if (
                    self._same_time_value is None
                    or abs(reading.current - self._same_time_value) > 0.2
                ):
                    self._same_time_value = reading.current
                    self._same_time_since = now
        self.time_reading = reading
        self.duration = reading.duration
        bucket = int(reading.percent * 10)
        if bucket != self.last_logged_bucket and bucket >= 0:
            if self.last_logged_bucket < 0 or bucket > self.last_logged_bucket:
                self.last_logged_bucket = bucket
                self._log(f"播放进度 {reading.percent:.0%}（{self._time_status_text()}）")
        if now - self.last_status > 1.0:
            self.last_status = now
            if self._is_near_end(reading):
                self._status("completed", f"视频接近结束 {self._time_status_text()}")
            elif now - self.last_progress_wall >= self.settings.stall_seconds:
                self._status("paused", f"视频暂停或卡住 {self._time_status_text()}")
            else:
                self._status("playing", f"播放中 {self._time_status_text()}")
            self._emit_preview_now()

    def _emit_preview_now(self) -> None:
        # 主循环每轮会截一张图；这里不额外截图，避免开销
        pass

    def _read_time_fast(self, frame: Frame) -> TimeReading | None:
        if not self.ocr or not self.time_box or not self.duration:
            return None
        h, w = frame.image.shape[:2]
        pad = max(8, int(self.time_box.height * 0.55))
        x0 = max(0, self.time_box.x0 - pad)
        y0 = max(0, self.time_box.y0 - pad)
        x1 = min(w, self.time_box.x1 + pad)
        y1 = min(h, self.time_box.y1 + pad)
        if x1 - x0 < 20 or y1 - y0 < 10:
            return None
        crop = frame.image[y0:y1, x0:x1]
        text = self.ocr.line(crop)
        parsed = parse_time_text(text)
        if not parsed:
            return None
        current, duration = parsed
        if current > self.duration + 3:
            return None
        if self.last_current is not None and current < self.last_current - 5:
            return None
        # 用首次扫描到的时长作为基准，避免最后一位 OCR 抖动
        if abs(duration - self.duration) > max(5.0, self.duration * 0.08):
            duration = self.duration
        line = OcrLine(
            text=text,
            score=1.0,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        )
        return TimeReading(current, duration, line, text)

    # ------------------------------------------------------------------ 扫描 / 暂停
    def _scan(self, frame: Frame) -> PageView:
        if not self.ocr:
            raise RuntimeError("OCR 未初始化")
        try:
            lines = self.ocr.full(frame.image)
        except OcrError as exc:
            raise RuntimeError(str(exc)) from exc
        view = analyze_page(
            frame.image, lines, self.current_key, line_reader=self.ocr.line
        )
        self.view = view
        self.last_full_scan = time.time()
        if view.sections:
            self.sections = view.sections
            if view.current_idx is not None:
                self.current_idx = view.current_idx
                self.current_key = view.sections[view.current_idx].key
        if view.time:
            self._adopt_time(view.time)
        if view.quiz.detected:
            self._log(
                f"检测到题目：{view.quiz.kind}，{len(view.quiz.options)} 个选项",
                "warning",
            )
        self._emit_preview(frame, force=True)
        return view

    def _handle_stall(self, frame: Frame) -> None:
        self.last_stall_attempt = time.time()
        before = self.time_reading.current if self.time_reading else None
        if self.time_reading and self._is_near_end(self.time_reading):
            # 结尾处播放按钮会变成“重播”，此时点播放会把视频重头播放，必须先切下一节。
            self._log("画面停在结尾附近，优先尝试进入下一节，避免误点重播", "warning")
            self.completion_handled = True
            if not self._advance("结尾处画面停滞"):
                raise NavigationFailed(self._stop_reason)
            return
        self._log("播放时间长时间未变化，检查是否暂停/出现弹窗…", "warning")
        self._status("paused", "检测到暂停或弹窗")
        view = self._scan(frame)
        if self._retry_video_error(view):
            return
        if view.quiz.detected:
            self._handle_quiz(view, frame)
            return
        if view.quiz_entry:
            self._click_line(view.quiz_entry, "开始答题")
            self._sleep(2.0)
            return
        if view.time and before is not None and view.time.current > before + 0.8:
            self._log("画面已自行恢复播放")
            self.last_progress_wall = time.time()
            return
        # 整页 OCR 需要数秒，期间视频可能已经恢复；点击前用最新画面复核一次
        if before is not None:
            fresh = self._read_time_fast(self._grab_with_controls())
            if fresh and fresh.current > before + 0.8:
                self._log("复核发现视频已在播放，不执行恢复点击")
                self._on_reading(fresh)
                self.last_progress_wall = time.time()
                return
        if view.resume_button:
            self._click_line(view.resume_button, f"继续学习（{view.resume_button.text}）")
            self._sleep(2.0)
            self.last_progress_wall = time.time()
            return
        if not self.settings.auto_resume:
            self._status("paused", "视频暂停；自动恢复已关闭")
            return
        if self.resume_attempts >= self.settings.max_resume_attempts:
            self._status("paused", "多次尝试恢复播放失败")
            self._log("多次尝试恢复播放失败，等待人工处理", "warning")
            self._sleep(5.0)
            return

        # 时间 OCR 可能只是丢失，画面其实在动。必须静态才允许点播放。
        if not self._looks_static():
            self._log(
                "视频画面仍在变化，判定为播放中；不点击播放，避免误暂停",
                "warning",
            )
            self.last_progress_wall = time.time()
            self.resume_attempts = 0
            return

        if time.time() - self.last_time_read_at > 2.5:
            self._log(
                "播放时间已丢失，无法确认是否暂停；先不点击，等待时间恢复",
                "warning",
            )
            self.last_progress_wall = time.time()
            self.resume_attempts = 0
            return
        self._log(
            "时间已停滞，交由统一恢复逻辑确认左下角播放按钮",
            "warning",
        )
        self.last_progress_wall = time.time()

    @staticmethod
    def _has_center_triangle(patch: np.ndarray) -> bool:
        """播放圆钮中心必须有一个明显的三角形，排除舰船照片里的圆形误检。"""
        h, w = patch.shape[:2]
        if h < 16 or w < 16:
            return False
        total = float(h * w)
        for invert, threshold in ((False, 185), (False, 150), (True, 110), (True, 150)):
            mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, thresh = cv2.threshold(patch, threshold, 255, mode)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < total * 0.035 or area > total * 0.55:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter <= 0:
                    continue
                approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
                if len(approx) != 3:
                    continue
                m = cv2.moments(contour)
                if m["m00"] <= 0:
                    continue
                cx = m["m10"] / m["m00"]
                cy = m["m01"] / m["m00"]
                if abs(cx - w / 2) <= w * 0.22 and abs(cy - h / 2) <= h * 0.22:
                    return True
        return False

    def _find_center_play_button(self, frame: Frame) -> tuple[int, int, bool]:
        """OpenCV 找黑色视频区域中央的播放圆钮，找不到再用视频区域估算中心。"""
        img = frame.image
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 左侧内容区域，排除右侧目录
        sx0, sx1 = int(w * 0.04), int(w * 0.70)
        sy0, sy1 = int(h * 0.14), int(h * 0.90)
        video_rect: tuple[int, int, int, int] | None = None
        crop_mask = (gray[sy0:sy1, sx0:sx1] < 40).astype(np.uint8) * 255
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > (w * h) * 0.08:
                vx, vy, vw, vh = cv2.boundingRect(largest)
                if vw > w * 0.35 and vh > h * 0.25:
                    video_rect = (sx0 + vx, sy0 + vy, vw, vh)
        if video_rect is None:
            video_rect = (
                int(w * 0.06),
                int(h * 0.20),
                int(w * 0.62),
                int(h * 0.68),
            )
        vx, vy, vw, vh = video_rect
        goal_x = vx + vw / 2
        goal_y = vy + vh / 2
        best: tuple[float, int, int] | None = None
        roi = gray[max(0, vy):min(h, vy + vh), max(0, vx):min(w, vx + vw)]
        if roi.size:
            circles = cv2.HoughCircles(
                roi,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(60, min(roi.shape[:2]) // 4),
                param1=80,
                param2=30,
                minRadius=18,
                maxRadius=max(24, min(100, min(roi.shape[:2]) // 5)),
            )
            if circles is not None:
                for cx, cy, radius in circles[0]:
                    ax, ay = vx + float(cx), vy + float(cy)
                    if not (vx + radius * 0.5 < ax < vx + vw - radius * 0.5):
                        continue
                    if not (vy + radius * 0.5 < ay < vy + vh - radius * 0.5):
                        continue
                    r = int(radius)
                    patch = gray[
                        max(0, int(ay) - r) : min(h, int(ay) + r),
                        max(0, int(ax) - r) : min(w, int(ax) + r),
                    ]
                    if patch.size == 0 or float(patch.mean()) < 55:
                        continue
                    if not self._has_center_triangle(patch):
                        continue
                    dist = ((ax - goal_x) ** 2 + (ay - goal_y) ** 2) ** 0.5
                    if best is None or dist < best[0]:
                        best = (dist, int(ax), int(ay))
        if best is not None and best[0] <= min(vw, vh) * 0.25:
            return self.window.left + best[1], self.window.top + best[2], True
        return self.window.left + int(goal_x), self.window.top + int(goal_y), False

    def _click_bottom_play_control(self) -> None:
        """点播放器左下角的播放/暂停按钮（时间文字左侧固定位置）。"""
        if not self.time_box:
            self._click_video_center()
            return
        self._wake_controls()
        self._sleep(0.05)
        offset = max(45, int(self.time_box.height * 3.2))
        x = self.window.left + max(12, self.time_box.x0 - offset)
        y = self.window.top + self.time_box.cy
        self._click_screen(x, y, "左下角播放按钮")

    def _maybe_resume_paused(self, frame: Frame) -> bool:
        """唯一的自动恢复入口：8 秒同一时间 + 确认播放三角 + 15 秒冷却。"""
        if not self.settings.auto_resume or not self.time_box or not self.time_reading:
            return False
        now = time.time()
        if now - self.last_time_read_at > 2.0:
            return False
        if self._is_near_end(self.time_reading):
            return False
        since = self._same_time_since or self.last_progress_wall
        if now - since < 8.0:
            return False
        if now - self._last_icon_resume < 15.0:
            return False
        if self.resume_attempts >= 2:
            return False
        # 这个播放器的左下角始终画播放三角形（播放中也不变），
        # 不能再拿它当暂停依据；只认“同一时间 8 秒”+“画面静止”。
        if not self._looks_static():
            return False
        old_current = self.time_reading.current
        self.resume_attempts += 1
        self._last_icon_resume = now
        self._log(
            f"时间停在 {self.time_reading.current_text()} 已 8 秒，确认播放按钮后恢复",
            "warning",
        )
        self._status("paused", "确认暂停，恢复播放")
        self._click_bottom_play_control()
        self._sleep(3.0)
        check = self._grab()
        reading = self._read_time_fast(check)
        if reading and reading.current > old_current + 0.2:
            self._on_reading(reading)
            self._log("已恢复播放", "success")
            self.resume_attempts = 0
        else:
            self._log("恢复后 3 秒仍未确认进度，等待 15 秒后再试", "warning")
        return True

    def _click_video_center(self) -> None:
        try:
            frame = self._grab()
            x, y, detected = self._find_center_play_button(frame)
        except Exception:
            x = self.window.left + int(self.window.width * 0.37)
            y = self.window.top + int(self.window.height * 0.55)
            detected = False
        desc = "视频中央播放按钮（OpenCV定位）" if detected else "视频区域中央"
        self._click_screen(x, y, desc)

    def _ensure_playing(self) -> None:
        if not self.time_reading:
            return
        start = self.time_reading.current
        if self.duration and start >= self.duration - self.settings.completion_margin:
            return
        self._sleep(2.5)
        frame = self._grab_with_controls()
        reading = self._read_time_fast(frame)
        if reading is None:
            # 播放器控制栏可能自动隐藏，或时间框已失效；整页确认一次，避免误点。
            view = self._scan(frame)
            if view.quiz.detected:
                self._handle_quiz(view, frame)
                return
            reading = view.time
        if reading and reading.current > start + 0.5:
            self._on_reading(reading)
            return
        if reading is None:
            self._log("暂时无法确认视频状态，先不点击播放，交由卡顿检测处理", "warning")
            self.last_progress_wall = time.time()
            return
        # 再观察 1.2 秒，排除 OCR 抖动/短暂缓冲
        self._sleep(1.2)
        frame = self._grab_with_controls()
        reading = self._read_time_fast(frame)
        if reading and reading.current > start + 0.4:
            self._on_reading(reading)
            return

        # 关键：时间识别失败≠视频暂停。先看视频画面是否还在变化，
        # 避免把正在播放的视频误点暂停。
        if not self._looks_static():
            self._log("两帧仍在变化，判定视频正在播放，取消暂停判断", "warning")
            self.last_progress_wall = time.time()
            return

        self._log(
            "启动时时间停滞，等待统一恢复逻辑确认播放按钮",
            "warning",
        )
        self.last_progress_wall = time.time()
        return

    def _wake_controls(self) -> None:
        """在视频区域内做几次明显的鼠标移动，让自动隐藏的控制栏出现。"""
        if not is_foreground(self.window.hwnd):
            return
        w, h = self.window.width, self.window.height
        points = [
            (self.window.left + int(w * 0.34), self.window.top + int(h * 0.64)),
            (self.window.left + int(w * 0.56), self.window.top + int(h * 0.80)),
            (self.window.left + int(w * 0.43), self.window.top + int(h * 0.72)),
        ]
        try:
            import ctypes

            for x, y in points:
                ctypes.windll.user32.SetCursorPos(int(x), int(y))
                time.sleep(0.12)
        except Exception:
            pass

    def _grab_with_controls(self) -> Frame:
        self._keep_controls_awake(force=True)
        self._sleep(0.25)
        return self._grab()

    def _keep_controls_awake(self, force: bool = False) -> None:
        """在视频区域内左右移动鼠标，防止播放器控制栏（含时间）自动隐藏。"""
        now = time.time()
        if not force and now - getattr(self, "_last_jiggle", 0) < 1.8:
            return
        self._last_jiggle = now
        if not is_foreground(self.window.hwnd):
            return
        self._jiggle_toggle = 1 - getattr(self, "_jiggle_toggle", 0)
        w, h = self.window.width, self.window.height
        x1 = self.window.left + int(w * (0.38 if self._jiggle_toggle else 0.52))
        x2 = self.window.left + int(w * (0.52 if self._jiggle_toggle else 0.38))
        y = self.window.top + int(h * 0.76)
        try:
            import ctypes

            ctypes.windll.user32.SetCursorPos(int(x1), int(y))
            time.sleep(0.08)
            ctypes.windll.user32.SetCursorPos(int(x2), int(y))
        except Exception:
            pass

    # ------------------------------------------------------------------ 章节测验
    def _scroll_exam(self, up: bool, amount: int = 5, quiet: bool = False) -> None:
        direction = "上" if up else "下"
        if not quiet:
            self._log(f"滚动页面（{direction}）")
        if self.settings.dry_run:
            if not quiet:
                self._log(f"[试运行] 将滚动章节测验（{direction}）", "dry")
            return
        if not self._ensure_foreground():
            return
        x = self.window.left + int(self.window.width * 0.35)
        y = self.window.top + int(self.window.height * 0.58)
        self.clicker.scroll(x, y, amount if up else -amount)

    def _content_diff(self, first: Frame, second: Frame) -> float:
        """比较左侧题目内容两帧差异，用于判断是否已滚到顶部/底部。"""
        try:
            h, w = first.image.shape[:2]
            x1 = int(w * 0.70)
            y0, y1 = int(h * 0.06), int(h * 0.94)
            a = cv2.cvtColor(first.image[y0:y1, :x1], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(second.image[y0:y1, :x1], cv2.COLOR_BGR2GRAY)
            a = cv2.resize(a, (360, 220), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (360, 220), interpolation=cv2.INTER_AREA)
            return float(np.mean(cv2.absdiff(a, b)))
        except Exception:
            return 999.0

    def _exam_layout_signature(self, view: PageView) -> tuple:
        items = []
        for q in view.exam.questions:
            items.append(
                (
                    q.number,
                    int(q.start.y0 // 8),
                    len(q.options),
                    bool(q.complete),
                )
            )
        submit_y = int(view.exam.submit.cy // 8) if view.exam.submit else None
        return tuple(items), submit_y

    def _scroll_exam_to_top(self, max_rounds: int = 20) -> Frame:
        """反复向上滚动，直到连续两次画面不再变化，避免测验很长时遗漏上半部分。"""
        previous: Frame | None = None
        stable = 0
        for _ in range(max_rounds):
            self._scroll_exam(up=True, amount=8, quiet=True)
            self._sleep(0.25)
            frame = self._grab()
            if previous is not None:
                diff = self._content_diff(previous, frame)
                if diff < 0.7:
                    stable += 1
                else:
                    stable = 0
                if stable >= 2:
                    self._log("章节测验已滚动到顶部")
                    return frame
            previous = frame
        self._log("多次向上滚动后仍未确认顶部，从当前可见题目开始", "warning")
        return previous if previous is not None else self._grab()

    def _exam_score(self, view: PageView) -> str:
        parts = []
        for line in view.lines:
            text = line.text.strip()
            if "分" in text and len(text) <= 20:
                parts.append(text)
            if len(parts) >= 6:
                break
        return "；".join(parts)

    @staticmethod
    def _should_audit_outline(view: PageView) -> bool:
        rows = [row for row in view.sections if row.level >= 2]
        return bool(rows) and all(row.completed for row in rows)

    def _audit_outline_large(
        self, frame: Frame, view: PageView, wanted: tuple[int, ...] = (1, 2)
    ) -> bool:
        """右侧目录大幅上下滚动，寻找屏幕外漏掉的橙色 1/2 小节。"""
        now = time.time()
        if now - self._last_outline_audit < 8.0:
            return False
        if not self.settings.auto_next or not view.sections:
            return False
        self._last_outline_audit = now
        current_key = None
        if view.current_idx is not None and view.sections:
            current_key = view.sections[view.current_idx].key
        x = self.window.left + int(self.window.width * 0.88)
        y = self.window.top + int(self.window.height * 0.58)
        for direction in (-9, 9):  # 先大幅向下，再大幅向上
            last_sig = None
            stable = 0
            for _ in range(12):
                try:
                    self.clicker.scroll(x, y, direction)
                except Exception:
                    break
                self._sleep(0.35)
                frame = self._grab()
                view = self._scan(frame)
                rows = view.sections
                if not rows:
                    break
                candidates = [
                    row
                    for row in rows
                    if row.level >= 2
                    and not row.completed
                    and row.pending_tasks in wanted
                    and row.key != current_key
                ]
                if candidates:
                    row = candidates[0]
                    self._log(
                        f"右侧目录大幅滚动发现漏掉的小节：{row.display}"
                        f"（橙色 {row.pending_tasks}），点击进入",
                        "warning",
                    )
                    self._click_line(row.line, f"目录漏项-{row.display}")
                    self._sleep(1.5)
                    return True
                sig = tuple((r.key, r.pending_tasks, r.completed) for r in rows)
                if sig == last_sig:
                    stable += 1
                else:
                    stable = 0
                    last_sig = sig
                if stable >= 2:
                    break
        return False

    def _find_section_row(self, view: PageView, key: str):
        for row in view.sections:
            if row.key == key:
                return row
        return None

    def _backfill_pending_quizzes(
        self, frame: Frame, view: PageView
    ) -> tuple[Frame, PageView]:
        """优先补做“视频已看、章节测验未做”的小节（右侧目录橙色数字 1）。"""
        if self._backfill_done or not self.settings.auto_quiz:
            return frame, view
        self._backfill_done = True
        targets = [
            row for row in view.sections if row.pending_tasks == 1 and not row.completed
        ]
        if not targets:
            return frame, view
        self._backfill_active = True
        try:
            current_row = None
            if view.current_idx is not None and view.sections:
                current_row = view.sections[view.current_idx]
            resume_key: str | None = None
            if current_row is not None:
                if current_row.pending_tasks == 2 and not current_row.completed:
                    resume_key = current_row.key
                else:
                    start_idx = view.current_idx + 1 if view.current_idx is not None else 0
                    for row in view.sections[start_idx:]:
                        if row.pending_tasks == 2 and not row.completed and row.level >= 2:
                            resume_key = row.key
                            break
            if resume_key is None:
                for row in view.sections:
                    if row.pending_tasks == 2 and not row.completed and row.level >= 2:
                        resume_key = row.key
                        break

            target_keys = [row.key for row in targets]
            resume_display = resume_key or "（未找到，补测后停留）"
            self._log(
                f"优先补做章节测验：发现 {len(targets)} 个待补小节"
                f"（{', '.join(target_keys[:8])}{'…' if len(target_keys) > 8 else ''}）；"
                f"补完后返回 {resume_display}",
                "warning",
            )
            processed = 0
            for key in target_keys:
                self._check_stop()
                initial_target = next((r for r in targets if r.key == key), None)
                if initial_target is None:
                    continue
                self._status("exam", f"优先补测验：{initial_target.display}")
                frame = self._grab()
                view = self._scan(frame)
                row = self._find_section_row(view, key)
                if row is None:
                    self._log(f"目录中暂时看不到 {key}，跳过该小节", "warning")
                    continue
                already_here = (
                    view.current_idx is not None
                    and view.sections
                    and view.sections[view.current_idx].key == key
                    and (view.exam.quiz_tab is not None or view.exam.detected)
                )
                if not already_here:
                    self._log(f"切换到待补小节：{row.display}")
                    self._click_line(row.line, f"补测验-{row.display}")
                    self._sleep(3.2)
                    frame = self._grab()
                    view = self._scan(frame)
                    if not (view.exam.quiz_tab or view.exam.detected):
                        retry = self._find_section_row(view, key)
                        if retry is not None:
                            self._click_line(retry.line, f"重试切换-{retry.display}")
                            self._sleep(2.5)
                            frame = self._grab()
                            view = self._scan(frame)
                if not view.exam.quiz_tab and not view.exam.detected:
                    # 从测验结果页跳到目录小节时，页面可能保留在底部，先回到顶部找“章节测验”
                    self._scroll_exam(up=True, amount=24)
                    self._sleep(1.2)
                    frame = self._grab()
                    view = self._scan(frame)
                try:
                    if view.exam.detected:
                        self._handle_exam(view, frame)
                    elif view.exam.quiz_tab:
                        self._enter_chapter_quiz(view)
                    else:
                        self._log(f"小节 {key} 未找到章节测验入口，跳过", "warning")
                        continue
                    processed += 1
                    self._chapter_quiz_done = True
                    self._emit(
                        "stats", sections=self.sections_finished, quizzes=self.quizzes_solved
                    )
                except NavigationFailed:
                    raise

            # 返回进入补测前的“2”小节，继续看视频
            frame = self._grab()
            view = self._scan(frame)
            if resume_key:
                resume_row = self._find_section_row(view, resume_key)
                if resume_row is not None:
                    self._log(f"补测完成（{processed} 个），返回原小节：{resume_row.display}")
                    self._click_line(resume_row.line, f"返回原小节-{resume_row.display}")
                    self._sleep(3.2)
                    self._scroll_exam(up=True, amount=24)
                    self._sleep(1.2)
                    frame = self._grab()
                    view = self._scan(frame)
                    self._chapter_quiz_done = False
                else:
                    self._log(
                        f"补测完成，但目录中找不到原小节 {resume_key}，请手动切回",
                        "warning",
                    )
            self._status("scanning", f"补测完成（{processed} 个），返回视频")
            self._force_full = True
            return frame, view
        finally:
            self._backfill_active = False

    def _enter_chapter_quiz(self, view: PageView) -> None:
        """从视频页点击顶部“章节测验”，等待题目加载后完成测验。"""
        tab = view.exam.quiz_tab
        if not tab:
            return
        self._log(f"视频已完成，进入章节测验：{tab.text}", "warning")
        self._click_line(tab, "章节测验")
        self._sleep(3.0)
        frame = self._grab()
        view = self._scan(frame)
        if view.quiz_entry:
            self._click_line(view.quiz_entry, "开始答题")
            self._sleep(2.0)
            frame = self._grab()
            view = self._scan(frame)
        if not view.exam.detected:
            # 页面可能加载较慢，再等一次
            self._sleep(2.5)
            frame = self._grab()
            view = self._scan(frame)
        if not view.exam.detected:
            self._save_debug(frame, view, "exam_entry_failed")
            self._stop_reason = "点击章节测验后未识别到题目，请检查页面是否加载或手动完成"
            raise NavigationFailed(self._stop_reason)
        self._handle_exam(view, frame)
        self._chapter_quiz_done = True

    def _restart_exam_after_incomplete(
        self, view: PageView, frame: Frame, depth: int
    ) -> bool:
        self._exam_incomplete_rescan = False
        if depth >= 2:
            self._stop_reason = (
                "确认框连续提示还有未做题，重新扫描两次仍未补齐；请人工检查"
            )
            self._log(self._stop_reason, "error")
            raise NavigationFailed(self._stop_reason)
        self._log(
            f"重新扫描章节测验（第 {depth + 1} 次）：从顶部重新识别并补齐未做题",
            "warning",
        )
        self.exam_active = False
        try:
            return self._handle_exam(view, frame, _rescan_depth=depth + 1)
        finally:
            self.exam_active = True

    def _handle_exam(
        self, view: PageView, frame: Frame, _rescan_depth: int = 0
    ) -> bool:
        if self.exam_active:
            return True
        if not self.settings.auto_quiz:
            self._log("检测到章节测验，但自动答题已关闭；为避免跳过测验，自动化停止", "warning")
            self._stop_reason = "章节测验需要人工完成（自动答题已关闭）"
            raise StopAutomation()
        self.exam_active = True
        try:
            if view.exam.result:
                score = self._exam_score(view)
                if not self._chapter_quiz_done:
                    self._log(f"章节测验已提交完成{('：' + score) if score else ''}", "success")
                self._chapter_quiz_done = True
                self._status("exam", "章节测验已完成，准备下一节")
                return True
            self._status("exam", "准备章节测验…")
            self._save_debug(frame, view, "exam_start")
            self._exam_incomplete_rescan = False
            if view.exam.confirm:
                view = self._confirm_exam(view)
                if self._exam_incomplete_rescan:
                    return self._restart_exam_after_incomplete(
                        view, frame, _rescan_depth
                    )
                if view.exam.result:
                    self._log(f"章节测验已提交完成：{self._exam_score(view)}", "success")
                    self._chapter_quiz_done = True
                    return True

            # 回到测验顶部，拿到题量/标题，并确保从第 1 题开始顺序作答。
            # 不能只固定上滚 24 格：很长的测验可能滚不到顶，会漏掉上半部分题目。
            frame = self._scroll_exam_to_top()
            view = self._scan(frame)
            total = view.exam.total
            answered: dict[str, set[str]] = {}
            self._log(
                f"章节测验开始：识别到题量 {total if total is not None else '未知'}"
            )
            previous_sig: tuple | None = None
            previous_frame: Frame | None = None
            stable_scrolls = 0

            for step in range(90):
                self._check_stop()
                if view.exam.confirm:
                    view = self._confirm_exam(view)
                    if self._exam_incomplete_rescan:
                        return self._restart_exam_after_incomplete(
                            view, frame, _rescan_depth
                        )
                    if view.exam.result:
                        score = self._exam_score(view)
                        self._log(f"章节测验提交成功{('：' + score) if score else ''}", "success")
                        self._chapter_quiz_done = True
                        self._status("exam", "章节测验已完成")
                        return True
                    continue
                if view.exam.result:
                    score = self._exam_score(view)
                    if not self._chapter_quiz_done:
                        self._log(f"章节测验提交成功{('：' + score) if score else ''}", "success")
                    self._chapter_quiz_done = True
                    self._status("exam", "章节测验已完成")
                    return True
                if view.exam.alert:
                    self._log(f"提交提示：{view.exam.alert}", "warning")
                    ok = find_button(view.lines, ["确定", "确认"], exact=True)
                    if ok:
                        self._click_line(ok, "关闭提交提示")
                        self._sleep(1.2)
                        frame = self._grab()
                        view = self._scan(frame)
                    continue
                notice_text = " ".join(
                    l.text
                    for l in view.lines
                    if any(k in l.text for k in ("提交成功", "操作成功", "已提交"))
                )
                if notice_text:
                    ok = find_button(view.lines, ["确定", "关闭", "知道了"], exact=True)
                    if ok:
                        self._log(f"提交提示：{notice_text[:100]}，点击关闭")
                        self._click_line(ok, "关闭提交成功提示")
                        self._sleep(1.5)
                        frame = self._grab()
                        view = self._scan(frame)
                        continue
                total = view.exam.total or total
                visible = view.exam.questions
                solved_here = 0
                for q in visible:
                    if q.key in answered:
                        continue
                    if not q.complete:
                        continue
                    if len(q.options) < 2:
                        continue
                    if q.kind in ("填空题", "简答题"):
                        self._log(f"第 {q.number} 题是{q.kind}，无法自动作答", "error")
                        self._save_debug(frame, view, f"exam_manual_{q.number}")
                        self._stop_reason = "章节测验包含填空题/简答题，需人工完成"
                        raise NavigationFailed(self._stop_reason)
                    if not q.text:
                        continue
                    options = {o.letter: o.text for o in q.options}
                    self._status(
                        "exam",
                        f"章节测验：第 {q.number} 题（已答 {len(answered)}/{total or '?'}）",
                    )
                    self._log(
                        f"章节测验第 {q.number} 题【{q.kind}】：{q.text[:160]}"
                    )
                    self._log("选项：" + " | ".join(f"{k}.{v[:45]}" for k, v in options.items()))
                    try:
                        answer = self.llm.solve(q.text, options, kind=q.kind)
                    except LLMError as exc:
                        self._log(f"LLM 答题失败：{exc}", "error")
                        self._save_debug(frame, view, f"exam_llm_error_{q.number}")
                        self._stop_reason = f"章节测验 LLM 答题失败：{exc}"
                        raise NavigationFailed(self._stop_reason) from exc
                    letters = [x for x in answer.letters if x in options]
                    if not letters:
                        self._log(f"LLM 答案 {answer.letters} 不在选项 {list(options)} 中", "error")
                        self._stop_reason = "章节测验 LLM 答案与选项不匹配"
                        raise NavigationFailed(self._stop_reason)
                    self._select_exam_options(q, letters)
                    answered[q.key] = set(letters)
                    self.quizzes_solved += 1
                    solved_here += 1
                    reason = f"；{answer.reason}" if answer.reason else ""
                    self._log(
                        f"第 {q.number} 题选择 {','.join(letters)}（{answer.latency:.1f}s）{reason}",
                        "success",
                    )
                    self._emit(
                        "quiz_answer",
                        question=f"{q.number}【{q.kind}】{q.text}"[:500],
                        options=options,
                        answer=letters,
                        reason=answer.reason,
                        confidence=answer.confidence,
                    )
                    self._sleep(0.08)

                if solved_here:
                    # 选项选中会改变画面；重新取一帧作为“滚动前基准”，
                    # 避免把选中状态误判成页面仍在滚动。
                    frame = self._grab()

                # 判断是否真正到达页面底部：题目布局不再变化，且左侧内容两帧基本相同。
                # 不能因为“提交”按钮固定在底部可见就提前提交，否则长测验会漏题。
                sig = self._exam_layout_signature(view)
                page_diff = (
                    self._content_diff(previous_frame, frame)
                    if previous_frame is not None
                    else 999.0
                )
                if (
                    previous_sig is not None
                    and sig == previous_sig
                    and page_diff < 1.5
                    and solved_here == 0
                ):
                    stable_scrolls += 1
                    self._log(
                        f"页面滚动后内容未变化（{stable_scrolls}/2，差异 {page_diff:.2f}），接近底部"
                    )
                else:
                    stable_scrolls = 0

                incomplete_visible = [
                    q
                    for q in visible
                    if q.key not in answered
                    and (not q.complete or len(q.options) < 2)
                    and q.kind not in ("填空题", "简答题")
                ]
                if incomplete_visible and stable_scrolls >= 2:
                    self._log(
                        "页面底部仍有题目未完整显示，向上微调后继续识别："
                        + ", ".join(q.number for q in incomplete_visible[:6]),
                        "warning",
                    )
                    self._scroll_exam(up=True, amount=3)
                    self._sleep(1.0)
                    frame = self._grab()
                    view = self._scan(frame)
                    previous_sig = None
                    previous_frame = None
                    stable_scrolls = 0
                    continue

                submit = view.exam.submit
                if submit and stable_scrolls >= 2:
                    if total is not None and len(answered) < total:
                        missing_numbers = [
                            str(n)
                            for n in range(1, total + 1)
                            if str(n) not in answered
                        ]
                        self._log(
                            f"自检发现只答了 {len(answered)}/{total} 题，"
                            f"缺第 {', '.join(missing_numbers[:8])} 题",
                            "warning",
                        )
                        if _rescan_depth < 1:
                            self._log(
                                "提交前先从顶部重新扫描，尝试补齐漏题",
                                "warning",
                            )
                            return self._restart_exam_after_incomplete(
                                view, frame, _rescan_depth
                            )
                        self._log(
                            "已重扫一次仍缺题，先提交并检查平台提示",
                            "warning",
                        )
                    else:
                        self._log(
                            f"已到页面底部，共答 {len(answered)} 题，点击提交"
                        )
                    self._status("exam", "正在提交章节测验…")
                    frame, view = self._submit_exam(view, frame)
                    if self._exam_incomplete_rescan:
                        return self._restart_exam_after_incomplete(
                            view, frame, _rescan_depth
                        )
                    if view.exam.result:
                        score = self._exam_score(view)
                        self._log(f"章节测验提交成功{('：' + score) if score else ''}", "success")
                        self._chapter_quiz_done = True
                        self._status("exam", "章节测验已完成")
                        return True
                    if view.exam.confirm:
                        continue
                    if view.exam.alert:
                        self._log(f"提交提示：{view.exam.alert}", "warning")
                        ok = find_button(view.lines, ["确定", "确认"], exact=True)
                        if ok:
                            self._click_line(ok, "关闭提交提示")
                            self._sleep(1.5)
                            frame = self._grab()
                            view = self._scan(frame)
                        continue

                previous_sig = sig
                previous_frame = frame
                if step >= 89:
                    break
                # 滚动步长不能太大，否则长题目会被整屏跳过；3 格保证有重叠。
                self._scroll_exam(up=False, amount=3)
                self._sleep(0.55)
                frame = self._grab()
                view = self._scan(frame)

            self._save_debug(frame, view, "exam_failed")
            self._stop_reason = "章节测验未能完成：未找到全部题目或提交按钮"
            raise NavigationFailed(self._stop_reason)
        finally:
            self.exam_active = False
            self.time_box = None
            self.time_reading = None
            self.duration = 0.0
            self.last_current = None
            self.last_progress_wall = time.time()
            self.end_streak = 0
            self.resume_attempts = 0
            self.completion_handled = False
            self._force_full = True
            self._no_time_since = None
            self._emit("stats", sections=self.sections_finished, quizzes=self.quizzes_solved)

    def _select_exam_options(self, q: ExamQuestion, letters: list[str]) -> None:
        desired = set(letters)
        option_map = {o.letter: o for o in q.options}
        if q.kind == "多选题":
            for opt in q.options:
                want = opt.letter in desired
                if want != opt.selected:
                    self._click_exam_option(q, opt)
                    self._sleep(0.05)
            return
        for letter in letters:
            opt = option_map.get(letter)
            if not opt:
                continue
            if opt.selected:
                self._log(f"第 {q.number} 题选项 {letter} 已被选中，跳过点击")
                continue
            self._click_exam_option(q, opt)
            self._sleep(0.05)
            break

    def _click_exam_option(self, q: ExamQuestion, opt) -> None:
        scale = max(0.8, q.start.height / 28.0)
        x = self.window.left + q.start.x0 + int(27 * scale)
        y = self.window.top + opt.line.cy
        self._click_screen(x, y, f"章节测验第 {q.number} 题选项 {opt.letter}")

    def _confirm_exam(self, view: PageView) -> PageView:
        dialog_text = " ".join(line.text for line in view.lines)
        incomplete = any(
            key in dialog_text
            for key in ("未做完", "未作答", "还有题目", "还有未", "请完成所有题目")
        )
        if incomplete:
            cancel = view.exam.cancel or find_button(
                view.lines, ["取消"], exact=True
            )
            if not cancel:
                self._stop_reason = "确认框提示还有未做题，但未找到取消按钮"
                raise NavigationFailed(self._stop_reason)
            self._log(
                "确认框提示还有未做完的题目，取消提交并重新扫描整页",
                "warning",
            )
            self._click_line(cancel, "取消提交-还有未做题")
            self._exam_incomplete_rescan = True
            self._status("exam", "发现漏题：取消提交，重新扫描")
            self._sleep(0.6)
            frame = self._grab()
            return self._scan(frame)

        submit = view.exam.confirm_submit
        if not submit:
            self._stop_reason = "章节测验确认弹窗中未找到提交按钮"
            raise NavigationFailed(self._stop_reason)
        self._click_line(submit, "确认提交章节测验")
        self._status("exam", "已确认提交，等待结果…")
        self._sleep(1.2)
        frame = self._grab()
        return self._scan(frame)

    def _submit_exam(self, view: PageView, frame: Frame) -> tuple[Frame, PageView]:
        submit = view.exam.submit
        if not submit:
            return frame, view
        self._click_line(submit, "提交章节测验")
        self._sleep(0.8)
        frame = self._grab()
        view = self._scan(frame)
        if view.exam.confirm:
            view = self._confirm_exam(view)
            frame = self._grab()
        elif not view.exam.result:
            self._sleep(0.8)
            frame = self._grab()
            view = self._scan(frame)
        return frame, view

    # ------------------------------------------------------------------ 弹窗答题
    def _handle_quiz(self, view: QuizView | PageView, frame: Frame) -> None:
        if self.quiz_active:
            return
        quiz: QuizView = view.quiz if isinstance(view, PageView) else view
        if not self.settings.auto_quiz:
            self._log("检测到题目，但自动答题已关闭；请手动完成后重新启动", "warning")
            self._status("quiz", "检测到题目，等待人工处理")
            self._stop_reason = "检测到题目且自动答题已关闭"
            raise StopAutomation()
        self.quiz_active = True
        self._status("quiz", "检测到视频弹题，直接枚举答案…")
        self._save_debug(frame, self.view, "quiz")
        try:
            self._solve_popup_enumeration(quiz, frame)
        finally:
            self.quiz_active = False
            self.time_box = None
            self.time_reading = None
            self.duration = 0.0
            self.last_current = None
            self.last_progress_wall = time.time()
            self.end_streak = 0
            self.resume_attempts = 0
            self.completion_handled = False
            self._force_full = True
            self._no_time_since = None
            self._log(f"答题流程结束，已答 {self.quizzes_solved} 题")
            self._emit("stats", sections=self.sections_finished, quizzes=self.quizzes_solved)

    @staticmethod
    def _popup_enumeration_plan(quiz: QuizView) -> list[list[str]]:
        """生成完整的弹题枚举顺序；不调用 LLM。"""
        from itertools import combinations

        letters = sorted({o.letter for o in quiz.options})
        n = len(letters)
        if n == 0 or n > 10:
            return []
        kind = quiz.kind
        if kind in ("单选题", "判断题"):
            sizes = [1]
        elif kind in ("多选题", "不定项选择题"):
            # 先试与题干空格数一致的组合，减少尝试次数；但仍会遍历 1..N 全部大小。
            blanks = quiz.question.count("()") + quiz.question.count("（）")
            blanks += quiz.question.count("( )") + quiz.question.count("（ ）")
            preferred = blanks if 2 <= blanks <= n else 2
            sizes = [preferred] + [size for size in range(1, n + 1) if size != preferred]
        else:
            # 题型未知：按大小从 1 到 N 全部遍历。
            sizes = list(range(1, n + 1))
        plan: list[list[str]] = []
        for size in sizes:
            for combo in combinations(letters, size):
                plan.append(list(combo))
        return plan

    def _quick_popup_feedback(
        self, frame: Frame, allow_ocr: bool = True
    ) -> tuple[str, bool, bool]:
        """先按红/绿颜色判断对错（毫秒级），读不到再 OCR 反馈区。"""
        try:
            h, w = frame.image.shape[:2]
            hsv = cv2.cvtColor(
                frame.image[int(h * 0.75) : int(h * 0.92), int(w * 0.45) : int(w * 0.75)],
                cv2.COLOR_BGR2HSV,
            )
            red = cv2.inRange(hsv, (0, 120, 110), (8, 255, 255)) | cv2.inRange(
                hsv, (170, 120, 110), (180, 255, 255)
            )
            green = cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))
            red_n = int((red > 0).sum())
            green_n = int((green > 0).sum())
            if red_n > 100 and red_n > green_n * 1.5:
                return "回答错误", False, True
            if green_n > 100 and green_n > red_n * 1.5:
                return "回答正确", True, False
        except Exception:
            pass
        if not allow_ocr or not self.ocr:
            return "", False, False
        try:
            h, w = frame.image.shape[:2]
            crop = frame.image[int(h * 0.50) : int(h * 0.97), : int(w * 0.72)]
            lines = self.ocr.full(crop)
        except Exception:
            return "", False, False
        text = " ".join(line.text for line in lines)
        positive = any(k in text for k in ("回答正确", "答案正确", "答对了", "恭喜"))
        negative = any(k in text for k in ("回答错误", "答案错误", "答错了", "不正确"))
        return text, positive, negative

    def _popup_submit_present(self, frame: Frame, quiz: QuizView) -> bool:
        """判断弹窗的“提交”按钮是否还在；按钮消失通常说明弹窗已关闭/答对。"""
        if not quiz.submit or not self.ocr:
            return True
        try:
            line = quiz.submit
            x0 = max(0, line.x0 - 18)
            x1 = min(frame.image.shape[1], line.x1 + 18)
            y0 = max(0, line.y0 - 10)
            y1 = min(frame.image.shape[0], line.y1 + 10)
            if x1 <= x0 or y1 <= y0:
                return True
            text = self.ocr.line(frame.image[y0:y1, x0:x1]) or ""
            return "提交" in text or "交卷" in text
        except Exception:
            return True

    def _read_popup_feedback(self, frame: Frame) -> tuple[str, bool, bool]:
        """只识别弹窗底部一小块反馈区，避免视频画面颜色造成误判。"""
        if not self.ocr:
            return "", False, False
        try:
            h, w = frame.image.shape[:2]
            # 反馈文字固定在弹窗底部偏右；用识别-only 的小图，约 5ms。
            crop = frame.image[
                int(h * 0.805) : int(h * 0.865), int(w * 0.54) : int(w * 0.61)
            ]
            text = self.ocr.line(crop) or ""
            if not text:
                wide = frame.image[
                    int(h * 0.79) : int(h * 0.88), int(w * 0.50) : int(w * 0.70)
                ]
                text = self.ocr.line(wide) or ""
        except Exception:
            return "", False, False
        positive = any(k in text for k in ("回答正确", "答案正确", "答对了", "恭喜"))
        negative = any(k in text for k in ("回答错误", "答案错误", "答错了", "不正确"))
        return text, positive, negative

    def _popup_after_success(self, frame: Frame, old_signature: str) -> QuizView | None:
        """答对后寻找下一道弹题；没有题就返回 None，让视频继续。"""
        for _ in range(4):
            view = self._scan(frame)
            quiz = view.quiz
            if not quiz.detected:
                return None
            if quiz.options and quiz.signature != old_signature:
                return quiz
            button = quiz.next_button or find_button(
                view.lines,
                ["下一题", "继续答题", "继续", "关闭", "完成", "确定"],
                exact=False,
            )
            if button:
                self._click_line(button, f"弹题继续（{button.text}）")
                self._sleep(1.2)
                frame = self._grab()
                continue
            return None
        return None

    def _popup_selected_letters(self, frame: Frame, quiz: QuizView) -> set[str]:
        """通过选项左侧复选框的紫色字母判断当前已选中的项。"""
        selected: set[str] = set()
        try:
            hsv = cv2.cvtColor(frame.image, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]
            for opt in quiz.options:
                y = int(opt.line.cy)
                x1 = max(0, int(opt.line.x0) - 20)
                x0 = max(0, int(opt.line.x0) - 95)
                y0, y1 = max(0, y - 22), min(h, y + 22)
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = hsv[y0:y1, x0:x1]
                mask = cv2.inRange(crop, (90, 60, 60), (175, 255, 255))
                if int((mask > 0).sum()) >= 15:
                    selected.add(opt.letter)
                    continue
                # 有些播放器不是给复选框字母上色，而是把整个选项条染成紫色。
                # 再看选项文字区域的紫色填充比例。
                rx0 = max(0, int(opt.line.x0) - 25)
                rx1 = min(w, int(opt.line.x1) + 20)
                ry0, ry1 = max(0, y - max(16, opt.line.height // 2 + 6)), min(
                    h, y + max(16, opt.line.height // 2 + 6)
                )
                if rx1 > rx0 and ry1 > ry0:
                    row = hsv[ry0:ry1, rx0:rx1]
                    row_mask = cv2.inRange(row, (90, 55, 45), (175, 255, 255))
                    if float((row_mask > 0).mean()) > 0.12:
                        selected.add(opt.letter)
        except Exception:
            return set()
        return selected

    def _click_popup_option(self, opt) -> None:
        """点复选框本身，而不是文字中心；不同播放器只有方框可勾选。"""
        scale = max(0.9, opt.line.height / 28.0)
        x = self.window.left + max(8, int(opt.line.x0) - int(42 * scale))
        y = self.window.top + opt.line.cy
        self._click_screen(x, y, f"弹题选项 {opt.letter}")

    def _set_popup_combo(
        self, quiz: QuizView, combo: list[str], frame: Frame
    ) -> tuple[Frame, bool]:
        """把复选框精确设置成 combo，并复查；返回 (新画面, 是否确认成功)。"""
        desired = set(combo)
        for attempt in range(3):
            selected = self._popup_selected_letters(frame, quiz)
            if selected == desired:
                return frame, True
            if attempt > 0 and not selected and desired:
                self._log(
                    "无法从复选框颜色确认当前已选项，停止枚举，避免提交错组合",
                    "error",
                )
                return frame, False
            for opt in quiz.options:
                want = opt.letter in desired
                if want != (opt.letter in selected):
                    self._click_popup_option(opt)
                    self._sleep(0.025)
            self._sleep(0.08)
            frame = self._grab()
        selected = self._popup_selected_letters(frame, quiz)
        if selected != desired:
            self._log(
                f"选项状态校正失败：想要 {sorted(desired)}，实际 {sorted(selected)}",
                "error",
            )
            return frame, False
        return frame, True

    def _solve_popup_enumeration(self, quiz: QuizView, frame: Frame) -> None:
        """视频弹题不调用 LLM，直接按组合枚举，速度优先且不漏组合。"""
        current = quiz
        solved_questions = 0
        while (
            not self.stop_event.is_set()
            and solved_questions < max(1, self.settings.max_quiz_questions)
        ):
            self._check_stop()
            if current.is_result:
                self._log(f"题目结果：{current.feedback}")
                self._click_quiz_continue(current)
                self._sleep(1.2)
                frame = self._grab()
                current = self._scan(frame).quiz
                if not current.detected:
                    return
                continue
            if not current.options:
                self._log("弹题没有识别到可点击选项，停止自动作答", "error")
                self._stop_reason = "弹题选项识别失败，需人工处理"
                raise NavigationFailed(self._stop_reason)

            page_text = " ".join(
                line.text.replace(" ", "")
                for line in (self.view.lines if self.view else [])
            )
            if current.kind != "多选题" and (
                "多选题" in page_text or "不定项" in page_text
            ):
                current.kind = "多选题"
            blanks = current.question.count("()") + current.question.count("（）")
            blanks += current.question.count("( )") + current.question.count("（ ）")
            selected_now = self._popup_selected_letters(frame, current)
            if current.kind not in ("多选题",) and (
                blanks >= 2 or len(selected_now) >= 2
            ):
                current.kind = "多选题"
                self._log(
                    f"题干有 {blanks} 个空/已选中 {len(selected_now)} 项，按多选题完全遍历",
                    "warning",
                )

            option_letters = sorted({opt.letter for opt in current.options})
            expected_letters = [chr(ord("A") + i) for i in range(len(option_letters))]
            if option_letters != expected_letters:
                self._stop_reason = (
                    "弹题选项字母不连续（"
                    + ",".join(option_letters)
                    + "），可能漏识别选项；为避免假遍历，停止自动作答"
                )
                self._log(self._stop_reason, "error")
                raise NavigationFailed(self._stop_reason)
            plan = self._popup_enumeration_plan(current)
            if not plan:
                self._stop_reason = "弹题没有可枚举的答案组合"
                raise NavigationFailed(self._stop_reason)
            if current.kind in ("多选题", "不定项选择题", "选择题"):
                expected_count = (1 << len(option_letters)) - 1
                if len(plan) != expected_count:
                    self._stop_reason = (
                        f"枚举计划不完整：{len(plan)}/{expected_count}，停止以免漏组合"
                    )
                    self._log(self._stop_reason, "error")
                    raise NavigationFailed(self._stop_reason)
            self._log(
                f"视频弹题：完全遍历 {len(option_letters)} 个选项的全部 "
                f"{len(plan)} 种非空组合（{current.kind}）",
                "warning",
            )
            self._log(
                f"题目：{(current.question or '（题干未识别）')[:160].replace(chr(10), ' ')}"
            )

            solved = False
            next_quiz: QuizView | None = None
            old_signature = current.signature
            for index, combo in enumerate(plan, 1):
                self._check_stop()
                self._status(
                    "quiz",
                    f"直接枚举 {index}/{len(plan)}：{','.join(combo)}",
                )
                self._log(
                    f"枚举第 {index}/{len(plan)} 组：{','.join(combo)}",
                    "info",
                )
                if current.kind in ("单选题", "判断题"):
                    # 单选/判断是 radio，点击目标选项即可，不存在“多选取消”问题。
                    opt = current.option_map().get(combo[0])
                    if not opt:
                        self._stop_reason = f"找不到单选项 {combo[0]}"
                        raise NavigationFailed(self._stop_reason)
                    self._click_popup_option(opt)
                    self._sleep(0.03)
                else:
                    frame, state_ok = self._set_popup_combo(current, combo, frame)
                    if not state_ok:
                        self._stop_reason = (
                            f"无法精确设置选项组合 {','.join(combo)}，停止枚举"
                        )
                        self._log(self._stop_reason, "error")
                        raise NavigationFailed(self._stop_reason)
                submit = current.submit
                if not submit:
                    self._stop_reason = "弹题未找到提交按钮"
                    raise NavigationFailed(self._stop_reason)
                self._click_line(submit, f"提交组合 {','.join(combo)}")
                positive = False
                negative = False
                closed = False
                feedback = ""
                for _ in range(7):
                    self._sleep(0.12)
                    frame = self._grab()
                    feedback, positive, negative = self._read_popup_feedback(frame)
                    if positive:
                        break
                    if negative:
                        break
                    if not self._popup_submit_present(frame, current):
                        closed = True
                        break
                if positive:
                    self._log(f"组合 {','.join(combo)} 回答正确", "success")
                    solved = True
                    break
                if negative:
                    self._log(f"组合 {','.join(combo)} 回答错误，继续下一个", "warning")
                    continue
                if closed:
                    self._log(f"组合 {','.join(combo)} 提交后弹窗关闭，视为完成")
                    solved = True
                    break

                # 没读到反馈：可能是弹窗关闭/自动下一题，做一次完整确认
                view = self._scan(frame)
                found = view.quiz
                if not found.detected:
                    self._log(f"组合 {','.join(combo)} 提交后弹窗关闭，视为完成")
                    solved = True
                    break
                if found.options and found.signature != old_signature:
                    self._log("提交后出现下一道弹题")
                    next_quiz = found
                    solved = True
                    break
                self._log(f"组合 {','.join(combo)} 未读到对错提示，继续", "warning")

            if not solved:
                self._save_debug(frame, self.view, "quiz_enum_exhausted")
                self._stop_reason = (
                    f"弹题 {len(plan)} 种组合全部尝试完，仍没有答对；需人工处理"
                )
                self._log(self._stop_reason, "error")
                raise NavigationFailed(self._stop_reason)

            self.quizzes_solved += 1
            solved_questions += 1
            if next_quiz is None:
                next_quiz = self._popup_after_success(frame, old_signature)
            if next_quiz is None:
                self._log("弹题已全部完成", "success")
                return
            current = next_quiz

        if solved_questions >= max(1, self.settings.max_quiz_questions):
            self._stop_reason = "弹题数量超过上限"
            raise NavigationFailed(self._stop_reason)

    def _solve_quiz_loop(self, quiz: QuizView, frame: Frame) -> None:
        current_quiz = quiz
        answers_by_signature: dict[str, list[str]] = {}
        tried_answers: dict[str, set[tuple[str, ...]]] = {}
        forced_answers: dict[str, list[str]] = {}
        retry_left = max(0, int(self.settings.quiz_retry))
        question_count = 0
        while not self.stop_event.is_set() and question_count < max(1, self.settings.max_quiz_questions):
            self._check_stop()
            if current_quiz.is_result:
                self._log(f"题目结果：{current_quiz.feedback}")
                self._click_quiz_continue(current_quiz)
                frame = self._grab()
                current_quiz = self._scan(frame).quiz
                if not current_quiz.detected:
                    return
                continue
            if not current_quiz.options:
                self._log("题目没有识别到可点击选项（可能是填空题），已停止并保留截图", "error")
                self._stop_reason = "题目选项识别失败，需人工处理"
                raise NavigationFailed(self._stop_reason)

            if current_quiz.kind != "多选题":
                page_text = " ".join(
                    line.text.replace(" ", "")
                    for line in (self.view.lines if self.view else [])
                )
                if "多选题" in page_text or "不定项" in page_text:
                    current_quiz.kind = "多选题"
                    self._log("页面有“多选题”标记，按多选题处理", "info")

            question = current_quiz.question or "（题干未能完整识别，请仅根据选项判断）"
            options = {o.letter: o.text for o in current_quiz.options}
            feedback = self._feedback_text(self.view.lines if self.view else [])
            forced = forced_answers.pop(current_quiz.signature, None)
            if forced:
                letters = [x for x in forced if x in options]
                answer_reason = "上一轮回答错误，改为枚举提交"
                answer_confidence = 0.0
                answer_latency = 0.0
                self._log(
                    f"LLM 答案已判定错误，枚举尝试：{','.join(letters)}，重新提交",
                    "warning",
                )
            else:
                self._log(f"题目：{question[:160].replace(chr(10), ' ')}")
                self._log("选项：" + " | ".join(f"{k}.{v[:50]}" for k, v in options.items()))
                self._status("quiz", "正在请求 LLM 计算答案…")
                try:
                    answer = self.llm.solve(
                        question,
                        options,
                        kind=current_quiz.kind,
                        feedback=feedback if feedback else "",
                        previous_letters=answers_by_signature.get(current_quiz.signature),
                    )
                except LLMError as exc:
                    self._log(f"LLM 答题失败：{exc}", "error")
                    self._stop_reason = f"LLM 答题失败：{exc}"
                    raise NavigationFailed(self._stop_reason) from exc
                letters = [x for x in answer.letters if x in options]
                answer_reason = answer.reason
                answer_confidence = answer.confidence
                answer_latency = answer.latency
                if not letters:
                    self._log(f"LLM 返回的答案 {answer.letters} 不在识别到的选项中", "error")
                    self._stop_reason = "LLM 答案与页面选项不匹配"
                    raise NavigationFailed(self._stop_reason)
                if current_quiz.kind == "多选题" and len(letters) == 1:
                    self._log("警告：多选题只返回了一个选项", "warning")
                reason = f"；理由：{answer.reason}" if answer.reason else ""
                self._log(
                    f"LLM 选择：{','.join(letters)}（置信度 {answer.confidence:.2f}，{answer.latency:.1f}s）{reason}",
                    "success",
                )
            tried_answers.setdefault(current_quiz.signature, set()).add(tuple(sorted(letters)))
            self._emit(
                "quiz_answer",
                question=question[:500],
                options=options,
                answer=letters,
                reason=answer_reason,
                confidence=answer_confidence,
            )
            answers_by_signature[current_quiz.signature] = list(letters)
            self._status("quiz", f"选择 {','.join(letters)} 并提交")

            for letter in letters:
                opt = current_quiz.option_map().get(letter)
                if opt:
                    self._click_line(opt.line, f"选项 {letter}")
                    self._sleep(0.35)
            submit = current_quiz.submit
            if submit:
                self._click_line(submit, f"提交答案（{submit.text}）")
            else:
                self._log("未识别到提交按钮，可能点击选项后自动提交")

            self._sleep(2.4)
            frame = self._grab()
            new_view = self._scan(frame)
            new_quiz = new_view.quiz
            new_feedback = self._feedback_text(new_view.lines)
            positive, negative = self._feedback_polarity(new_feedback)

            # 二次确认弹窗
            if new_quiz.detected and new_quiz.options and not new_feedback:
                confirm = find_button(
                    new_view.lines, ["确认提交", "确认", "确定"], exact=True
                )
                if confirm and new_quiz.signature == current_quiz.signature:
                    self._click_line(confirm, "确认提交")
                    self._sleep(2.0)
                    frame = self._grab()
                    new_view = self._scan(frame)
                    new_quiz = new_view.quiz
                    new_feedback = self._feedback_text(new_view.lines)
                    positive, negative = self._feedback_polarity(new_feedback)

            if positive:
                self._log("答案正确", "success")
                self.quizzes_solved += 1
                question_count += 1
                if new_quiz.detected and new_quiz.options and new_quiz.signature != current_quiz.signature:
                    # 平台已自动切到下一题，不要再去点“下一题”
                    current_quiz = new_quiz
                    continue
                self._click_quiz_continue(new_quiz if new_quiz.detected else None)
                self._sleep(1.5)
                frame = self._grab()
                new_view = self._scan(frame)
                new_quiz = new_view.quiz
                if not new_quiz.detected:
                    return
                if new_quiz.signature == current_quiz.signature:
                    self._click_quiz_continue(new_quiz)
                    self._sleep(1.2)
                    frame = self._grab()
                    new_quiz = self._scan(frame).quiz
                    if not new_quiz.detected:
                        return
                current_quiz = new_quiz
                continue

            if negative:
                self._log("LLM 答案错误", "error")
                if new_feedback:
                    self._log(f"页面反馈：{new_feedback[:300]}", "warning")
                retry_btn = find_button(new_view.lines, ["重新答题", "再试一次", "重做"], exact=False)
                if retry_btn and retry_left > 0:
                    retry_left -= 1
                    self._log("点击重新答题，再次请求 LLM", "warning")
                    self._click_line(retry_btn, "重新答题")
                    self._sleep(2.0)
                    frame = self._grab()
                    new_view = self._scan(frame)
                    current_quiz = new_view.quiz
                    if not current_quiz.detected or not current_quiz.options:
                        return
                    continue
                if new_quiz.detected and new_quiz.options and new_quiz.signature != current_quiz.signature:
                    current_quiz = new_quiz
                    continue
                # 视频弹题答错后仍保留选项和“提交”：枚举其他答案重新提交，
                # 直到答对/弹窗关闭/所有组合枚举完。整页章节测验不会走这里。
                if new_quiz.detected and new_quiz.options and new_quiz.submit:
                    tried = tried_answers.setdefault(current_quiz.signature, set())
                    candidates = self._popup_candidate_answers(current_quiz, tried)
                    if candidates:
                        nxt = candidates[0]
                        forced_answers[current_quiz.signature] = list(nxt)
                        self._log(
                            f"页面提示回答错误，自动枚举下一组答案：{','.join(nxt)}",
                            "warning",
                        )
                        current_quiz = new_quiz
                        continue
                    self._save_debug(frame, new_view, "quiz_enum_exhausted")
                    self._stop_reason = "视频弹题所有可选答案组合均已尝试，仍提示回答错误"
                    self._log(self._stop_reason, "error")
                    raise NavigationFailed(self._stop_reason)
                self._click_quiz_continue(new_quiz if new_quiz.detected else None)
                self._sleep(2.0)
                frame = self._grab()
                new_view = self._scan(frame)
                new_quiz = new_view.quiz
                if not new_quiz.detected:
                    return
                current_quiz = new_quiz
                continue

            # 没有明确对错提示
            if not new_quiz.detected:
                self._log("题目弹窗已消失，视为提交完成", "success")
                self.quizzes_solved += 1
                return
            if new_quiz.signature != current_quiz.signature:
                self._log("出现下一题")
                self.quizzes_solved += 1
                question_count += 1
                current_quiz = new_quiz
                continue
            self._log("提交后题目仍在，未检测到对错提示；避免重复提交，停止自动答题", "error")
            self._save_debug(frame, new_view, "quiz_stuck")
            self._stop_reason = "答题提交后页面无变化，需人工检查"
            raise NavigationFailed(self._stop_reason)
        if question_count >= max(1, self.settings.max_quiz_questions):
            self._stop_reason = "答题数量超过上限"
            raise NavigationFailed(self._stop_reason)

    @staticmethod
    def _popup_candidate_answers(
        quiz: QuizView, tried: set[tuple[str, ...]]
    ) -> list[list[str]]:
        """枚举弹题还没提交过的答案组合。"""
        letters = [o.letter for o in quiz.options]
        candidates: list[list[str]] = []
        if quiz.kind == "多选题":
            from itertools import combinations

            for size in range(1, min(len(letters), 4) + 1):
                for combo in combinations(letters, size):
                    key = tuple(combo)
                    if key not in tried:
                        candidates.append(list(combo))
                    if len(candidates) >= 20:
                        return candidates
            return candidates
        for letter in letters:
            key = (letter,)
            if key not in tried:
                candidates.append([letter])
        if len(letters) >= 3 and all((letter,) in tried for letter in letters):
            # 题型识别成“选择题”但页面实际是多选时的兜底：
            # 单选项都试过仍错误，再试两两/三三组合。
            from itertools import combinations

            for size in (2, 3):
                if size > len(letters):
                    break
                for combo in combinations(letters, size):
                    key = tuple(combo)
                    if key not in tried:
                        candidates.append(list(combo))
                    if len(candidates) >= 20:
                        return candidates
        return candidates

    def _click_quiz_continue(self, quiz: QuizView | None) -> None:
        if not quiz:
            return
        line = quiz.next_button
        if not line:
            line = find_button(
                self.view.lines if self.view else [],
                ["下一题", "继续答题", "继续", "关闭", "完成", "确定"],
                exact=False,
            )
        if line:
            self._click_line(line, f"继续（{line.text}）")

    @staticmethod
    def _feedback_text(lines: list[OcrLine]) -> str:
        texts = [l.text for l in lines if l.text.strip()]
        hits = [
            t
            for t in texts
            if any(
                k in t
                for k in (
                    "回答正确",
                    "答案正确",
                    "答对了",
                    "恭喜",
                    "回答错误",
                    "答案错误",
                    "答错了",
                    "不正确",
                    "正确答案",
                    "你的答案",
                    "答案解析",
                )
            )
        ]
        return " ".join(hits[:8])

    @staticmethod
    def _feedback_polarity(text: str) -> tuple[bool, bool]:
        if any(k in text for k in ("回答错误", "答案错误", "答错了", "不正确")):
            return False, True
        if any(k in text for k in ("回答正确", "答案正确", "答对了", "恭喜")):
            return True, False
        return False, False

    # ------------------------------------------------------------------ 切下一节
    def _note_page_videos(self, view: PageView) -> None:
        """记录当前小节页面有几个视频（任务点），用于同页第二个视频。"""
        key = None
        row = None
        if view.current_idx is not None and view.sections:
            row = view.sections[view.current_idx]
            key = row.key
        if key is None:
            key = view.page_title or self.current_key
        if not key or key == self._page_video_key:
            return
        self._page_video_key = key
        self._page_video_index = 0
        labels = sum(
            1 for line in view.lines if compact_text(line.text) == "任务点"
        )
        pending = row.pending_tasks if row else 0
        self._page_video_total = max(labels, pending or 0)
        if self._page_video_total >= 2:
            self._log(
                f"当前小节识别到 {self._page_video_total} 个视频任务点，"
                "播完第一个会向下滚动找第二个",
                "warning",
            )

    def _try_next_video_on_page(self, frame: Frame) -> bool:
        """视频播完后向下滚动；只要发现“另一个未结束的视频”就继续播。"""
        x = self.window.left + int(self.window.width * 0.42)
        y = self.window.top + int(self.window.height * 0.58)
        for _ in range(4):
            try:
                self.clicker.scroll(x, y, -8)
            except Exception:
                break
            self._sleep(1.0)
            frame = self._grab()
            view = self._scan(frame)
            if not view.time:
                # 同页第二个视频常是黑色画面 + 中央播放钮，时间还没加载。
                labels = sum(
                    1 for line in view.lines if compact_text(line.text) == "任务点"
                )
                if labels >= 2:
                    px, py, found = self._find_center_play_button(frame)
                    if found:
                        self._click_screen(px, py, "同页下一个视频播放按钮")
                        self._sleep(2.0)
                        frame = self._grab()
                        view = self._scan(frame)
            if not view.time or view.time.duration <= 0:
                continue
            # 第一个视频刚播完时仍可能留在画面上，时间接近结尾，必须排除。
            if view.time.current >= view.time.duration - 5:
                continue
            self._adopt_time(view.time)
            self.completion_handled = False
            self.end_streak = 0
            self.last_progress_wall = time.time()
            self._log(
                f"同页发现下一个视频：{view.time.current_text()} / "
                f"{view.time.duration_text()}，继续播放",
                "warning",
            )
            self._status("playing", "继续播放同页下一个视频")
            return True
        self._log("已向下滚动，没有发现同页下一个未结束的视频", "warning")
        return False

    def _advance(self, reason: str) -> bool:
        if self.navigating:
            return True
        self.navigating = True
        try:
            if not self.settings.auto_next:
                self._log("视频已结束；自动下一节已关闭", "warning")
                self._status("completed", "视频已结束；自动下一节已关闭")
                self._stop_reason = "视频已结束（自动下一节已关闭）"
                return False
            self._log(f"准备进入下一节（{reason}）")
            self._status("navigating", "正在查找下一节…")
            frame = self._grab()
            view = self._scan(frame)
            if view.quiz.detected and not view.exam.detected:
                # 视频上的单题弹窗：直接作答，绝不进入会滚动的整页测验流程
                self._log("检测到视频弹题，直接作答（不滚动）")
                self._handle_quiz(view, frame)
                self.completion_handled = False
                return True
            if view.exam.detected:
                self._log("检测到章节测验页面，开始自动答题")
                self._handle_exam(view, frame)
                frame = self._grab()
                view = self._scan(frame)
            elif view.exam.quiz_tab:
                if not self.settings.auto_quiz:
                    self._stop_reason = "检测到章节测验，但自动答题已关闭；为避免跳过测验而停止"
                    raise NavigationFailed(self._stop_reason)
                current_completed = False
                if view.current_idx is not None and view.sections:
                    current_completed = view.sections[view.current_idx].completed
                elif self.current_key:
                    for row in view.sections:
                        if row.key == self.current_key:
                            current_completed = row.completed
                            break
                if current_completed:
                    self._log(
                        "当前小节右侧目录已打勾（含章节测验），跳过重复测验，直接进入下一节",
                        "success",
                    )
                    self._chapter_quiz_done = True
                else:
                    self._enter_chapter_quiz(view)
                    frame = self._grab()
                    view = self._scan(frame)
            if view.quiz.detected:
                self._handle_quiz(view, frame)
                self.completion_handled = False
                return True
            if view.exam.result and not view.next_button:
                self._scroll_exam(up=False, amount=6)
                self._sleep(1.0)
                frame = self._grab()
                view = self._scan(frame)
            before_title = view.page_title
            before_key = self.sections[view.current_idx].key if view.current_idx is not None and view.sections else self.current_key
            before_current = view.time.current if view.time else None
            before_duration = self.duration

            target_button = view.next_button
            target_row = None
            target_number: str | None = None
            idx = view.current_idx
            if idx is None and before_key:
                for i, row in enumerate(view.sections):
                    if row.key == before_key:
                        idx = i
                        break
            if idx is not None:
                nxt = pick_next_section(view.sections, idx)
                if nxt is not None:
                    target_row = view.sections[nxt]
                    target_number = target_row.number
            if not target_button and not target_row:
                if self._audit_outline_large(frame, view):
                    self.completion_handled = False
                    return True
                self._status("finished", "已经是最后一节")
                self._log("没有找到下一节，可能已到课程最后一节", "success")
                self._stop_reason = "已到最后一节"
                return False

            attempts = max(1, int(self.settings.max_nav_attempts))
            for attempt in range(1, attempts + 1):
                self._check_stop()
                if attempt == 1 and target_button:
                    self._click_line(target_button, f"下一节按钮（{target_button.text}）")
                elif attempt == 1 and target_row:
                    self._click_line(target_row.line, f"下一节：{target_row.display}")
                elif target_row and target_button:
                    # 第一次点按钮/目录失败时换另一个入口
                    if attempt % 2 == 0:
                        self._click_line(target_row.line, f"重试目录：{target_row.display}")
                    else:
                        self._click_line(target_button, "重试下一节按钮")
                elif target_row:
                    self._click_line(target_row.line, f"重试：{target_row.display}")
                else:
                    self._click_line(target_button, "重试下一节按钮")
                self._status("navigating", f"已点击下一节，等待页面加载（{attempt}/{attempts}）…")
                # 页面切换通常 0.5~1.5 秒；先短等再确认，避免每次白等 5 秒。
                self._sleep(max(0.9, min(1.5, self.settings.nav_wait_seconds)))
                new_frame = self._grab()
                new_view = self._scan(new_frame)
                if new_view.quiz.detected:
                    self._handle_quiz(new_view, new_frame)
                    self.completion_handled = False
                    return True
                new_key = self.current_key
                if new_view.current_idx is not None and new_view.sections:
                    new_key = new_view.sections[new_view.current_idx].key
                changed = False
                detail = ""
                if target_number and new_key == target_number:
                    changed, detail = True, f"目录高亮已到 {new_key}"
                elif self._looks_like_intro_page(new_view):
                    # 点击“下一节”后先到学习目标页，也是有效切换；
                    # 后续主循环会自动点学习目标页的“下一节”。
                    changed, detail = True, "进入学习目标页"
                elif new_view.page_title and (
                    not before_title or new_view.page_title != before_title
                ):
                    changed, detail = True, f"页面标题变化：{new_view.page_title}"
                elif new_view.time and before_current is not None:
                    if before_duration and abs(self.duration - before_duration) > 2:
                        changed, detail = True, "视频时长变化"
                    elif new_view.time.current < max(8.0, before_current - 8.0):
                        changed, detail = True, "视频进度重置"
                elif new_view.time and before_current is None:
                    changed, detail = True, "进入新的视频页面"
                elif new_view.current_idx is not None and idx is not None and new_view.current_idx != idx:
                    changed, detail = True, "目录当前项变化"
                if new_view.quiz_entry:
                    self._log("新页面需要先开始答题，点击进入")
                    self._click_line(new_view.quiz_entry, "开始答题")
                    self._sleep(2.0)
                    changed = True
                    detail = "进入测验"
                if changed:
                    self._log(f"已进入下一节（{detail}）", "success")
                    self.sections_finished += 1
                    self.current_key = new_key or target_number
                    self.current_idx = new_view.current_idx if new_view.current_idx is not None else idx + 1 if idx is not None else None
                    self.completion_handled = False
                    self.time_box = new_view.time.line if new_view.time else None
                    self.time_reading = new_view.time
                    self.duration = new_view.time.duration if new_view.time else 0.0
                    self.last_current = new_view.time.current if new_view.time else None
                    self.last_progress_wall = time.time()
                    self.end_streak = 0
                    self.completion_handled = False
                    self.fast_fail = 0
                    self.last_full_scan = 0
                    self._force_full = True
                    self._no_time_since = None
                    self._chapter_quiz_done = False
                    self._emit("stats", sections=self.sections_finished, quizzes=self.quizzes_solved)
                    self._status("playing", "已进入下一节")
                    if new_view.time:
                        if self._is_near_end(new_view.time):
                            # 新视频还没加载出时间时不要误判；这里只处理明确的已结束
                            self._sleep(2.0)
                        else:
                            if new_view.task_completed is False:
                                self._log(
                                    "新小节任务点为黄色，自动开始/继续播放",
                                    "warning",
                                )
                            self._ensure_playing()
                    elif self._video_needs_play(new_view):
                        # 新小节是黄色任务点，且播放器时间还没出现：尝试点击中央开始
                        self._try_start_pending_video()
                    return True
                self._log(f"第 {attempt} 次切换后页面似乎未变化，重试…", "warning")
            # “下一节”按钮点不动时，大幅滚动右侧目录找漏掉的橙色小节。
            if self._audit_outline_large(new_frame, new_view):
                self.completion_handled = False
                return True
            self._save_debug(self._grab(), self.view, "nav_failed")
            self._stop_reason = "连续切换下一节失败，请检查右侧目录是否已展开或手动点击。"
            raise NavigationFailed(self._stop_reason)
        finally:
            self.navigating = False

    # ------------------------------------------------------------------ 点击
    def _click_line(self, line: OcrLine, desc: str) -> None:
        x = self.window.left + int((line.x0 + line.x1) / 2)
        y = self.window.top + line.cy
        self._click_screen(x, y, desc)

    def _click_screen(self, x: int, y: int, desc: str) -> None:
        self._check_stop()
        if not self.window.contains(x, y):
            self._log(f"点击坐标超出目标窗口，已忽略：{desc} ({x},{y})", "warning")
            return
        if self.settings.dry_run:
            self._log(f"[试运行] 将点击：{desc} @ ({x}, {y})", "dry")
            return
        if not self._ensure_foreground():
            return
        try:
            self.clicker.click(x, y)
        except MouseInputError as exc:
            self._log(f"鼠标操作失败：{exc}", "error")
            self._stop_reason = (
                "鼠标未能成功操作。可能被系统/安全软件拦截，或程序与浏览器权限级别不同；"
                "请尝试以相同权限运行，并查看日志。"
            )
            raise NavigationFailed(self._stop_reason) from exc
        except Exception as exc:
            self._log(f"鼠标点击异常：{exc}", "error")
            self._stop_reason = f"鼠标点击异常：{exc}"
            raise NavigationFailed(self._stop_reason) from exc
        method = self.clicker.last_method
        before = getattr(self.clicker, "last_before", (-1, -1))
        after = getattr(self.clicker, "last_after", (-1, -1))
        self._log(f"点击：{desc} @ ({x}, {y})；鼠标 {before}->{after}，方式 {method}")
        if self.clicker.last_error:
            self._log(f"鼠标输入告警：{self.clicker.last_error}", "warning")

    def _ensure_foreground(self, timeout: float = 180.0) -> bool:
        if self.settings.dry_run:
            return True
        start = time.time()
        warned = False
        while time.time() - start < timeout:
            if self.stop_event.is_set():
                return False
            if is_foreground(self.window.hwnd):
                if warned:
                    self._log("目标窗口已回到前台，继续", "success")
                return True
            if not warned:
                self._log("目标窗口不在前台，已暂停点击；切回浏览器后自动继续", "warning")
                self._status("waiting_focus", "等待浏览器回到前台")
                warned = True
            time.sleep(0.4)
        self._log("目标窗口长时间不在前台，停止自动化", "error")
        self.stop_event.set()
        return False
