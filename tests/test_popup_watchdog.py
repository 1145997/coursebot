"""视频弹层未暴露到 UIA 时，仍须通过 OCR 进入原弹题流程。"""
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from coursebot.automation import AutomationWorker, NavigationFailed, StopAutomation
from coursebot.capture import Frame
from coursebot.config import Settings
from coursebot.course_state import CourseSnapshot, CourseVideo
from coursebot.ocr import OcrLine
from coursebot.vision import PageView, analyze_page
from coursebot.windows import WindowInfo


def popup_fixture():
    # 来自用户截图的 OCR 几何；不保存截图、登录参数或网页原文。
    lines = [
        OcrLine("判断题", .99, 176, 371, 264, 403),
        OcrLine("国防动员的行为主体是国家。", .99, 246, 439, 543, 462),
        OcrLine("A、对", .99, 298, 528, 369, 558),
        OcrLine("B、错", .99, 297, 619, 370, 651),
        OcrLine("提交", .99, 1507, 1101, 1558, 1131),
    ]
    pixels = np.zeros((1364, 2442, 3), np.uint8)
    return Frame(pixels, 0, 0, 0), analyze_page(pixels, lines)


class PopupWatchdogTests(unittest.TestCase):
    def setUp(self):
        self.frame, self.view = popup_fixture()
        self.worker = AutomationWorker(Settings(save_debug_frames=False),
            WindowInfo(1, "test", "test", 0, 0, 2442, 1364), queue.Queue(), threading.Event())
        w = self.worker
        w._browser_reader = Mock()
        w._grab = Mock(return_value=self.frame)
        w._scan = Mock(return_value=self.view)
        w._handle_quiz = Mock()
        w._click_screen = Mock()
        w._ensure_foreground = Mock(return_value=True)
        w._status = Mock()
        w._sleep = Mock()
        w._body_signature = ("course:chapter", "video", 1, False, False)
        self.snapshot = CourseSnapshot("course:chapter",
            [CourseVideo("video1", 1, False, 1299, 1606, False, "pause")], page_kind="video")

    def test_screenshot_shape_is_a_complete_popup_not_a_chapter_exam(self):
        self.assertTrue(self.view.quiz.detected)
        self.assertFalse(self.view.exam.detected)
        self.assertEqual(self.view.quiz.kind, "判断题")
        self.assertEqual([option.letter for option in self.view.quiz.options], ["A", "B"])
        self.assertEqual(self.view.quiz.submit.text, "提交")

    def test_uia_popup_false_and_playing_state_still_run_the_popup_handler_first(self):
        with patch("coursebot.automation.time.monotonic", return_value=100):
            self.assertTrue(self.worker._accessible_video_tick(self.snapshot))
        self.worker._handle_quiz.assert_called_once_with(self.view, self.frame)
        self.worker._browser_reader.activate.assert_not_called()
        self.worker._click_screen.assert_not_called()

    def test_normal_playback_limits_full_ocr_to_one_check_per_five_seconds(self):
        self.worker._scan.return_value = PageView(2442, 1364, [])
        with patch("coursebot.automation.time.monotonic", return_value=100):
            for now in (100, 101, 104):
                self.assertFalse(self.worker._probe_accessible_popup(self.snapshot, now))
        self.assertEqual(self.worker._scan.call_count, 1)
        self.assertFalse(self.worker._probe_accessible_popup(self.snapshot, 105))
        self.assertEqual(self.worker._scan.call_count, 2)

    def test_ocr_without_a_popup_refreshes_uia_before_a_play_decision(self):
        self.worker._scan.return_value = PageView(2442, 1364, [])
        paused = CourseSnapshot("course:chapter",
            [CourseVideo("video1", 1, False, 0, 1606, True, "play")], page_kind="video")
        self.worker._browser_reader.read.return_value = self.snapshot  # 扫描期间已恢复播放。
        with patch("coursebot.automation.time.monotonic", return_value=100):
            self.worker._accessible_video_tick(paused)
        self.worker._browser_reader.read.assert_called_once()
        self.worker._browser_reader.activate.assert_not_called()

    def test_uia_hint_waits_for_ocr_without_clicking_play(self):
        self.snapshot.popup_quiz = True
        self.worker._scan.return_value = PageView(2442, 1364, [])
        self.assertTrue(self.worker._probe_accessible_popup(self.snapshot, 100))
        with self.assertRaises(NavigationFailed):
            self.worker._probe_accessible_popup(self.snapshot, 121)
        self.worker._browser_reader.activate.assert_not_called()

    def test_nonvideo_page_keeps_the_existing_chapter_quiz_flow(self):
        snapshot = CourseSnapshot("course:chapter@1", page_kind="quiz")
        self.assertFalse(self.worker._probe_accessible_popup(snapshot, 100))
        self.worker._scan.assert_not_called()

    def test_disabled_auto_answer_stops_before_any_popup_input(self):
        self.worker.settings.auto_quiz = False
        self.worker._handle_quiz = AutomationWorker._handle_quiz.__get__(self.worker)
        with self.assertRaises(StopAutomation):
            self.worker._probe_accessible_popup(self.snapshot, 100)
        self.worker._click_screen.assert_not_called()
        self.worker._browser_reader.activate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
