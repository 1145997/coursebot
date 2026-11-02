"""前台等待不能丢任务，恢复后不能提交等待前的坐标或控件。"""
import queue
import threading
import unittest
from unittest.mock import Mock, patch

from coursebot.automation import AutomationWorker, ForegroundRecheck, NavigationFailed
from coursebot.config import Settings
from coursebot.course_state import CourseSnapshot
from coursebot.windows import WindowInfo


class ForegroundWaitTests(unittest.TestCase):
    def setUp(self):
        self.w = AutomationWorker(Settings(preview_enabled=False),
            WindowInfo(1, "course", "test", 0, 0, 1400, 900), queue.Queue(), threading.Event())
        self.w._log = Mock()
        self.w._status = Mock()
        self.w._refresh_geometry = Mock()
        self.w.clicker = Mock()
        self.now = 0.0
        for name, mock in [
            ("window_exists", Mock(return_value=True)),
            ("foreground_description", Mock(return_value="助手主界面")),
            ("time.monotonic", Mock(side_effect=lambda: self.now)),
        ]:
            p = patch("coursebot.automation." + name, mock)
            p.start()
            self.addCleanup(p.stop)

    def wait(self, seconds):
        self.now += 100
        return False

    def test_wait_beyond_old_three_minute_limit_then_resume_without_stopping(self):
        self.w.stop_event.wait = Mock(side_effect=self.wait)
        self.w.sections_finished, self.w.quizzes_solved = 1, 4
        self.w.time_box = object()
        with patch("coursebot.automation.is_foreground", side_effect=[False] * 5 + [True]):
            self.assertTrue(self.w._ensure_foreground())
        self.assertEqual(self.now, 500)
        self.assertFalse(self.w.stop_event.is_set())
        self.assertEqual((self.w.sections_finished, self.w.quizzes_solved), (1, 4))
        self.assertIsNone(self.w.time_box)
        self.assertEqual(self.w._focus_generation, 1)
        self.w._refresh_geometry.assert_called_once()
        self.w.clicker.click.assert_not_called()

    def test_stop_interrupts_wait_immediately_without_input(self):
        self.w.stop_event.wait = Mock(side_effect=lambda _seconds: self.w.stop_event.set())
        with patch("coursebot.automation.is_foreground", return_value=False):
            self.assertFalse(self.w._ensure_foreground())
        self.w.stop_event.wait.assert_called_once_with(.4)
        self.w.clicker.click.assert_not_called()
        self.w._refresh_geometry.assert_not_called()

    def test_closed_target_reports_actual_reason_without_claiming_user_stop(self):
        with patch("coursebot.automation.window_exists", return_value=False):
            with self.assertRaisesRegex(NavigationFailed, "浏览器窗口已关闭"):
                self.w._ensure_foreground()
        self.assertFalse(self.w.stop_event.is_set())

    def test_foreground_target_does_not_reset_tracking(self):
        self.w._force_full = False
        with patch("coursebot.automation.is_foreground", return_value=True):
            self.assertTrue(self.w._ensure_foreground())
        self.assertEqual(self.w._focus_generation, 0)
        self.assertFalse(self.w._force_full)

    def resumed(self):
        self.w._focus_generation += 1
        return True

    def test_coordinate_computed_before_wait_is_never_clicked(self):
        self.w._ensure_foreground = Mock(side_effect=self.resumed)
        with self.assertRaises(ForegroundRecheck):
            self.w._click_screen(200, 300, "stale answer")
        self.w.clicker.click.assert_not_called()

    def test_native_token_from_before_wait_is_never_invoked(self):
        self.w._browser_reader = Mock()
        self.w._ensure_foreground = Mock(side_effect=self.resumed)
        with self.assertRaises(ForegroundRecheck):
            self.w._activate_accessible("stale token", "video")
        self.w._browser_reader.activate.assert_not_called()

    def test_recheck_restarts_main_loop_without_finishing_session(self):
        self.w._main_loop_run = Mock(side_effect=[ForegroundRecheck(), None])
        self.w._main_loop()
        self.assertEqual(self.w._main_loop_run.call_count, 2)
        self.assertFalse(self.w.stop_event.is_set())

    def test_supplied_inventory_is_discarded_after_foreground_wait(self):
        self.w._browser_reader = Mock()
        self.w._browser_reader.read.return_value = CourseSnapshot("new", complete=True)
        self.w._ensure_foreground = Mock(side_effect=self.resumed)
        self.w.settings.save_debug_frames = False
        self.assertTrue(self.w._accessible_video_tick(CourseSnapshot("old", complete=True)))
        self.w._browser_reader.read.assert_called_once()
        self.assertEqual(self.w._body_signature[0], "new")
