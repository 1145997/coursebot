"""多播放器回归测试；所有桌面输入和截图均使用替身，不操作真实浏览器。"""
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from coursebot.automation import AutomationWorker, NavigationFailed, StopAutomation
from coursebot.capture import Frame
from coursebot.config import Settings
from coursebot.ocr import OcrLine
from coursebot.vision import PageView, SectionRow, TimeReading, VideoTask, analyze_page, parse_time_text
from coursebot.windows import WindowInfo
from coursebot.course_state import CourseSnapshot, CourseVideo, PendingChapter
from coursebot.accessibility import AccessibilityActionError


def line(text, y, x=130):
    return OcrLine(text, 0.99, x, y, x + 125, y + 20)


def reading(current, duration, y):
    box = line(f"{int(current)//60:02}:{int(current)%60:02} / {int(duration)//60:02}:{int(duration)%60:02}", y)
    return TimeReading(current, duration, box, box.text)


def image():
    # 彩色封面：视频检测不能依赖整块黑色。
    return np.full((900, 1400, 3), (180, 120, 60), np.uint8)


def view_with(*readings):
    return analyze_page(image(), [r.line for r in readings])


def worker():
    result = AutomationWorker(
        Settings(preview_enabled=False, auto_quiz=False),
        WindowInfo(1, "test", "test", 350, 180, 1400, 900),
        queue.Queue(), threading.Event(),
    )
    result._sleep = Mock()
    result._click_screen = Mock()
    result._ensure_foreground = Mock(return_value=True)
    result._probe_accessible_popup = Mock(return_value=False)
    result.clicker = Mock()
    return result


class VideoRecognitionTests(unittest.TestCase):
    def test_ocr_letter_o_in_zero_minutes_is_corrected(self):
        self.assertEqual(parse_time_text("O:12/13:33"), (12, 813))
        view = analyze_page(image(), [line("O:12/13:33", 400), reading(335, 736, 800).line])
        self.assertEqual(len(view.times), 2)
        self.assertEqual(view.time.current, 12)

    def test_initial_scan_chooses_first_of_two_unfinished_players(self):
        first, second = reading(12, 813, 400), reading(73, 736, 800)
        view = view_with(first, second)
        self.assertEqual(len(view.videos), 2)
        self.assertEqual(view.time.current, 12)
        self.assertEqual(view.time.duration, 813)

    def test_completed_task_does_not_hide_second_pending_task(self):
        lines = [line("任务点已完成", 180), reading(813, 813, 400).line,
                 line("任务点", 480), reading(73, 736, 800).line]
        img = image()
        cv2.circle(img, (110, 490), 9, (0, 180, 255), -1)
        view = analyze_page(img, lines)
        self.assertIs(view.task_completed, False)
        self.assertEqual(view.time.duration, 736)
        self.assertIs(view.active_video, view.videos[1])

    def test_periodic_scan_stays_on_current_player(self):
        first, second = reading(12, 813, 400), reading(73, 736, 800)
        view = analyze_page(image(), [first.line, second.line], preferred_time=second.line)
        self.assertEqual(view.time.duration, 736)

    def test_hidden_active_controls_do_not_switch_to_another_player(self):
        old_first = reading(12, 813, 400)
        second = reading(73, 736, 800)
        view = analyze_page(image(), [second.line], preferred_time=old_first.line)
        self.assertIsNone(view.time)

    def test_equal_duration_players_remain_separate(self):
        view = view_with(reading(120, 120, 400), reading(0, 120, 800))
        self.assertEqual(len(view.videos), 2)
        self.assertEqual(view.time.current, 0)

    def test_sidebar_time_is_not_a_player(self):
        view = analyze_page(image(), [line("00:12 / 13:33", 400), line("00:00 / 03:00", 800, x=1200)])
        self.assertEqual(len(view.times), 1)


class VideoNavigationTests(unittest.TestCase):
    def setUp(self):
        self.worker = worker()
        self.frame = Frame(image(), 350, 180, 0)

    def test_switch_resets_old_progress_and_resume_state(self):
        w = self.worker
        w._adopt_time(reading(813, 813, 400))
        w.end_streak, w.resume_attempts, w.fast_fail = 3, 2, 4
        w.completion_handled = True
        w._reset_video_tracking(reading(73, 736, 800))
        w._on_reading(reading(74, 736, 800))
        self.assertEqual(w.last_current, 74)
        self.assertEqual(w.duration, 736)
        self.assertEqual((w.end_streak, w.resume_attempts, w.fast_fail), (0, 0, 0))
        self.assertFalse(w.completion_handled)

    def test_completion_uses_each_readings_own_duration(self):
        self.worker.duration = 813
        self.assertTrue(self.worker._is_near_end(reading(120, 120, 400)))
        self.assertFalse(self.worker._is_near_end(reading(73, 736, 800)))

    def test_orange_two_does_not_prove_there_is_a_pending_video(self):
        row = SectionRow("3.1", "chapter", line("3.1 chapter", 200), 2, pending_tasks=2)
        view = PageView(1400, 900, [], sections=[row], current_idx=0)
        self.assertFalse(self.worker._video_needs_play(view))

    def test_orange_one_does_not_prove_video_page_is_finished(self):
        row = SectionRow("3.1", "chapter", line("3.1 chapter", 200), 2, pending_tasks=1)
        view = PageView(1400, 900, [], sections=[row], current_idx=0, task_completed=True)
        view.exam.quiz_tab = line("章节测验", 100)
        self.assertFalse(self.worker._maybe_enter_chapter_quiz(view))

    def test_current_view_is_checked_before_scrolling(self):
        w = self.worker
        finished, pending = reading(120, 120, 400), reading(0, 120, 800)
        w._adopt_time(finished)
        view = view_with(finished, pending)
        w._scan = Mock(return_value=view)
        self.assertTrue(w._try_next_video_on_page(self.frame))
        w.clicker.scroll.assert_not_called()
        w._click_screen.assert_not_called()
        self.assertEqual(w.last_current, 0)

    def test_scroll_discards_old_vertical_cutoff(self):
        w = self.worker
        finished = reading(813, 813, 750)
        w._adopt_time(finished)
        w._scan = Mock(side_effect=[view_with(finished), view_with(reading(73, 736, 360))])
        w._grab = Mock(return_value=self.frame)
        self.assertTrue(w._try_next_video_on_page(self.frame))
        self.assertEqual(w.time_reading.current, 73)
        self.assertEqual(w.time_box.y0, 360)
        w._click_screen.assert_not_called()
        self.assertEqual(w.clicker.scroll.call_count, 1)

    def test_confirmed_button_uses_screen_coordinates_once(self):
        w = self.worker
        task = VideoTask((100, 250, 900, 700), line("任务点", 200), False)
        view = PageView(1400, 900, [], videos=[task])
        w._find_center_play_button = Mock(return_value=(850, 655, True))
        self.assertTrue(w._activate_unfinished_video(self.frame, view, 0))
        w._click_screen.assert_called_once_with(850, 655, "同页未完成视频播放按钮")
        self.assertIsNone(w.last_current)

    def test_unconfirmed_poster_is_not_clicked(self):
        w = self.worker
        task = VideoTask((100, 250, 900, 700), line("任务点", 200), False)
        view = PageView(1400, 900, [], videos=[task])
        w._find_center_play_button = Mock(return_value=(850, 655, False))
        self.assertFalse(w._activate_unfinished_video(self.frame, view, 0))
        w._click_screen.assert_not_called()

    def test_scan_does_not_adopt_another_video_during_navigation(self):
        w = self.worker
        w._adopt_time(reading(813, 813, 750))
        w.ocr = Mock()
        w.ocr.full.return_value = [reading(73, 736, 360).line]
        view = w._scan(self.frame, track_time=False)
        self.assertEqual(view.time.current, 73)
        self.assertEqual(w.last_current, 813)

    def test_stable_bottom_allows_leaving_only_after_two_scrolls(self):
        w = self.worker
        finished = reading(120, 120, 400)
        w._adopt_time(finished)
        w._scan = Mock(return_value=view_with(finished))
        w._grab = Mock(return_value=self.frame)
        self.assertFalse(w._try_next_video_on_page(self.frame))
        self.assertEqual(w.clicker.scroll.call_count, 2)
        self.assertTrue(w._page_scan_complete)

    def test_unresolved_pending_video_at_bottom_prevents_leaving(self):
        w = self.worker
        task = VideoTask((100, 250, 900, 700), line("任务点", 200), False)
        w._scan = Mock(return_value=PageView(1400, 900, [], videos=[task]))
        w._grab = Mock(return_value=self.frame)
        w._find_center_play_button = Mock(return_value=(850, 655, False))
        with self.assertRaises(NavigationFailed):
            w._try_next_video_on_page(self.frame)
        self.assertFalse(w._page_scan_complete)

    def test_scroll_failure_prevents_leaving(self):
        w = self.worker
        w._scan = Mock(return_value=PageView(1400, 900, []))
        w.clicker.scroll.side_effect = RuntimeError("test scroll failure")
        with self.assertRaises(NavigationFailed):
            w._try_next_video_on_page(self.frame)

    def test_startup_completion_also_checks_other_players(self):
        w = self.worker
        w.view = view_with(reading(120, 120, 400))
        w._grab = Mock(return_value=self.frame)
        w._try_next_video_on_page = Mock(return_value=True)
        self.assertTrue(w._advance("开始检测到视频已结束"))
        w._try_next_video_on_page.assert_called_once()
        w._click_screen.assert_not_called()

    def test_dry_run_does_not_scroll_or_click(self):
        w = self.worker
        w.settings.dry_run = True
        with self.assertRaises(StopAutomation):
            w._try_next_video_on_page(self.frame)
        w.clicker.scroll.assert_not_called()
        w._click_screen.assert_not_called()

    def test_accessibility_loading_gap_does_not_call_old_playback(self):
        w = self.worker
        w.settings.save_debug_frames = False
        w._browser_reader = Mock()
        w._handle_accessible_nonvideo = Mock()
        w._scan = Mock()
        self.assertTrue(w._accessible_video_tick(CourseSnapshot("course:newchapter")))
        w._handle_accessible_nonvideo.assert_not_called()
        w._scan.assert_not_called()
        w._click_screen.assert_not_called()

    def test_changed_browser_control_is_retried_only_after_a_fresh_decision(self):
        w = self.worker
        w._browser_reader = Mock()
        w._browser_reader.activate.side_effect = AccessibilityActionError("stale")
        self.assertFalse(w._activate_accessible("old-token", "video3"))
        w._browser_reader.activate.assert_called_once()
        w._click_screen.assert_not_called()
        w._browser_reader.activate.side_effect = None
        self.assertTrue(w._activate_accessible("fresh-token", "video3"))
        self.assertNotIn("video3", w._action_failures)

    def test_repeated_action_failure_is_bounded(self):
        w = self.worker
        w._browser_reader = Mock()
        w._browser_reader.activate.side_effect = AccessibilityActionError("stale")
        self.assertFalse(w._activate_accessible("token", "video3"))
        self.assertFalse(w._activate_accessible("token", "video3"))
        with self.assertRaises(NavigationFailed):
            w._activate_accessible("token", "video3")

    def test_uncertain_invoke_is_not_repeated_when_fresh_snapshot_is_playing(self):
        w = self.worker
        w.settings.save_debug_frames = False
        w._browser_reader = Mock()
        w._browser_reader.activate.side_effect = AccessibilityActionError("invoke outcome unknown")
        paused = CourseSnapshot("course:123", [CourseVideo("v1", 1, False, 0, 500, True, "play")], page_kind="video")
        with patch("coursebot.automation.time.monotonic", return_value=10):
            w._accessible_video_tick(paused)
        with patch("coursebot.automation.time.monotonic", return_value=14):
            w._accessible_video_tick(paused)
        playing = CourseSnapshot("course:123", [CourseVideo("v1", 1, False, 1, 500, False, "pause")], page_kind="video")
        with patch("coursebot.automation.time.monotonic", return_value=15):
            w._accessible_video_tick(playing)
        w._browser_reader.activate.assert_called_once()
        w._click_screen.assert_not_called()

    def test_new_chapter_does_not_skip_using_temporarily_old_completed_player(self):
        w = self.worker
        w.settings.save_debug_frames = False
        w._browser_reader = Mock()
        w._open_accessible_pending = Mock()
        snapshot = CourseSnapshot("course:new", [CourseVideo("v1", 1, True)], page_kind="video")
        with patch("coursebot.automation.time.monotonic", return_value=10):
            self.assertTrue(w._accessible_video_tick(snapshot))
        w._open_accessible_pending.assert_not_called()
        w._browser_reader.activate.assert_not_called()

    def test_green_players_with_pending_current_chapter_wait_for_writeback(self):
        w = self.worker
        w.settings.save_debug_frames = False
        w._browser_reader = Mock()
        w._open_accessible_pending = Mock()
        snapshot = CourseSnapshot("course:123", [CourseVideo("v1", 1, True)],
                                  pending=[PendingChapter("cur123", "current", "token")], page_kind="video")
        with patch("coursebot.automation.time.monotonic", return_value=10):
            w._accessible_video_tick(snapshot)
        with patch("coursebot.automation.time.monotonic", return_value=14):
            w._accessible_video_tick(snapshot)
        w._open_accessible_pending.assert_not_called()
        w._browser_reader.activate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
