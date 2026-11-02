"""截图失败也必须恢复悬浮窗的普通截图行为，不影响其它窗口。"""
import unittest
from unittest.mock import Mock, patch

from coursebot import capture


class OverlayCaptureTests(unittest.TestCase):
    def setUp(self):
        self.api = Mock(return_value=True)
        self.api_patch = patch.object(capture, "_display_affinity", return_value=self.api)
        self.api_patch.start()
        self.addCleanup(self.api_patch.stop)
        self.windows_patch = patch.object(capture, "_overlay_windows", set())
        self.windows_patch.start()
        self.addCleanup(self.windows_patch.stop)

    def test_grab_failure_restores_capture_and_unregistered_windows_are_untouched(self):
        self.assertTrue(capture.register_capture_overlay(101))
        self.assertTrue(capture.register_capture_overlay(202))
        capture.unregister_capture_overlay(202)
        self.api.reset_mock()
        with self.assertRaises(RuntimeError):
            with capture.exclude_capture_overlays():
                self.api.assert_called_once_with(101, 0x11)
                raise RuntimeError("capture failed")
        self.assertEqual([call.args for call in self.api.call_args_list], [(101, 0x11), (101, 0)])

    def test_unsupported_window_does_not_remain_registered(self):
        self.api.return_value = False
        self.assertFalse(capture.register_capture_overlay(101))
        self.api.reset_mock()
        with capture.exclude_capture_overlays():
            pass
        self.api.assert_not_called()

    def test_capture_failure_to_exclude_one_window_restores_only_successful_handles(self):
        capture.register_capture_overlay(101)
        capture.register_capture_overlay(202)
        self.api.reset_mock()
        self.api.side_effect = lambda hwnd, mode: hwnd == 101
        with capture.exclude_capture_overlays():
            pass
        self.assertIn((101, 0), [call.args for call in self.api.call_args_list])
        self.assertNotIn((202, 0), [call.args for call in self.api.call_args_list])
