"""短数字、侧栏隔离、选中核验和补扫去重；不操作桌面、不调用真实 LLM。"""
import queue
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from coursebot.automation import AutomationWorker, ExamProgress, NavigationFailed, StopAutomation
from coursebot.capture import Frame
from coursebot.config import Settings
from coursebot.course_state import CourseSnapshot
from coursebot.ocr import OcrLine, TextOCR
from coursebot.vision import (ExamPage, ExamQuestion, PageView, QuizOption,
                             _build_exam_questions, _read_band_text)
from coursebot.windows import WindowInfo


def line(text, x, y, w=180, h=26):
    return OcrLine(text, .99, x, y, x + w, y + h)


def question(number="1", selected=None):
    return ExamQuestion(number, "单选题", "普通选择题", line(f"{number}【单选题】普通选择题", 100, 200),
        [QuizOption("A", "选项甲", line("选项甲", 160, 300), selected == "A", (127, 313)),
         QuizOption("B", "选项乙", line("选项乙", 160, 380), selected == "B", (127, 393))])


def page(questions, total=2, result=False):
    return PageView(1400, 1000, [], exam=ExamPage(detected=True, questions=questions, total=total,
        submit=line("提交", 700, 900), result=result))


def worker(view):
    w = AutomationWorker(Settings(save_debug_frames=False),
        WindowInfo(1, "test", "test", 0, 0, 1400, 1000), queue.Queue(), threading.Event())
    frame = Frame(np.full((1000, 1400, 3), 255, np.uint8), 0, 0, 0)
    w._log = Mock(); w._status = Mock(); w._save_debug = Mock(); w._sleep = Mock()
    w._scroll_exam = Mock(); w._scroll_exam_to_top = Mock(return_value=frame)
    w._grab = Mock(return_value=frame); w._scan = Mock(return_value=view)
    w._content_diff = Mock(return_value=0)
    w.llm = Mock()
    w.llm.solve.return_value = Mock(letters=["A"], reason="测试", latency=0, confidence=1)
    w._submit_exam = Mock(return_value=(frame, page([], result=True)))
    return w, frame


class ExamRecoveryTests(unittest.TestCase):
    def test_only_explicit_numbers_get_relaxed_recognition_threshold(self):
        ocr = TextOCR.__new__(TextOCR)
        ocr._ensure = Mock(); ocr._lock = threading.RLock()
        crop = np.full((40, 60, 3), 255, np.uint8)
        for text, score, expected in [("４", .28, "4"), ("7", .1, ""),
                                       ("侧栏文字", .3, ""), ("普通选项", .9, "普通选项")]:
            with self.subTest(text=text, score=score):
                ocr._line = Mock(return_value=([[[], text, str(score)]], None))
                self.assertEqual(ocr.line(crop), expected)

    def test_sidebar_text_never_fills_an_empty_option_row(self):
        q = question()
        image = np.full((1000, 1400, 3), 255, np.uint8)
        sidebar = line("5.3.1 目录项", 1100, 300)
        self.assertEqual(_read_band_text(image, q, 300, 326, None, [sidebar]), "")
        body = line("7", 180, 300, 12)
        self.assertEqual(_read_band_text(image, q, 300, 326, None, [sidebar, body]), "7")

    def test_short_numeric_options_and_partial_next_question_are_retained(self):
        image = np.full((1000, 1400, 3), 255, np.uint8)
        lines = [line("1【单选题】选择一个数值", 100, 200),
                 line("2【单选题】另一道题", 100, 760)]
        for cy, text in zip([310, 390, 470, 550, 850, 960], ["7", "8", "9", "6", "7", "8"]):
            cv2.circle(image, (127, cy), 20, (235, 235, 235), 2)
            cv2.putText(image, text, (180, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, .7, (20, 20, 20), 1)
            lines.append(line(text, 180, cy - 10, 15, 22))
        found = _build_exam_questions(image, lines, 1400, 1000, None)
        self.assertEqual([q.number for q in found], ["1", "2"])
        self.assertEqual([o.text for o in found[0].options], ["7", "8", "9", "6"])
        self.assertTrue(found[0].complete)
        self.assertFalse(found[1].complete)
        self.assertIsNotNone(found[0].options[0].control_center)

    def test_visible_question_without_options_is_not_silently_dropped(self):
        image = np.full((1000, 1400, 3), 255, np.uint8)
        found = _build_exam_questions(image, [line("1【单选题】题干", 100, 200)], 1400, 1000, None)
        self.assertEqual(len(found), 1)
        self.assertFalse(found[0].complete)

    def test_unreadable_control_row_keeps_later_option_letters(self):
        image = np.full((1000, 1400, 3), 255, np.uint8)
        lines = [line("1【单选题】普通题目", 100, 200),
                 line("选项乙", 180, 380), line("选项丙", 180, 460)]
        rows = [(300, 326, 127, 313, 20), (380, 406, 127, 393, 20),
                (460, 486, 127, 473, 20)]
        with patch("coursebot.vision._exam_control_rows", return_value=rows):
            found = _build_exam_questions(image, lines, 1400, 1000, None)[0]
        self.assertEqual([o.letter for o in found.options], ["B", "C"])
        self.assertFalse(found.complete)

    def test_existing_manual_choices_are_preserved_without_llm_or_counting(self):
        view = page([question("1", "A"), question("2", "B")])
        w, frame = worker(view)
        w._select_exam_options = Mock()
        self.assertTrue(w._handle_exam(view, frame))
        w.llm.solve.assert_not_called()
        w._select_exam_options.assert_not_called()
        w._submit_exam.assert_called_once()
        self.assertEqual(w.quizzes_solved, 0)

    def test_known_missing_question_never_reaches_submit(self):
        view = page([question("2", "B")])
        w, frame = worker(view)
        with self.assertRaisesRegex(NavigationFailed, "仍缺第 1 题，未提交"):
            w._handle_exam(view, frame, _rescan_depth=2)
        w._submit_exam.assert_not_called()
        w.llm.solve.assert_not_called()

    def test_confirmed_answer_is_not_recounted_or_resolved_on_rescan(self):
        first = page([question("1")])
        second = page([question("1", "A"), question("2", "B")])
        w, frame = worker(first)
        stage = [0]
        def top():
            stage[0] += 1
            return frame
        w._scroll_exam_to_top.side_effect = top
        w._scan.side_effect = lambda _frame: first if stage[0] == 1 else second
        def select(q, letters):
            for opt in q.options:
                opt.selected = opt.letter in letters
            return True
        w._select_exam_options = Mock(side_effect=select)
        self.assertTrue(w._handle_exam(first, frame))
        self.assertEqual(w._scroll_exam_to_top.call_count, 2)
        w.llm.solve.assert_called_once()
        self.assertEqual(w.quizzes_solved, 1)
        w._submit_exam.assert_called_once()

    def test_unconfirmed_click_is_not_counted_or_submitted(self):
        view = page([question()], total=1)
        w, frame = worker(view)
        w._select_exam_options = Mock(return_value=False)
        with self.assertRaisesRegex(NavigationFailed, "三次未确认选中"):
            w._handle_exam(view, frame)
        self.assertEqual(w.quizzes_solved, 0)
        w._submit_exam.assert_not_called()
        w.llm.solve.assert_called_once()
        self.assertEqual(w._select_exam_options.call_count, 3)

    def test_click_is_not_success_without_actual_selected_pixels(self):
        view = page([question()], total=1)
        w, frame = worker(view)
        w._click_exam_option = Mock()
        w._sleep = Mock()
        self.assertFalse(w._select_exam_options(view.exam.questions[0], ["A"]))
        self.assertEqual(w._click_exam_option.call_count, 2)

    def test_click_confirmation_uses_actual_control_center(self):
        q = question()
        w, frame = worker(page([q], total=1))
        w._click_exam_option = Mock()
        cv2.circle(frame.image, (127, 313), 20, (255, 160, 50), -1)
        self.assertTrue(w._select_exam_options(q, ["A"]))

    def test_multiselect_retries_only_the_remaining_difference(self):
        q = question(); q.kind = "多选题"
        w, frame = worker(page([q], total=1))
        w._click_exam_option = Mock()
        with patch("coursebot.automation.selected_exam_options", side_effect=[{"A", "B"}, {"B"}]):
            self.assertTrue(w._select_exam_options(q, ["B"]))
        self.assertEqual([call.args[1].letter for call in w._click_exam_option.call_args_list], ["B", "A"])

    def test_question_identity_not_answer_count_determines_missing_numbers(self):
        p = ExamProgress(total=2, answered={"2": {"A"}, "3": {"A"}})
        self.assertEqual(p.missing(), ["1"])
        p.total = None; p.seen = {"2", "3"}
        self.assertEqual(p.missing(), ["1"])

    def test_exam_completion_respects_auto_next_setting(self):
        w, _frame = worker(page([]))
        w.settings.auto_next = False
        with self.assertRaises(StopAutomation):
            w._open_accessible_pending(CourseSnapshot("course:chapter", complete=True))
        self.assertIn("自动切换已关闭", w._stop_reason)

    def test_verification_only_never_solves_or_changes_unselected_question(self):
        view = page([question()], total=1)
        w, frame = worker(view)
        w.verify_exam_only = True
        w._select_exam_options = Mock()
        with self.assertRaisesRegex(NavigationFailed, "仍缺第 1 题"):
            w._handle_exam(view, frame, _rescan_depth=2)
        w.llm.solve.assert_not_called()
        w._select_exam_options.assert_not_called()
        w._submit_exam.assert_not_called()

    def test_verification_only_accepts_existing_answers_without_auto_quiz(self):
        view = page([question("1", "A"), question("2", "B")])
        w, frame = worker(view)
        w.verify_exam_only = True
        w.settings.auto_quiz = False
        w._select_exam_options = Mock()
        self.assertTrue(w._handle_exam(view, frame))
        w.llm.solve.assert_not_called()
        w._select_exam_options.assert_not_called()
        w._submit_exam.assert_called_once()

    def test_verification_entry_initializes_capture_and_stays_on_current_quiz(self):
        view = page([question("1", "A"), question("2", "B")])
        w, frame = worker(view)
        w.verify_exam_only = True
        w._prepare_window_capture = Mock(side_effect=lambda: setattr(w, "capture", Mock()))
        w._main_loop = Mock()
        w._verify_answered_exam()
        w._prepare_window_capture.assert_called_once()
        w._submit_exam.assert_called_once()
        w.llm.solve.assert_not_called()
        w._main_loop.assert_not_called()
        self.assertIn("未进入下一节", w._stop_reason)
