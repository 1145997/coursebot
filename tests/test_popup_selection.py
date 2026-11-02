"""弹题漏字、侧栏干扰、单选状态与提交反馈的回归。"""
import queue
import threading
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from coursebot.automation import AutomationWorker, NavigationFailed
from coursebot.capture import Frame
from coursebot.config import Settings
from coursebot.ocr import OcrLine
from coursebot.vision import PageView, QuizOption, QuizView, analyze_page, detect_quiz
from coursebot.windows import WindowInfo


def line(text, x, y, x1=None, y1=None):
    return OcrLine(text, .99, x, y, x1 or x + 80, y1 or y + 26)


def incomplete_ocr():
    return [line("判断题", 289, 531, 374, 561),
            line("在一超与多强之间综合国力差距缩小。", 355, 593, 804, 623),
            line("B、错", 412, 781, 493, 807),
            line("√3 军事思想", 1927, 854, 2057, 879),
            line("不懂", 1666, 923, 1708, 947),
            line("回答错误", 1422, 1261, 1532, 1290),
            line("提交", 1619, 1261, 1669, 1290)]


def quiz():
    return QuizView(detected=True, kind="判断题", question="题干", signature="q1",
        options=[QuizOption("A", "对", line("A、对", 412, 690, 479, 714)),
                 QuizOption("B", "错", line("B、错", 412, 781, 493, 807))],
        submit=line("提交", 1619, 1261, 1669, 1290))


class PopupSelectionTests(unittest.TestCase):
    def setUp(self):
        self.w = AutomationWorker(Settings(save_debug_frames=False),
            WindowInfo(1, "test", "test", 0, 0, 2560, 1528), queue.Queue(), threading.Event())
        self.w._click_screen = Mock()
        self.w._click_line = Mock()
        self.w._sleep = Mock()
        self.w._log = Mock()
        self.w._save_debug = Mock()
        self.q = quiz()
        pixels = np.zeros((1528, 2560, 3), np.uint8)
        cv2.circle(pixels, (371, 702), 12, (255, 255, 255), -1)
        cv2.circle(pixels, (371, 794), 12, (255, 150, 60), -1)
        self.frame = Frame(pixels, 0, 0, 0)

    def test_missing_a_never_uses_sidebar_check_or_wrong_feedback_as_options(self):
        found = detect_quiz(incomplete_ocr(), 2560, 1528)
        self.assertEqual([opt.letter for opt in found.options], ["B"])
        self.assertEqual(found.options[0].line.x0, 412)
        self.assertNotIn("不懂", found.question)

    def test_focused_column_recovers_a_and_preserves_window_coordinates(self):
        reader = Mock(return_value=[line("A、对", 153, 129, 220, 153),
                                    line("B、错", 153, 220, 234, 246),
                                    line("在一超与多强之间", 98, 35, 329, 58)])
        view = analyze_page(self.frame.image, incomplete_ocr(), popup_reader=reader)
        self.assertEqual([opt.letter for opt in view.quiz.options], ["A", "B"])
        self.assertEqual((view.quiz.options[0].line.x0, view.quiz.options[0].line.y0), (412, 690))
        self.assertEqual(view.quiz.question.count("在一超与多强之间"), 1)
        reader.assert_called_once()

    def test_radio_click_is_verified_from_the_actual_selected_circle(self):
        self.assertEqual(self.w._popup_selected_letters(self.frame, self.q), {"B"})
        after = self.frame.image.copy()
        cv2.circle(after, (371, 702), 12, (255, 150, 60), -1)
        cv2.circle(after, (371, 794), 12, (255, 255, 255), -1)
        self.w._grab = Mock(return_value=Frame(after, 0, 0, 0))
        _, confirmed = self.w._set_popup_radio(self.q, "A", self.frame)
        self.assertTrue(confirmed)
        x, y, _ = self.w._click_screen.call_args.args
        self.assertLessEqual(abs(x - 371), 6)
        self.assertLessEqual(abs(y - 702), 2)

    def test_blue_lecture_background_does_not_select_unchecked_radio_controls(self):
        pixels = np.full((1528, 2560, 3), (100, 50, 0), np.uint8)
        cv2.circle(pixels, (371, 702), 12, (255, 255, 255), -1)
        cv2.circle(pixels, (371, 794), 12, (255, 255, 255), -1)
        frame = Frame(pixels, 0, 0, 0)
        self.assertEqual(set(self.w._popup_radio_controls(frame, self.q)), {"A", "B"})
        self.assertEqual(self.w._popup_selected_letters(frame, self.q), set())
        cv2.circle(pixels, (371, 794), 12, (255, 150, 60), -1)
        self.assertEqual(self.w._popup_selected_letters(frame, self.q), {"B"})

    def test_missing_round_control_is_not_clicked_or_submitted(self):
        pixels = np.full((1528, 2560, 3), (100, 50, 0), np.uint8)
        frame = Frame(pixels, 0, 0, 0)
        self.w._grab = Mock(return_value=frame)
        self.w._scan = Mock(return_value=PageView(2560, 1528, [], quiz=self.q))
        _, confirmed = self.w._set_popup_radio(self.q, "A", frame)
        self.assertFalse(confirmed)
        self.w._click_screen.assert_not_called()
        self.w._click_line.assert_not_called()
        self.assertEqual(self.w._grab.call_count, 2)

    def test_large_merged_option_box_does_not_change_radio_size_or_offset(self):
        self.q.options[0] = QuizOption("A", "背景讲义", line("A)背景讲义", 396, 673, 758, 764))
        controls = self.w._popup_radio_controls(self.frame, self.q)
        self.assertEqual(set(controls), {"A", "B"})
        self.assertLessEqual(abs(controls["A"][0][0] - 371), 3)
        self.assertLessEqual(abs(controls["A"][0][1] - 702), 3)
        self.assertEqual(self.w._popup_selected_letters(self.frame, self.q), {"B"})

    def test_large_a_box_cannot_borrow_bs_circle_when_a_is_missing(self):
        self.q.options[0] = QuizOption("A", "背景讲义", line("A)背景讲义", 396, 673, 758, 764))
        pixels = self.frame.image.copy()
        cv2.circle(pixels, (371, 702), 18, (0, 0, 0), -1)
        controls = self.w._popup_radio_controls(Frame(pixels, 0, 0, 0), self.q)
        self.assertNotIn("A", controls)
        self.assertIn("B", controls)

    def test_two_options_still_trigger_focused_recovery_when_a_contains_lecture(self):
        lines = [line("判断题", 289, 531, 374, 561), line("判断题题干", 355, 593, 800, 623),
                 line("A)背景讲义", 396, 673, 758, 764), line("B、错", 410, 778, 484, 812),
                 line("提交", 1619, 1261, 1669, 1290)]
        reader = Mock(return_value=[line("A、对", 153, 129, 220, 155),
                                    line("B、错", 151, 220, 221, 249),
                                    line("背景文字", 210, 130, 330, 187)])
        found = analyze_page(self.frame.image, lines, popup_reader=reader).quiz
        reader.assert_called_once()
        self.assertEqual(found.options[0].text, "对")
        self.assertEqual(found.options[0].line.height, 26)
        self.assertEqual((found.options[0].line.x0, found.options[0].line.y0), (412, 690))

    def test_canonical_judgment_label_wins_over_longer_lecture_in_either_order(self):
        bad = line("A)背景讲义文字", 396, 673, 758, 764)
        good = line("A、对", 412, 690, 480, 716)
        for ordered in ([bad, good], [good, bad]):
            with self.subTest(first=ordered[0].text):
                found = detect_quiz(incomplete_ocr() + ordered, 2560, 1528)
                self.assertEqual(found.options[0].text, "对")
                self.assertEqual(found.options[0].line.height, 26)

    def test_temporary_missing_circle_is_reread_before_clicking_the_real_control(self):
        before = Frame(np.zeros_like(self.frame.image), 0, 0, 0)
        after = self.frame.image.copy()
        cv2.circle(after, (371, 702), 12, (255, 150, 60), -1)
        cv2.circle(after, (371, 794), 12, (255, 255, 255), -1)
        self.w._grab = Mock(side_effect=[self.frame, Frame(after, 0, 0, 0)])
        self.w._scan = Mock(return_value=PageView(2560, 1528, [], quiz=self.q))
        _, confirmed = self.w._set_popup_radio(self.q, "A", before)
        self.assertTrue(confirmed)
        self.w._click_screen.assert_called_once()
        self.w._click_line.assert_not_called()

    def test_recovery_rejects_a_different_question_without_clicking(self):
        changed = quiz()
        changed.question = "另一道题"
        before = Frame(np.zeros_like(self.frame.image), 0, 0, 0)
        selected_a = self.frame.image.copy()
        cv2.circle(selected_a, (371, 702), 12, (255, 150, 60), -1)
        cv2.circle(selected_a, (371, 794), 12, (255, 255, 255), -1)
        self.w._grab = Mock(return_value=Frame(selected_a, 0, 0, 0))
        self.w._scan = Mock(return_value=PageView(2560, 1528, [], quiz=changed))
        _, confirmed = self.w._set_popup_radio(self.q, "A", before)
        self.assertFalse(confirmed)
        self.w._click_screen.assert_not_called()

    def test_explicit_judgment_type_is_not_changed_by_color_or_parentheses(self):
        self.q.question = "题干包含（）和()"
        self.w.view = PageView(2560, 1528, [line("讲义中出现多选题字样", 600, 900)])
        self.w._popup_selected_letters = Mock(return_value={"A", "B"})
        self.w._set_popup_radio = Mock(return_value=(self.frame, True))
        self.w._set_popup_combo = Mock()
        self.w._submit_popup_answer = Mock(return_value=(self.frame, "closed", None))
        self.w._popup_after_success = Mock(return_value=None)
        self.w._solve_popup_enumeration(self.q, self.frame)
        self.assertEqual(self.q.kind, "判断题")
        self.assertEqual(self.w._popup_enumeration_plan(self.q), [["A"], ["B"]])
        self.w._set_popup_combo.assert_not_called()

    def test_popup_header_takes_priority_over_question_type_words_in_lecture(self):
        lines = incomplete_ocr() + [line("A、对", 412, 690), line("讲义中的多选题示例", 700, 900)]
        self.assertEqual(detect_quiz(lines, 2560, 1528).kind, "判断题")

    def test_unconfirmed_radio_never_submits_or_counts_as_a_wrong_answer(self):
        self.w._set_popup_radio = Mock(return_value=(self.frame, False))
        self.w._submit_popup_answer = Mock()
        with self.assertRaisesRegex(NavigationFailed, "未确认选中"):
            self.w._solve_popup_enumeration(self.q, self.frame)
        self.w._submit_popup_answer.assert_not_called()
        self.assertEqual(self.w.quizzes_solved, 0)

    def test_feedback_ignores_red_lecture_content_outside_the_submit_row(self):
        self.w.ocr = Mock()
        self.w.ocr.line.return_value = ""
        cv2.rectangle(self.frame.image, (500, 580), (1200, 640), (0, 0, 255), -1)
        self.assertEqual(self.w._read_popup_feedback(self.frame, self.q)[1:], (False, False))
        cv2.rectangle(self.frame.image, (1422, 1261), (1510, 1288), (0, 0, 255), -1)
        self.assertEqual(self.w._read_popup_feedback(self.frame, self.q)[1:], (False, True))

    def test_old_error_is_not_consumed_before_the_new_submission_can_close(self):
        self.w._grab = Mock(return_value=self.frame)
        self.w._popup_submit_present = Mock(side_effect=[True, True, True, False])
        self.w._read_popup_feedback = Mock(return_value=("回答错误", False, True))
        self.w._scan = Mock(return_value=PageView(2560, 1528, []))
        _, result, _ = self.w._submit_popup_answer(self.q)
        self.assertEqual(result, "closed")

    def test_unknown_feedback_retries_the_same_choice_without_consuming_b(self):
        self.w._set_popup_radio = Mock(return_value=(self.frame, True))
        self.w._submit_popup_answer = Mock(side_effect=[(self.frame, "unknown", self.q), (self.frame, "closed", None)])
        self.w._popup_after_success = Mock(return_value=None)
        self.w._solve_popup_enumeration(self.q, self.frame)
        self.assertEqual([call.args[1] for call in self.w._set_popup_radio.call_args_list], ["A", "A"])
        self.assertEqual(self.w.quizzes_solved, 1)

    def test_confirmed_wrong_a_then_closed_b_is_a_completed_question(self):
        self.w._set_popup_radio = Mock(return_value=(self.frame, True))
        self.w._submit_popup_answer = Mock(side_effect=[(self.frame, "wrong", None), (self.frame, "closed", None)])
        self.w._popup_after_success = Mock(return_value=None)
        self.w._solve_popup_enumeration(self.q, self.frame)
        self.assertEqual([call.args[1] for call in self.w._set_popup_radio.call_args_list], ["A", "B"])
        self.assertEqual(self.w.quizzes_solved, 1)

    def test_two_unknown_submissions_are_not_reported_as_two_wrong_choices(self):
        self.w._set_popup_radio = Mock(return_value=(self.frame, True))
        self.w._submit_popup_answer = Mock(return_value=(self.frame, "unknown", self.q))
        with self.assertRaisesRegex(NavigationFailed, "反馈仍未确认"):
            self.w._solve_popup_enumeration(self.q, self.frame)
        self.assertEqual([call.args[1] for call in self.w._set_popup_radio.call_args_list], ["A", "A"])
        self.assertEqual(self.w.quizzes_solved, 0)


if __name__ == "__main__":
    unittest.main()
