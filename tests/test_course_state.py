import unittest

from coursebot.accessibility import AccessibleNode as Node, snapshot_from_tree
from coursebot.course_state import Action, CourseController, CourseSnapshot, CourseVideo


def button(name="播放视频", token="play"):
    return Node(name=name, kind="ButtonControl", token=token)


def player(order, current=None, duration=None, paused=True):
    children = [button("播放视频" if paused else "暂停", f"play-{order}")]
    if current is not None:
        children += [Node(name="当前时间"), Node(name=current), Node(name="/"), Node(name="时长"), Node(name=duration)]
    return Node(name="视频播放器", automation_id="video", children=children)


def page(*children):
    return Node(name="学生学习页面", kind="DocumentControl",
                value="https://mooc1.chaoxing.com/mycourse/studentstudy?chapterId=chapter&courseId=course&enc=private",
                children=list(children))


def video(key="1", order=1, done=False, current=0, duration=500, paused=True):
    return CourseVideo(key, order, done, current, duration, paused, f"play-{key}")


class WholePageInventoryTests(unittest.TestCase):
    def test_actual_three_video_mixed_completion_case(self):
        tree = page(
            Node(help="任务点未完成", token="task1"), player(1),
            Node(help="任务点已完成", token="task2"), player(2, "5:15", "8:51"),
            # 第三个播放器在屏幕外仍在可访问性树中。
            Node(help="任务点未完成", token="task3"), player(3),
        )
        snapshot = snapshot_from_tree(tree)
        self.assertTrue(snapshot.complete)
        self.assertEqual(snapshot.page_key, "course:chapter")
        self.assertEqual(len(snapshot.videos), 3)
        self.assertEqual([v.completed for v in snapshot.videos], [False, True, False])
        self.assertNotIn("private", snapshot.page_key)
        self.assertIn("整页视频 3 个", snapshot.inventory_text())

    def test_separate_time_nodes_are_combined(self):
        snapshot = snapshot_from_tree(page(Node(name="任务点未完成", token="task1"), player(1, "5:15", "8:51", False)))
        self.assertEqual((snapshot.videos[0].current, snapshot.videos[0].duration), (315, 531))
        self.assertIs(snapshot.videos[0].paused, False)

    def test_missing_task_marker_does_not_copy_previous_status(self):
        snapshot = snapshot_from_tree(page(Node(name="任务点已完成", token="task1"), player(1), player(2)))
        self.assertEqual([v.completed for v in snapshot.videos], [True, None])

    def test_nested_video_wrappers_are_not_double_counted(self):
        tree = page(Node(name="任务点未完成", token="task1"), Node(name="视频播放器", children=[player(1)]))
        self.assertEqual(len(snapshot_from_tree(tree).videos), 1)

    def test_sidebar_pending_count_does_not_imply_video_count(self):
        row = Node(automation_id="cur123", children=[Node(name="3.1 心理健康", token="chapter-link"), Node(name="2")])
        snapshot = snapshot_from_tree(page(row))
        self.assertEqual(len(snapshot.videos), 0)
        self.assertEqual(snapshot.pending[0].token, "chapter-link")

    def test_zero_remaining_tasks_is_not_a_pending_chapter(self):
        row = Node(automation_id="cur123", children=[Node(name="11.1"), Node(name="问卷调查"), Node(name="0")])
        self.assertEqual(snapshot_from_tree(page(row)).pending, [])

    def test_sidebar_title_is_not_duplicated_by_nested_text_nodes(self):
        row = Node(automation_id="cur123", children=[Node(name="3.2"), Node(name="大学生常见情绪困扰", token="link",
                   children=[Node(name="大学生常见情绪困扰")]), Node(name="1")])
        self.assertEqual(snapshot_from_tree(page(row)).pending[0].title, "3.2 大学生常见情绪困扰")

    def test_complete_tree_is_required(self):
        s = snapshot_from_tree(page(Node(name="任务点未完成", token="task1"), player(1)), complete=False, problem="truncated")
        self.assertEqual(CourseController().decide(s, 0).action, Action.STOP)

    def test_video_tab_recognized_without_guessing_from_sidebar(self):
        snapshot = snapshot_from_tree(page(Node(name="2 视频", token="video-tab", role="option")))
        self.assertEqual(CourseController().decide(snapshot, 0).action, Action.OPEN_VIDEO_TAB)

    def test_sidebar_quiz_title_is_not_current_chapter_quiz_tab(self):
        snapshot = snapshot_from_tree(page(Node(name="章节测验", token="sidebar-quiz")))
        self.assertEqual(snapshot.quiz_tab_token, "")
        snapshot = snapshot_from_tree(page(Node(name="3 章节测验", token="real-tab", role="option")))
        self.assertEqual(snapshot.quiz_tab_token, "real-tab")

    def test_actual_listitem_tabs_are_found_only_inside_the_content_tab_bar(self):
        tree = page(Node(automation_id="prev_tab", children=[
            Node(name="视频", automation_id="dct1", role="listitem", token="video-tab"),
            Node(name="章节测验", automation_id="dct2", role="listitem", token="quiz-tab")]),
            Node(name="章节测验", role="listitem", token="sidebar-quiz"))
        snapshot = snapshot_from_tree(tree)
        self.assertEqual(snapshot.quiz_tab_token, "quiz-tab")
        self.assertEqual(snapshot.video_tab_token, "video-tab")

    def test_popup_quiz_is_not_treated_as_an_ordinary_pause(self):
        p = player(1)
        p.children += [Node(name="判断题"), button("提交", "submit")]
        snapshot = snapshot_from_tree(page(Node(name="任务点未完成", token="task1"), p))
        self.assertTrue(snapshot.popup_quiz)

    def test_nested_video_document_is_counted_without_a_region_label(self):
        doc = Node(kind="DocumentControl", value="https://mooc1.chaoxing.com/ananas/modules/video/index.html", children=[button()])
        self.assertEqual(len(snapshot_from_tree(page(Node(name="任务点未完成", token="task1"), doc)).videos), 1)

    def test_card_index_is_part_of_player_identity(self):
        card = Node(kind="DocumentControl", value="https://mooc1.chaoxing.com/knowledge/cards?courseid=course&knowledgeid=chapter&num=2")
        self.assertEqual(snapshot_from_tree(page(card)).page_key, "course:chapter@2")

    def test_previous_chapter_iframe_during_navigation_is_incomplete(self):
        card = Node(kind="DocumentControl", value="https://mooc1.chaoxing.com/knowledge/cards?courseid=course&knowledgeid=old&num=0")
        snapshot = snapshot_from_tree(page(card))
        self.assertFalse(snapshot.complete)
        self.assertEqual(snapshot.page_key, "course:chapter")


class SinglePlaybackControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = CourseController()

    def decide(self, *videos, now=0, page_key="chapter"):
        return self.controller.decide(CourseSnapshot(page_key, list(videos)), now)

    def test_completed_first_two_players_only_start_third(self):
        d = self.decide(video("1", 1, True), video("2", 2, True), video("3", 3, False))
        self.assertEqual(d.action, Action.PLAY)
        self.assertEqual(d.token, "play-3")

    def test_all_completed_players_never_replay(self):
        self.assertEqual(self.decide(video(done=True)).action, Action.PAGE_DONE)

    def test_unknown_task_status_prevents_all_input(self):
        self.assertEqual(self.decide(video(done=None)).action, Action.STOP)

    def test_playing_video_never_clicked_even_with_zero_progress(self):
        self.assertEqual(self.decide(video(paused=False)).action, Action.WAIT)

    def test_already_playing_later_segment_is_monitored_without_starting_another(self):
        d = self.decide(video("1"), video("2", 2, paused=False))
        self.assertEqual(d.action, Action.WAIT)
        self.assertEqual(d.video_key, "2")

    def test_completed_playing_video_is_paused_once_before_starting_pending(self):
        done_playing, pending = video("1", done=True, paused=False), video("2", 2)
        d = self.decide(done_playing, pending)
        self.assertEqual(d.action, Action.PAUSE_COMPLETED)
        self.controller.acted(d, 0)
        self.assertEqual(self.decide(done_playing, pending, now=1).action, Action.WAIT)
        self.assertEqual(self.decide(done_playing, pending, now=6).action, Action.STOP)
        self.assertEqual(self.decide(video("1", done=True), pending, now=7).action, Action.PLAY)

    def test_advancing_time_wins_over_misleading_play_button(self):
        self.assertEqual(self.decide(video(current=50), now=0).action, Action.WAIT)
        self.assertEqual(self.decide(video(current=51), now=10).action, Action.WAIT)

    def test_stopped_mid_video_waits_before_single_resume(self):
        self.assertEqual(self.decide(video(current=50), now=0).action, Action.WAIT)
        self.assertEqual(self.decide(video(current=50), now=7).action, Action.WAIT)
        d = self.decide(video(current=50), now=8)
        self.assertEqual(d.action, Action.PLAY)
        self.controller.acted(d, 8)
        self.assertEqual(self.decide(video(current=50), now=9).action, Action.WAIT)

    def test_two_failed_clicks_stop_without_toggle_loop(self):
        d = self.decide(video(), now=0)
        self.controller.acted(d, 0)
        d = self.decide(video(), now=15)
        self.controller.acted(d, 15)
        self.assertEqual(self.decide(video(), now=30).action, Action.STOP)

    def test_switching_to_lower_progress_is_new_identity(self):
        self.decide(video("1", current=490), now=0)
        self.assertEqual(self.decide(video("1", done=True), video("2", 2, current=0), now=5).action, Action.PLAY)

    def test_new_chapter_resets_previous_attempts(self):
        d = self.decide(video(), now=0)
        self.controller.acted(d, 0)
        self.assertEqual(self.decide(video(), now=1, page_key="newchapter").action, Action.PLAY)

    def test_end_time_without_completed_task_does_not_skip(self):
        self.assertEqual(self.decide(video(current=500), now=60).action, Action.WAIT)

    def test_empty_nonvideo_page_uses_existing_quiz_flow(self):
        self.assertEqual(self.controller.decide(CourseSnapshot("chapter", page_kind="quiz"), 0).action, Action.LEGACY_PAGE)

    def test_navigation_loading_gap_never_enters_legacy_playback(self):
        self.assertEqual(self.decide(now=0).action, Action.WAIT)
        self.assertEqual(self.decide(now=21).action, Action.STOP)

    def test_current_quiz_does_not_switch_back_to_video_tab(self):
        snapshot = CourseSnapshot("chapter", page_kind="quiz", video_tab_token="video-tab")
        self.assertEqual(self.controller.decide(snapshot, 0).action, Action.LEGACY_PAGE)

    def test_error_on_later_video_does_not_interrupt_current_video(self):
        later = CourseVideo("2", 2, False, play_token="play2", error_text="视频加载失败")
        self.assertEqual(self.decide(video(paused=False), later).action, Action.WAIT)
        self.assertEqual(self.decide(video(done=True), later).action, Action.LOAD_ERROR)

    def test_selected_video_tab_is_not_clicked_repeatedly(self):
        s = CourseSnapshot("chapter", video_tab_token="tab", video_tab_selected=True)
        self.assertEqual(self.controller.decide(s, 0).action, Action.WAIT)
        self.assertEqual(self.controller.decide(s, 31).action, Action.STOP)

    def test_loading_video_tab_obeys_cooldown_and_retry_limit(self):
        s = CourseSnapshot("chapter", video_tab_token="tab")
        d = self.controller.decide(s, 0)
        self.assertEqual(d.action, Action.OPEN_VIDEO_TAB)
        self.controller.acted(d, 0)
        self.assertEqual(self.controller.decide(s, 1).action, Action.WAIT)
        d = self.controller.decide(s, 10)
        self.controller.acted(d, 10)
        self.assertEqual(self.controller.decide(s, 20).action, Action.STOP)


if __name__ == "__main__":
    unittest.main()
