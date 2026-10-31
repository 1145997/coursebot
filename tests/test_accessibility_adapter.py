"""适配器契约测试：使用控件替身，不访问真实桌面。"""
import types
import unittest
from unittest.mock import Mock

from coursebot.accessibility import BrowserAccessibility
from coursebot.windows import WindowInfo


class FakeControl:
    def __init__(self, name="", kind="PaneControl", children=(), url="", help_text="", automation_id=""):
        self.Name = name
        self.ControlTypeName = kind
        self.HelpText = help_text
        self.AutomationId = automation_id
        self.url = url
        self.children = list(children)
        self.IsOffscreen = False
        self.BoundingRectangle = types.SimpleNamespace(left=100, right=200, top=100, bottom=150)
        self.patterns = {}

    def GetChildren(self):
        return self.children

    def GetPropertyValue(self, property_id):
        return self.url if property_id in (2, 3) else ""

    def GetPattern(self, pattern_id):
        return self.patterns.get(pattern_id)

    def SetFocus(self):
        self.IsOffscreen = False


def document():
    children = []
    for index, status in enumerate(("任务点未完成", "任务点已完成", "任务点未完成")):
        children.append(FakeControl(help_text=status))
        children.append(FakeControl("视频播放器", children=[FakeControl("播放视频", "ButtonControl")], automation_id="video"))
    return FakeControl("学生学习页面", "DocumentControl", children,
                       "https://mooc1.chaoxing.com/mycourse/studentstudy?courseId=course&chapterId=chapter")


def reader(doc=None, node_limit=6000):
    r = BrowserAccessibility(WindowInfo(1, "test", "test", 0, 0, 1400, 900), node_limit=node_limit)
    r._auto = types.SimpleNamespace(
        ControlFromHandle=Mock(return_value=FakeControl("browser", "WindowControl", [doc or document()])),
        PropertyId=types.SimpleNamespace(AriaRoleProperty=1, ValueValueProperty=2, LegacyIAccessibleValueProperty=3,
                                         SelectionItemIsSelectedProperty=4),
        PatternId=types.SimpleNamespace(ScrollItemPattern=5, InvokePattern=6, SelectionItemPattern=7),
    )
    return r


class AccessibilityAdapterTests(unittest.TestCase):
    def test_entire_document_is_traversed_including_third_player(self):
        r = reader()
        snapshot = r.read()
        self.assertTrue(snapshot.complete)
        self.assertEqual([v.completed for v in snapshot.videos], [False, True, False])

    def test_node_limit_invalidates_partial_inventory_and_action_tokens(self):
        r = reader(node_limit=4)
        snapshot = r.read()
        self.assertFalse(snapshot.complete)
        self.assertEqual(r._controls, {})

    def test_stale_target_is_rejected_after_rescan(self):
        r = reader()
        token = r.read().videos[0].play_token
        r.read()
        with self.assertRaises(RuntimeError):
            r.activate(token, Mock())

    def test_native_invoke_does_not_use_guessed_coordinates(self):
        doc = document()
        play = doc.children[1].children[0]
        invoke = Mock()
        invoke.Invoke.return_value = True
        play.patterns[6] = invoke
        r = reader(doc)
        token = r.read().videos[0].play_token
        click = Mock()
        r.activate(token, click)
        invoke.Invoke.assert_called_once()
        click.assert_not_called()

    def test_offscreen_target_is_focused_before_native_action(self):
        doc = document()
        play = doc.children[1].children[0]
        play.IsOffscreen = True
        invoke = Mock()
        invoke.Invoke.return_value = True
        play.patterns[6] = invoke
        r = reader(doc)
        r.activate(r.read().videos[0].play_token, Mock())
        self.assertFalse(play.IsOffscreen)
        invoke.Invoke.assert_called_once()

    def test_missing_course_document_never_returns_a_video_inventory(self):
        r = reader(FakeControl("another page", "DocumentControl"))
        s = r.read()
        self.assertFalse(s.complete)
        self.assertEqual(s.videos, [])

    def test_unknown_origin_is_not_accepted_as_a_course(self):
        doc = document()
        doc.url = "https://notchaoxing.com/studentstudy?courseId=course&chapterId=chapter"
        r = reader(doc)
        s = r.read()
        self.assertFalse(s.complete)
        self.assertEqual(r._controls, {})


if __name__ == "__main__":
    unittest.main()
