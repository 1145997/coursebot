"""适配器契约测试：使用控件替身，不访问真实桌面。"""
import types
import unittest
from unittest.mock import Mock

from coursebot.accessibility import AccessibleNode, AccessibilityActionError, BrowserAccessibility, select_course_document
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


class CachedElement:
    """只提供缓存读取接口，防止新实现退回逐属性跨进程读取。"""
    def __init__(self, control):
        self.control = control
        self.children = [CachedElement(child) for child in control.children]

    def GetCachedPropertyValue(self, prop):
        return {30005: self.control.Name, 30003: {"DocumentControl": 50030, "ButtonControl": 50000}.get(self.control.ControlTypeName, 50033),
                30011: self.control.AutomationId, 30013: self.control.HelpText, 30045: self.control.url}.get(prop, "")

    def GetCachedChildren(self):
        return types.SimpleNamespace(Length=len(self.children), GetElement=self.children.__getitem__)


def cached_reader(node_limit=6000):
    r = reader(node_limit=node_limit)
    doc = document()
    doc.Name = ""  # 浏览器不保证课程文档有标题。
    doc.children[-1].children[0].IsOffscreen = True
    cached = CachedElement(FakeControl("29 browser tabs", children=[FakeControl("tab") for _ in range(29)] + [doc]))
    request = types.SimpleNamespace(AddProperty=Mock())
    live_root = types.SimpleNamespace(BuildUpdatedCache=Mock(return_value=cached))
    r._native = types.SimpleNamespace(CreateCacheRequest=Mock(return_value=request), RawViewCondition=object(),
                                      ElementFromHandle=Mock(return_value=live_root))
    r._auto.Control = lambda element: element.control
    return r, request, live_root


class AccessibilityAdapterTests(unittest.TestCase):
    def test_native_cache_fetches_subtree_once_and_reads_all_three_players(self):
        r, request, live_root = cached_reader()
        snapshot = r.read()
        self.assertTrue(snapshot.complete)
        self.assertEqual([v.completed for v in snapshot.videos], [False, True, False])
        self.assertEqual(snapshot.diagnostics["strategy"], "native-cache")
        self.assertEqual(request.TreeScope, 7)
        self.assertEqual(request.AutomationElementMode, 1)
        live_root.BuildUpdatedCache.assert_called_once_with(request)
        self.assertTrue(r._controls[snapshot.videos[-1].play_token].IsOffscreen)

    def test_native_cache_truncation_revokes_action_tokens(self):
        r, _, _ = cached_reader(node_limit=35)
        self.assertFalse(r.read().complete)
        self.assertEqual(r._controls, {})

    def test_native_cache_failure_does_not_fall_back_to_partial_ocr_inventory(self):
        r, _, live_root = cached_reader()
        live_root.BuildUpdatedCache.side_effect = RuntimeError("provider disconnected")
        snapshot = r.read()
        self.assertFalse(snapshot.complete)
        self.assertEqual(snapshot.videos, [])
        self.assertEqual(r._controls, {})

    def test_document_is_identified_by_url_when_browser_changes_title(self):
        document = AccessibleNode(name="学生学习页面 - Microsoft Edge", kind="DocumentControl",
                                  value="https://mooc1.chaoxing.com/mycourse/studentstudy?courseId=c&chapterId=k")
        self.assertIs(select_course_document(AccessibleNode(children=[document])), document)

    def test_unnamed_course_document_is_also_identified_by_url(self):
        document = AccessibleNode(kind="DocumentControl",
                                  value="https://mooc1.chaoxing.com/mycourse/studentstudy?courseId=c&chapterId=k")
        self.assertIs(select_course_document(AccessibleNode(children=[document])), document)
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

    def test_offscreen_target_is_invoked_without_guessing_or_requiring_focus(self):
        doc = document()
        play = doc.children[1].children[0]
        play.IsOffscreen = True
        play.SetFocus = Mock(side_effect=RuntimeError("cannot focus iframe"))
        invoke = Mock()
        invoke.Invoke.return_value = True
        play.patterns[6] = invoke
        r = reader(doc)
        click = Mock()
        r.activate(r.read().videos[0].play_token, click)
        play.SetFocus.assert_not_called()
        click.assert_not_called()
        invoke.Invoke.assert_called_once()

    def test_failed_native_invoke_does_not_click_coordinates_or_reuse_tokens(self):
        doc = document()
        invoke = Mock()
        invoke.Invoke.side_effect = OSError("stale COM provider")
        doc.children[1].children[0].patterns[6] = invoke
        r = reader(doc)
        click = Mock()
        with self.assertRaises(AccessibilityActionError):
            r.activate(r.read().videos[0].play_token, click)
        click.assert_not_called()
        self.assertEqual(r._controls, {})

    def test_offscreen_default_action_does_not_require_coordinates(self):
        doc = document()
        play = doc.children[1].children[0]
        play.IsOffscreen = True
        legacy = Mock(DefaultAction="click")
        legacy.DoDefaultAction.return_value = True
        play.patterns[8] = legacy
        r = reader(doc)
        r._auto.PatternId.LegacyIAccessiblePattern = 8
        click = Mock()
        r.activate(r.read().videos[0].play_token, click)
        legacy.DoDefaultAction.assert_called_once()
        click.assert_not_called()

    def test_offscreen_without_semantic_action_never_uses_old_coordinates(self):
        doc = document()
        play = doc.children[1].children[0]
        play.IsOffscreen = True
        play.SetFocus = Mock()
        r = reader(doc)
        click = Mock()
        with self.assertRaises(AccessibilityActionError):
            r.activate(r.read().videos[0].play_token, click)
        click.assert_not_called()

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
