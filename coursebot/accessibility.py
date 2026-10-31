"""通过 Windows 浏览器可访问性树读取整页，而不是只数可见截图。

UIA 失效/截断时明确返回不完整，绝不把部分可见节点当成全部视频。
所有 COM 对象均在使用它们的工作线程创建和释放。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from .course_state import CourseSnapshot, CourseVideo, PendingChapter
from .vision import parse_time_text


@dataclass
class AccessibleNode:
    name: str = ""
    help: str = ""
    automation_id: str = ""
    role: str = ""
    kind: str = ""
    value: str = ""
    token: str = ""
    children: list[AccessibleNode] = field(default_factory=list)
    selected: bool | None = None

    @property
    def label(self) -> str:
        return self.name or self.help

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


def is_player(node: AccessibleNode) -> bool:
    return (node.label == "视频播放器" or node.automation_id == "video" or
            (node.kind == "DocumentControl" and
             any("/ananas/modules/video/" in value for value in (node.value, node.name))))


def task_status(node: AccessibleNode) -> bool | None:
    text = node.label.replace(" ", "")
    if text in {"任务点未完成", "未完成任务点"}:
        return False
    if text in {"任务点已完成", "已完成任务点"}:
        return True
    return None


def page_identity(root: AccessibleNode) -> str:
    base = ""
    card = ""
    for node in root.walk():
        if node.kind != "DocumentControl":
            continue
        for candidate in (node.value, node.name, node.help):
            parsed = urlparse(candidate)
            if not parsed.hostname or not (parsed.hostname == "chaoxing.com" or parsed.hostname.endswith(".chaoxing.com")):
                continue
            query = parse_qs(parsed.query)
            chapter = query.get("chapterId") or query.get("knowledgeid")
            course = query.get("courseId") or query.get("courseid")
            if chapter and course:
                # 身份键只保留章节和课程编号，不存储 enc 等登录参数。
                if not base:
                    base = f"{course[0]}:{chapter[0]}"
                if "num" in query and f"{course[0]}:{chapter[0]}" == base:
                    card = query["num"][0]
    return f"{base}@{card}" if base and card else base


def snapshot_from_tree(root: AccessibleNode, complete: bool = True, problem: str = "") -> CourseSnapshot:
    snapshot = CourseSnapshot(page_identity(root), complete=complete, problem=problem)
    if not snapshot.page_key:
        snapshot.complete = False
        snapshot.problem = snapshot.problem or "无法确认课程章节身份"
    ordered = list(root.walk())
    base = snapshot.page_key.split("@")[0]
    for node in ordered:
        if node.kind == "DocumentControl":
            parsed = urlparse(node.value or node.name)
            if "/knowledge/cards" in parsed.path:
                query = parse_qs(parsed.query)
                course, chapter = query.get("courseid"), query.get("knowledgeid")
                if course and chapter and base and f"{course[0]}:{chapter[0]}" != base:
                    snapshot.complete = False
                    snapshot.problem = "章节正文正在切换，整页任务清单尚未同步"
    # 跳过浏览器有时提供的嵌套 video/region 重复壳，只保留最外层播放器。
    players: list[AccessibleNode] = []
    def collect(node):
        if is_player(node):
            players.append(node)
        else:
            for child in node.children:
                collect(child)
    collect(root)
    last_task: bool | None = None
    previous_task_node = ""
    associated_tasks: set[str] = set()
    player_ids = {id(node) for node in players}
    for index, node in enumerate(ordered):
        status = task_status(node)
        if status is not None:
            last_task = status
            previous_task_node = node.token or f"marker:{index}"
        if id(node) not in player_ids:
            continue
        inside = list(node.walk())
        texts = [item.label for item in inside if item.label]
        raw = " ".join(texts).replace("当前时间", "").replace("时长", "")
        parsed = parse_time_text(raw)
        buttons = [item for item in inside if item.kind == "ButtonControl" or item.role == "button"]
        if (any(kind in raw for kind in ("判断题", "单选题", "多选题")) and
                any(item.label in {"提交", "提交答案"} for item in buttons)):
            snapshot.popup_quiz = True
        pause = next((item for item in buttons if item.label in {"暂停", "暂停视频", "Pause"}), None)
        play = next((item for item in buttons if item.label in {"播放视频", "播放", "Play"}), None)
        state = last_task
        if previous_task_node in associated_tasks:
            # 缺失一个任务标记时不能借用上一段的状态。
            state = None
        elif previous_task_node:
            associated_tasks.add(previous_task_node)
        order = len(snapshot.videos) + 1
        snapshot.videos.append(CourseVideo(
            key=f"{snapshot.page_key}:video:{order}", order=order, completed=state,
            current=parsed[0] if parsed else None, duration=parsed[1] if parsed else None,
            paused=False if pause else True if play else None,
            play_token=(pause or play).token if (pause or play) else "",
            error_text=next((cue for cue in ("下载中途失败", "尝试其他线路", "视频加载失败") if cue in raw), ""),
        ))
    for node in ordered:
        if node.automation_id.startswith("cur"):
            children = list(node.walk())[1:]
            names = [item.label.strip() for item in children if item.label.strip()]
            # 橙色数字是该目录行的未完成任务数，只用于确认有待办。
            # 不能据此判断它是视频、测验还是阅读。
            if not any(re.fullmatch(r"\d+", name) for name in names):
                continue
            title_node = next((item for item in children if item.label and
                               not re.fullmatch(r"[\d.\s]+", item.label)), None)
            if title_node:
                snapshot.pending.append(PendingChapter(node.automation_id, " ".join(names), title_node.token))
    if not snapshot.videos:
        video_tab = next((node for node in ordered if node.role in {"option", "tab"} and
                          re.fullmatch(r"\d*\s*视频", node.label) and node.token), None)
        if video_tab:
            snapshot.video_tab_token = video_tab.token
            snapshot.video_tab_selected = video_tab.selected
    quiz_tab = next((node for node in ordered if node.role in {"option", "tab"} and
                     re.fullmatch(r"\d*\s*章节测验", node.label) and node.token), None)
    if quiz_tab:
        snapshot.quiz_tab_token = quiz_tab.token
    return snapshot


class BrowserAccessibility:
    def __init__(self, window, node_limit: int = 6000, seconds: float = 10):
        self.window = window
        self.node_limit = node_limit
        self.seconds = seconds
        self._auto = None
        self._initializer = None
        self._controls = {}
        self._generation = 0

    def close(self):
        self._controls.clear()
        if self._initializer:
            self._initializer.Uninitialize()
            self._initializer = None

    def _open(self):
        if self._auto is None:
            import uiautomation as auto
            self._auto = auto
            self._initializer = auto.UIAutomationInitializerInThread()

    def read(self) -> CourseSnapshot:
        try:
            self._open()
            root = self._auto.ControlFromHandle(self.window.hwnd)
            if root is None:
                return CourseSnapshot("", complete=False, problem="目标浏览器窗口已失效")
            self._generation += 1
            self._controls.clear()
            started = time.monotonic()
            count = 0
            truncated = False
            def find_document(control, depth=0):
                if depth > 10 or time.monotonic() - started > self.seconds:
                    return None
                try:
                    if control.ControlTypeName == "DocumentControl" and control.Name == "学生学习页面":
                        return control
                    for child in control.GetChildren():
                        result = find_document(child, depth + 1)
                        if result is not None:
                            return result
                except Exception:
                    return None
                return None
            document = find_document(root)
            if document is None:
                return CourseSnapshot("", complete=False, problem="没有找到当前课程文档，请确认已选择课程浏览器窗口")
            def visit(control, depth=0):
                nonlocal count, truncated
                count += 1
                if count > self.node_limit or depth > 55 or time.monotonic() - started > self.seconds:
                    truncated = True
                    return AccessibleNode()
                try:
                    token = f"{self._generation}:{count}"
                    name = str(control.Name or "")
                    kind = str(control.ControlTypeName or "")
                    automation_id = str(control.AutomationId or "")
                    help_text = str(control.HelpText or "")
                    role = str(control.GetPropertyValue(self._auto.PropertyId.AriaRoleProperty) or "")
                    value = ""
                    if kind == "DocumentControl":
                        for prop in (self._auto.PropertyId.ValueValueProperty,
                                     self._auto.PropertyId.LegacyIAccessibleValueProperty):
                            candidate = str(control.GetPropertyValue(prop) or "")
                            if candidate.startswith("http"):
                                value = candidate
                                break
                    node = AccessibleNode(name, help_text, automation_id, role, kind, value, token)
                    if role in {"option", "tab"}:
                        selection = control.GetPattern(self._auto.PatternId.SelectionItemPattern)
                        if selection:
                            node.selected = bool(control.GetPropertyValue(self._auto.PropertyId.SelectionItemIsSelectedProperty))
                    self._controls[token] = control
                    # 地址栏/工具栏不属于课程页面；只遍历文档内的内容。
                    for child in control.GetChildren():
                        node.children.append(visit(child, depth + 1))
                    return node
                except Exception:
                    truncated = True
                    return AccessibleNode()
            tree = visit(document)
            result = snapshot_from_tree(tree, not truncated,
                                        "浏览器可访问性树读取不完整，请重试扫描" if truncated else "")
            if not result.complete or not result.page_key:
                self._controls.clear()
            return result
        except Exception as exc:
            return CourseSnapshot("", complete=False, problem=f"整页可访问性读取失败：{type(exc).__name__}")

    def activate(self, token: str, click) -> None:
        """仅使用最近一次快照的元素，先滚到可见，再调用原生按钮动作。"""
        control = self._controls.get(token)
        if control is None or not token.startswith(f"{self._generation}:"):
            raise RuntimeError("操作目标已过期，需要重新扫描")
        auto = self._auto
        scroll = control.GetPattern(auto.PatternId.ScrollItemPattern)
        if scroll:
            scroll.ScrollIntoView(waitTime=0.1)
        if control.IsOffscreen:
            control.SetFocus()
        if control.IsOffscreen:
            raise RuntimeError("操作目标仍在屏幕外，不能使用旧坐标点击")
        invoke = control.GetPattern(auto.PatternId.InvokePattern)
        if invoke:
            if not invoke.Invoke(waitTime=0.1):
                raise RuntimeError("浏览器未接受按钮操作")
            return
        rect = control.BoundingRectangle
        if control.IsOffscreen:
            raise RuntimeError("操作目标仍在屏幕外，不能使用旧坐标点击")
        x, y = int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)
        if rect.right <= rect.left or rect.bottom <= rect.top or not self.window.contains(x, y):
            raise RuntimeError("按钮未处于目标窗口可见区域")
        click(x, y, "整页任务清单定位的控件")
