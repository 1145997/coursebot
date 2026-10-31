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


class AccessibilityActionError(RuntimeError):
    """控件移动或失效后必须重新读取，再按当前状态决定操作。"""


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


def select_course_document(root: AccessibleNode) -> AccessibleNode | None:
    """优先识别课程 URL，不依赖浏览器为文档设置的精确标题。"""
    documents = [node for node in root.walk() if node.kind == "DocumentControl"]
    for node in documents:
        for value in (node.value, node.name, node.help):
            parsed = urlparse(value)
            if "/studentstudy" in parsed.path and page_identity(AccessibleNode(kind="DocumentControl", value=value)):
                return node
    for node in documents:
        if "学生学习页面" in node.label and page_identity(node):
            return node
    for node in documents:
        if page_identity(node) and any(child.automation_id.startswith("cur") for child in node.walk()):
            return node
    return None


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
            if not any(re.fullmatch(r"\d+", name) and int(name) > 0 for name in names):
                continue
            title_node = next((item for item in children if item.label and
                               not re.fullmatch(r"[\d.\s]+", item.label)), None)
            if title_node:
                number = next((name for name in names if re.fullmatch(r"\d+(?:\.\d+)+", name)), "")
                title = f"{number} {title_node.label}".strip()
                snapshot.pending.append(PendingChapter(node.automation_id, title, title_node.token))
    tab_bar = next((node for node in ordered if node.automation_id == "prev_tab"), None)
    tab_nodes = [node for node in ordered if node.role in {"option", "tab"}]
    if tab_bar:
        tab_nodes += [node for node in tab_bar.walk() if re.fullmatch(r"dct\d+", node.automation_id)]
    if not snapshot.videos:
        video_tab = next((node for node in tab_nodes if
                          re.fullmatch(r"\d*\s*视频", node.label) and node.token), None)
        if video_tab:
            snapshot.video_tab_token = video_tab.token
            snapshot.video_tab_selected = video_tab.selected
    quiz_tab = next((node for node in tab_nodes if
                     re.fullmatch(r"\d*\s*章节测验", node.label) and node.token), None)
    if quiz_tab:
        snapshot.quiz_tab_token = quiz_tab.token
    main = next((node for node in ordered if node.automation_id == "mainid"), root)
    main_nodes = list(main.walk())
    text = " ".join(node.label for node in main_nodes)
    if snapshot.videos:
        snapshot.page_kind = "video"
    elif any(cue in text for cue in ("暂无内容", "暂无课程内容", "暂无学习内容", "暂无资源")):
        snapshot.page_kind = "empty"
    elif any(cue in text for cue in ("本章学习目标", "通过本章", "你需要掌握")):
        snapshot.page_kind = "intro"
    elif ("题量" in text or any(node.label in {"开始答题", "重新答题"} for node in main_nodes)
          or ("章节测验" in text and any(cue in text for cue in ("得分", "成绩", "已提交")))):
        snapshot.page_kind = "quiz"
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
        self._native = None
        self._ancestors = {}
        self._offscreen = {}
        self.last_action = ""
        self.last_tree: AccessibleNode | None = None

    def close(self):
        self._controls.clear()
        self._ancestors.clear()
        self._offscreen.clear()
        self._native = None
        if self._initializer:
            self._initializer.Uninitialize()
            self._initializer = None

    def _open(self):
        if self._auto is None:
            import uiautomation as auto
            self._auto = auto
            self._initializer = auto.UIAutomationInitializerInThread()
            # 每个使用线程持有自己的 COM 客户端，不跨 GUI/工作线程共享控件。
            import comtypes.client
            typelib = comtypes.client.GetModule("UIAutomationCore.dll")
            self._native = comtypes.client.CreateObject(
                "{ff48dba4-60ef-4201-aa87-54103eef594e}", interface=typelib.IUIAutomation
            )

    def _read_cached(self) -> CourseSnapshot:
        """批量抓取原生树缓存，避开逐属性跨进程遍历浏览器标签栏。"""
        started = time.monotonic()
        self._generation += 1
        self._controls.clear()
        self._ancestors.clear()
        self._offscreen.clear()
        self.last_tree = None
        client = self._native
        request = client.CreateCacheRequest()
        request.TreeScope = 7  # Element | Children | Descendants
        request.TreeFilter = client.RawViewCondition
        request.AutomationElementMode = 1  # Full: 保留对最近快照控件的实际操作能力
        props = {
            "name": 30005, "type": 30003, "id": 30011, "help": 30013,
            "role": 30101, "value": 30045, "legacy_value": 30093, "selected": 30079, "offscreen": 30022,
        }
        for prop in props.values():
            request.AddProperty(prop)
        root_element = client.ElementFromHandle(self.window.hwnd)
        cached = root_element.BuildUpdatedCache(request)
        count = 0
        truncated = False

        def value(element, key):
            try:
                return element.GetCachedPropertyValue(props[key])
            except Exception:
                return None

        def string(element, key):
            raw = value(element, key)
            return raw if isinstance(raw, str) else ""

        def visit(element, depth=0, ancestors=()):
            nonlocal count, truncated
            count += 1
            if count > self.node_limit or depth > 90:
                truncated = True
                return AccessibleNode()
            kind = {50030: "DocumentControl", 50000: "ButtonControl"}.get(value(element, "type"), "")
            token = f"{self._generation}:{count}"
            url = string(element, "value") or string(element, "legacy_value")
            node = AccessibleNode(
                name=string(element, "name"), help=string(element, "help"),
                automation_id=string(element, "id"), role=string(element, "role"),
                kind=kind, value=url, token=token,
            )
            selected = value(element, "selected")
            if isinstance(selected, (bool, int)):
                node.selected = bool(selected)
            self._controls[token] = self._auto.Control(element=element)
            self._ancestors[token] = ancestors
            self._offscreen[token] = value(element, "offscreen") is True
            try:
                children = element.GetCachedChildren()
                if children:
                    for index in range(children.Length):
                        node.children.append(visit(children.GetElement(index), depth + 1, ancestors + (token,)))
            except Exception:
                truncated = True
            return node

        tree = visit(cached)
        documents = [node for node in tree.walk() if node.kind == "DocumentControl"]
        diagnostics = {
            "strategy": "native-cache", "nodes": count, "documents": len(documents),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "document_titles": [node.name.split("?")[0][:100] for node in documents[:10]],
        }
        document = select_course_document(tree)
        if document is None:
            self._controls.clear()
            return CourseSnapshot("", complete=False,
                                  problem="浏览器未提供可识别的课程正文；已记录文档树诊断",
                                  diagnostics=diagnostics)
        self.last_tree = document
        result = snapshot_from_tree(document, not truncated,
                                    "浏览器文档树读取被截断" if truncated else "")
        result.diagnostics = diagnostics
        tab_candidates = []
        def inspect_tabs(node, parents=()):
            if re.fullmatch(r"\d*\s*(?:视频|章节测验)", node.label):
                tab_candidates.append({"label": node.label, "id": node.automation_id, "role": node.role,
                                       "parents": list(parents[-5:])})
            label = node.label.split("?")[0][:60]
            for child in node.children:
                inspect_tabs(child, parents + (f"{node.automation_id}|{node.role}|{label}",))
        inspect_tabs(document)
        diagnostics["task_tabs"] = tab_candidates[:30]
        if not result.complete:
            self._controls.clear()
        return result

    def read(self) -> CourseSnapshot:
        try:
            self._open()
            if self._native is not None:
                return self._read_cached()
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
        """原生语义操作不依赖屏幕坐标；只有坐标回退才要求滚到可见。"""
        control = self._controls.get(token)
        if control is None or not token.startswith(f"{self._generation}:"):
            raise AccessibilityActionError("操作目标已过期，需要重新扫描")
        self.last_action = ""
        try:
            self._activate_control(token, control, click)
        except Exception as exc:
            self._controls.clear()
            if isinstance(exc, AccessibilityActionError):
                raise
            raise AccessibilityActionError(f"浏览器控件已变化，重新定位（{type(exc).__name__}）") from exc

    def _activate_control(self, token, control, click):
        auto = self._auto
        invoke = control.GetPattern(auto.PatternId.InvokePattern)
        if invoke:
            if not invoke.Invoke(waitTime=0.1):
                raise AccessibilityActionError("浏览器未确认按钮操作，先复核当前播放状态")
            self.last_action = "UIA Invoke" + ("（屏幕外控件）" if self._offscreen.get(token) else "")
            return
        legacy_id = getattr(auto.PatternId, "LegacyIAccessiblePattern", None)
        legacy = control.GetPattern(legacy_id) if legacy_id is not None else None
        if legacy and legacy.DefaultAction:
            if not legacy.DoDefaultAction(waitTime=0.1):
                raise AccessibilityActionError("浏览器未确认默认动作，先重新读取页面")
            self.last_action = "UIA 默认动作"
            return
        # 嵌套 iframe 中仅滚动按钮不一定能移动外层正文，按外到内依次滚动祖先。
        for ancestor in self._ancestors.get(token, ()) + (token,):
            target = self._controls.get(ancestor)
            scroll = target.GetPattern(auto.PatternId.ScrollItemPattern) if target else None
            if scroll:
                scroll.ScrollIntoView(waitTime=0.1)
        if control.IsOffscreen:
            control.SetFocus()
        if control.IsOffscreen:
            raise AccessibilityActionError("目标尚未滚到可见区域，重新定位后再操作")
        rect = control.BoundingRectangle
        if control.IsOffscreen:
            raise AccessibilityActionError("滚动后目标位置改变，需要重新定位")
        x, y = int((rect.left + rect.right) / 2), int((rect.top + rect.bottom) / 2)
        if rect.right <= rect.left or rect.bottom <= rect.top or not self.window.contains(x, y):
            raise AccessibilityActionError("按钮未处于目标窗口可见区域")
        click(x, y, "整页任务清单定位的控件")
        self.last_action = "重新定位后点击"
