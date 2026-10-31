from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

import cv2
import numpy as np

from .ocr import OcrLine, compact_text, normalize_text, similarity

TIME_RE = re.compile(
    r"(?<![\d:])(\d{1,3})\s*[:：]\s*(\d{1,2})\s*[/／|]\s*(\d{1,3})\s*[:：]\s*(\d{1,2})(?![\d:])"
)
SECTION_RE = re.compile(r"^(?:第\s*)?(\d+(?:\.\d+){1,3})\s*[、.．]?\s*(.+)$")
OPTION_RE = re.compile(r"^\s*([A-H])\s*[.．、:：)）]\s*(\S.*)$")
OPTION_SPACE_RE = re.compile(r"^\s*([A-H])\s+(\S.*)$")
OPTION_ONLY_RE = re.compile(r"^\s*([A-H])\s*$")
OPTION_LETTER_RE = re.compile(r"^\s*([A-H])\s*[.．、:：)）]?\s*$")
QUESTION_NO_RE = re.compile(r"第\s*(\d+)\s*题")
QUESTION_RE = re.compile(
    r"^\s*(\d{1,3})\s*[.．、]?\s*[【\[\(（]\s*(单选题|多选题|判断题|填空题|简答题)\s*[】\]\)）]\s*(.*)$"
)
TOTAL_RE = re.compile(r"题量\s*[:：]\s*(\d+)")

UI_NOISE = {
    "返回课程",
    "章节详情",
    "目录",
    "讨论",
    "笔记",
    "标注",
    "搜索",
    "写笔记",
    "知识点",
    "任务点",
    "AI助教",
    "聊天",
    "学生学习页面",
}


@dataclass
class TimeReading:
    current: float
    duration: float
    line: OcrLine
    raw: str

    @property
    def percent(self) -> float:
        if self.duration <= 0:
            return 0.0
        return max(0.0, min(1.0, self.current / self.duration))

    def current_text(self) -> str:
        return format_seconds(self.current)

    def duration_text(self) -> str:
        return format_seconds(self.duration)


@dataclass
class VideoTask:
    """同一屏里的一个播放器；坐标均相对于窗口截图。"""

    bounds: tuple[int, int, int, int]
    label: OcrLine | None = None
    completed: bool | None = None
    time: TimeReading | None = None


@dataclass
class SectionRow:
    number: str
    title: str
    line: OcrLine
    level: int
    completed: bool = False
    pending_tasks: int | None = None

    @property
    def key(self) -> str:
        return self.number

    @property
    def display(self) -> str:
        return f"{self.number} {self.title}".strip()


@dataclass
class QuizOption:
    letter: str
    text: str
    line: OcrLine
    selected: bool = False


@dataclass
class QuizView:
    detected: bool = False
    kind: str = "选择题"
    question: str = ""
    options: list[QuizOption] = field(default_factory=list)
    submit: OcrLine | None = None
    next_button: OcrLine | None = None
    feedback: str = ""
    is_result: bool = False
    multiple_sets: bool = False
    signature: str = ""

    def option_map(self) -> dict[str, QuizOption]:
        return {o.letter: o for o in self.options}


@dataclass
class ExamQuestion:
    number: str
    kind: str
    text: str
    start: OcrLine
    options: list[QuizOption] = field(default_factory=list)
    complete: bool = True

    @property
    def key(self) -> str:
        return self.number

    @property
    def display(self) -> str:
        return f"{self.number}【{self.kind}】{self.text}".strip()


@dataclass
class ExamPage:
    detected: bool = False
    questions: list[ExamQuestion] = field(default_factory=list)
    submit: OcrLine | None = None
    confirm: bool = False
    confirm_submit: OcrLine | None = None
    cancel: OcrLine | None = None
    result: bool = False
    next_button: OcrLine | None = None
    quiz_tab: OcrLine | None = None
    total: int | None = None
    alert: str = ""

    @property
    def answered_count(self) -> int:
        return sum(1 for q in self.questions if q.options)


@dataclass
class PageView:
    width: int
    height: int
    lines: list[OcrLine]
    time: TimeReading | None = None
    sections: list[SectionRow] = field(default_factory=list)
    current_idx: int | None = None
    current_evidence: str = ""
    page_title: str = ""
    page_title_line: OcrLine | None = None
    quiz: QuizView = field(default_factory=QuizView)
    exam: ExamPage = field(default_factory=ExamPage)
    next_button: OcrLine | None = None
    resume_button: OcrLine | None = None
    start_button: OcrLine | None = None
    quiz_entry: OcrLine | None = None
    completion_line: OcrLine | None = None
    task_line: OcrLine | None = None
    task_completed: bool | None = None
    times: list[TimeReading] = field(default_factory=list)
    videos: list[VideoTask] = field(default_factory=list)
    active_video: VideoTask | None = None


def format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    m, s = divmod(total, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_time_text(text: str) -> tuple[float, float] | None:
    text = normalize_text(text)
    # 浅色控制栏的零常被 OCR 读成字母 O，只在完整时间格式前纠正。
    text = re.sub(r"(?<!\w)[Oo](?=\s*:\s*\d{1,2}\s*[/／|])", "0", text)
    m = TIME_RE.search(text)
    if not m:
        return None
    a, b, c, d = (int(x) for x in m.groups())
    if b >= 60 or d >= 60:
        return None
    current = a * 60 + b
    duration = c * 60 + d
    if duration <= 1 or current > duration + 3:
        return None
    if duration > 6 * 3600:
        return None
    return float(current), float(duration)


def find_time_readings(lines: Iterable[OcrLine]) -> list[TimeReading]:
    candidates: list[TimeReading] = []
    for line in lines:
        parsed = parse_time_text(line.text)
        if not parsed:
            continue
        candidates.append(TimeReading(parsed[0], parsed[1], line, line.text))
    candidates.sort(key=lambda t: (t.line.y0, t.line.x0))
    return candidates


def find_time_reading(lines: Iterable[OcrLine]) -> TimeReading | None:
    candidates = find_time_readings(lines)
    return candidates[0] if candidates else None


def find_video_tasks(
    image: np.ndarray, lines: list[OcrLine], readings: list[TimeReading]
) -> list[VideoTask]:
    """将任务点、完成状态、时间关联起来，不要求视频画面为黑色。"""
    h, w = image.shape[:2]
    content_right = int(w * 0.72)
    labels: list[OcrLine] = []
    for line in sorted(lines, key=lambda item: item.y0):
        text = compact_text(line.text)
        if line.x0 >= content_right or not text.startswith("任务点"):
            continue
        if "完成条件" in text or len(text) > 12:
            continue
        if labels and abs(line.cy - labels[-1].cy) < line.height:
            continue
        labels.append(line)
    videos: list[VideoTask] = []
    used: set[int] = set()
    for index, label in enumerate(labels):
        bottom = labels[index + 1].y0 - 8 if index + 1 < len(labels) else h - 2
        matches = [
            (i, reading) for i, reading in enumerate(readings)
            if label.y1 < reading.line.cy < bottom and reading.line.x0 < content_right
        ]
        reading = matches[0][1] if matches else None
        if matches:
            used.add(matches[0][0])
        completed = True if "已完成" in compact_text(label.text) else None
        if "未完成" in compact_text(label.text):
            completed = False
        if completed is None:
            x0, x1 = max(0, label.x0 - 40), max(0, label.x0 - 2)
            y0, y1 = max(0, label.cy - 18), min(h, label.cy + 18)
            crop = image[y0:y1, x0:x1]
            if crop.size:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                yellow = int(np.count_nonzero(cv2.inRange(hsv, (8, 80, 100), (40, 255, 255))))
                green = int(np.count_nonzero(cv2.inRange(hsv, (40, 80, 80), (90, 255, 255))))
                if yellow >= 12 and yellow >= green:
                    completed = False
                elif green >= 12 and green > yellow:
                    completed = True
        x0 = max(0, label.x0 - 45)
        y0 = min(bottom - 1, label.y1 + 8)
        # 未显示控制栏时，仅估计这个任务点下方的画面，不跨入下个任务点。
        y1 = reading.line.y1 if reading else min(bottom, y0 + int((content_right - x0) * 0.65))
        videos.append(VideoTask((x0, y0, content_right, max(y0 + 1, y1)), label, completed, reading))
    for i, reading in enumerate(readings):
        if i in used or reading.line.x0 >= content_right:
            continue
        y0 = readings[i - 1].line.y1 + 12 if i else max(0, reading.line.y0 - int(w * 0.38))
        videos.append(VideoTask((int(w * 0.04), y0, content_right, reading.line.y1), time=reading))
    return sorted(videos, key=lambda video: video.bounds[1])


def select_video(videos: list[VideoTask], preferred: OcrLine | None = None) -> VideoTask | None:
    if preferred:
        # 周期性全页扫描继续跟踪当前控制栏，避免跳到同屏另一个播放器。
        nearby = [video for video in videos if video.time and
                  abs(video.time.line.cy - preferred.cy) <= max(24, preferred.height * 2) and
                  abs(video.time.line.cx - preferred.cx) <= max(80, preferred.width)]
        if nearby:
            return min(nearby, key=lambda video: abs(video.time.line.cy - preferred.cy))
        containing = [video for video in videos if video.time is None and
                      video.bounds[1] <= preferred.cy <= video.bounds[3] + 30]
        # 当前控制栏隐藏时不能把另一个播放器的时间接管过来。
        return containing[0] if containing else None
    pending = [video for video in videos if video.completed is not True and
               (video.time is None or video.time.current < video.time.duration - 0.5)]
    return next(iter(pending or videos), None)


def parse_section_line(text: str) -> tuple[str, str, int] | None:
    raw = normalize_text(text)
    if not raw or TIME_RE.search(raw):
        return None
    m = SECTION_RE.match(raw)
    if not m:
        return None
    number, title = m.group(1), m.group(2).strip(" .．、-—:：")
    if not title:
        return None
    if len(title) > 80:
        return None
    level = number.count(".") + 1
    if level > 4:
        return None
    if re.fullmatch(r"[\d\s.:：/]+", title):
        return None
    return number, title, level


def find_sections(lines: Iterable[OcrLine], width: int, height: int) -> list[SectionRow]:
    rows: dict[str, SectionRow] = {}
    for line in lines:
        if line.x0 < width * 0.58:
            continue
        if line.y0 < height * 0.05 or line.y0 > height * 0.99:
            continue
        parsed = parse_section_line(line.text)
        if not parsed:
            continue
        number, title, level = parsed
        row = SectionRow(number, title, line, level)
        old = rows.get(number)
        if old is None or line.score > old.line.score:
            rows[number] = row
    return sorted(rows.values(), key=lambda r: (r.line.y0, r.line.x0))


def _pale_blue_ratio(image: np.ndarray, row: SectionRow) -> float:
    h, w = image.shape[:2]
    x0 = max(0, row.line.x0 - 22)
    x1 = min(w, row.line.x1 + 70)
    y0 = max(0, row.line.y0 - 8)
    y1 = min(h, row.line.y1 + 8)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    b = patch[:, :, 0].astype(np.int16)
    g = patch[:, :, 1].astype(np.int16)
    r = patch[:, :, 2].astype(np.int16)
    mask = (b > 232) & (g > 228) & (r > 210) & ((b - r) >= 5) & ((b - g) >= 1)
    return float(mask.mean())


def find_page_title(
    lines: Iterable[OcrLine], width: int, sections: list[SectionRow]
) -> tuple[OcrLine | None, float, int | None]:
    """通过页内大标题与右侧目录标题的相似度识别当前小节。"""
    if not sections:
        return None, 0.0, None
    first_section_y = min(s.line.y0 for s in sections)
    candidates: list[OcrLine] = []
    for line in lines:
        if line.x0 > width * 0.60:
            continue
        if line.y0 > min(first_section_y - 8, 500):
            continue
        if compact_text(line.text) in {compact_text(x) for x in UI_NOISE}:
            continue
        if line.height < 16 or len(compact_text(line.text)) < 3:
            continue
        if TIME_RE.search(line.text):
            continue
        candidates.append(line)

    best_line = None
    best_score = 0.0
    best_idx = None
    for cand in candidates:
        for idx, sec in enumerate(sections):
            sim = similarity(cand.text, sec.title)
            # 标题越靠上、字号越大，越可能是页面标题
            score = sim + min(0.06, cand.height / 1000.0)
            if score > best_score:
                best_score, best_line, best_idx = score, cand, idx
    if best_line is None or best_score < 0.55:
        return None, 0.0, None
    return best_line, best_score, best_idx


def find_current_section(
    image: np.ndarray,
    sections: list[SectionRow],
    title_line: OcrLine | None,
    title_idx: int | None,
    title_score: float,
    previous_key: str | None = None,
) -> tuple[int | None, str]:
    if not sections:
        return None, ""
    ratios = [_pale_blue_ratio(image, row) for row in sections]
    best_ratio_idx = int(np.argmax(ratios)) if ratios else None
    best_ratio = ratios[best_ratio_idx] if best_ratio_idx is not None else 0.0

    selected: int | None = None
    evidence = ""
    if best_ratio >= 0.10 and best_ratio_idx is not None:
        selected = best_ratio_idx
        evidence = f"highlight:{best_ratio:.2f}"
    if title_idx is not None and title_score >= 0.58:
        if selected is None or title_score >= 0.80 and best_ratio < 0.18:
            selected = title_idx
            evidence = f"title:{title_score:.2f}"
    if selected is None and previous_key:
        for idx, row in enumerate(sections):
            if row.key == previous_key:
                selected = idx
                evidence = "previous"
                break
    if selected is None and ratios:
        # 没有高亮也没有标题匹配时，不贸然选择
        return None, ""
    return selected, evidence


def pick_next_section(
    sections: list[SectionRow], current_idx: int | None, current_level: int | None = None
) -> int | None:
    if not sections or current_idx is None or current_idx < 0:
        return None
    if current_idx >= len(sections) - 1:
        return None
    level = current_level or sections[current_idx].level
    for j in range(current_idx + 1, len(sections)):
        row = sections[j]
        if row.level < level:
            continue
        if row.completed:
            continue
        return j
    # 若被父节点截断，退回到紧邻的下一项（不点已完成项）
    fallback = current_idx + 1
    if fallback < len(sections) and not sections[fallback].completed:
        return fallback
    return None


def find_button(lines: Iterable[OcrLine], names: Iterable[str], exact: bool = True) -> OcrLine | None:
    wanted = [compact_text(n) for n in names]
    best: tuple[int, OcrLine] | None = None
    for line in lines:
        text = compact_text(line.text)
        if not text:
            continue
        for i, name in enumerate(wanted):
            ok = text == name if exact else name in text
            if ok:
                # 名称越靠前优先级越高；同优先级取面积较大的按钮
                key = i * 1_000_000 - min(line.width * line.height, 900_000)
                if best is None or key < best[0]:
                    best = (key, line)
                break
    return best[1] if best else None


def find_next_button(lines: Iterable[OcrLine]) -> OcrLine | None:
    return find_button(
        lines,
        [
            "下一节",
            "下一课",
            "下一讲",
            "继续下一节",
            "下一节视频",
            "下一章",
            "下一关",
        ],
        exact=True,
    )


def find_resume_button(lines: Iterable[OcrLine]) -> OcrLine | None:
    return find_button(
        lines,
        ["继续学习", "继续播放", "继续观看", "点击继续", "我知道了", "知道了", "好的", "确定", "关闭"],
        exact=False,
    )


def find_start_button(lines: Iterable[OcrLine]) -> OcrLine | None:
    return find_button(
        lines,
        ["开始学习", "开始播放", "开始观看", "继续学习", "播放视频", "播放"],
        exact=True,
    )


def find_quiz_entry(lines: Iterable[OcrLine]) -> OcrLine | None:
    return find_button(
        lines,
        ["开始答题", "开始测验", "开始考试", "去答题", "立即答题", "进入答题"],
        exact=False,
    )


def find_completion_line(lines: Iterable[OcrLine]) -> OcrLine | None:
    for line in lines:
        text = compact_text(line.text)
        if "任务点已完成" in text or "已完成学习" in text or "视频已看完" in text:
            return line
    return None


def find_quiz_tab(lines: Iterable[OcrLine], width: int, height: int) -> OcrLine | None:
    """视频页顶部的「2 章节测验」标签。"""
    candidates = []
    for line in lines:
        text = compact_text(line.text)
        if "章节测验" not in text:
            continue
        if line.y0 > height * 0.28:
            continue
        if line.x0 > width * 0.72:
            continue
        candidates.append(line)
    if not candidates:
        return None
    candidates.sort(key=lambda l: (l.y0, l.x0))
    return candidates[0]


def mark_completed_sections(image: np.ndarray, sections: list[SectionRow]) -> None:
    """右侧目录最右边的绿色勾表示已完成，橙色圆点数字表示未完成任务数。"""
    h, w = image.shape[:2]
    x0 = int(w * 0.925)
    for row in sections:
        y0 = max(0, row.line.cy - 26)
        y1 = min(h, row.line.cy + 26)
        patch = image[y0:y1, x0:w]
        if patch.size == 0:
            continue
        b = patch[:, :, 0].astype(np.int16)
        g = patch[:, :, 1].astype(np.int16)
        r = patch[:, :, 2].astype(np.int16)
        green = (g > 115) & (g > r + 25) & (g > b + 15) & (r < 190)
        row.completed = float(green.mean()) > 0.006
        if row.completed:
            row.pending_tasks = 0
            continue
        pending = _read_pending_badge(patch, row.line.cy - y0)
        if pending is not None:
            row.pending_tasks = pending


def _read_pending_badge(patch: np.ndarray, row_center_y: int) -> int | None:
    """识别橙色任务圆点中的白色数字：0/1/2。"""
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    orange = (
        (hsv[:, :, 0] > 5)
        & (hsv[:, :, 0] < 35)
        & (hsv[:, :, 1] > 80)
        & (hsv[:, :, 2] > 120)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_dist = 1_000_000
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if not (14 <= bw <= 52 and 14 <= bh <= 52 and area >= 120):
            continue
        dist = abs((y + bh / 2) - row_center_y)
        if dist < best_dist and dist < 35:
            best_dist = dist
            best = (x, y, bw, bh)
    if best is None:
        return None
    x, y, bw, bh = best
    roi = patch[y : y + bh, x : x + bw]
    hv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 橙色圆内白色数字
    digit = ((hv[:, :, 2] > 200) & (hv[:, :, 1] < 80)).astype(np.uint8)
    hh, ww = digit.shape
    yy, xx = np.ogrid[:hh, :ww]
    circle = (xx - ww / 2) ** 2 + (yy - hh / 2) ** 2 < (min(hh, ww) * 0.42) ** 2
    digit[~circle] = 0
    ys, xs = np.where(digit > 0)
    if len(xs) == 0:
        return None
    dw = int(xs.max() - xs.min() + 1)
    dh = int(ys.max() - ys.min() + 1)
    ratio = dw / max(1, dh)
    # 数字 0 有内孔；数字 1 很窄；其余按 2 处理
    sub = digit[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    cnts, hierarchy = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        holes = sum(1 for item in hierarchy[0] if int(item[3]) >= 0)
    if ratio < 0.50 and holes == 0:
        return 1
    if holes > 0:
        return 0
    return 2


def _clean_option_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[^0-9A-Za-z\u4e00-\u9fff]+", "", text)
    text = re.sub(r"^[A-Ha-h]\s*[.．、:：)）]\s*", "", text)
    text = re.sub(r"^[A-Ha-h]\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"[\s·•,，。;；:：!！?？、\-—_/\\|]+$", "", text)
    return text.strip()[:300]


def _question_starts(lines: list[OcrLine], width: int, height: int) -> list[ExamQuestion]:
    candidates: list[ExamQuestion] = []
    for line in lines:
        if line.x0 > width * 0.62 or line.y0 < height * 0.03:
            continue
        m = QUESTION_RE.match(normalize_text(line.text))
        if not m:
            continue
        number, kind, body = m.group(1), m.group(2), m.group(3).strip()
        candidates.append(ExamQuestion(number=number, kind=kind, text=body, start=line))
    # OCR 有时把“9【判断题】”和题干拆成两行；也可能把题干放在同一行右侧。
    for q in candidates:
        if q.text and len(compact_text(q.text)) >= 3:
            continue
        same_row: list[OcrLine] = []
        below: list[OcrLine] = []
        for line in lines:
            if line is q.start or line.x0 > width * 0.62:
                continue
            body = normalize_text(line.text)
            if not body or QUESTION_RE.match(body):
                continue
            if (
                abs(line.cy - q.start.cy) <= max(18, q.start.height)
                and line.x0 >= q.start.x1 - 6
            ):
                same_row.append(line)
            elif 0 < line.y0 - q.start.y1 < 45:
                below.append(line)
        for line in sorted(same_row, key=lambda item: item.x0)[:1] + below[:1]:
            q.text = normalize_text(line.text)
            if q.text:
                break
    dedup: dict[str, ExamQuestion] = {}
    for q in candidates:
        if q.number not in dedup:
            dedup[q.number] = q
    return sorted(dedup.values(), key=lambda q: (q.start.y0, q.start.x0))


def _row_bands(image: np.ndarray, q: ExamQuestion, next_y: int, width: int) -> list[tuple[int, int]]:
    h = image.shape[0]
    x0 = max(0, q.start.x0 - 10)
    x1 = min(width, q.start.x0 + 880)
    y0 = max(0, q.start.y0 + 8)
    y1 = min(h, next_y - 3)
    if x1 <= x0 or y1 <= y0:
        return []
    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    ink = (gray < 185).sum(axis=1)
    rows = ink > 3
    bands: list[tuple[int, int]] = []
    start = None
    for i, has_ink in enumerate(rows):
        if has_ink and start is None:
            start = i
        elif not has_ink and start is not None:
            if i - start >= 6:
                bands.append((y0 + start, y0 + i - 1))
            start = None
    if start is not None and len(rows) - start >= 6:
        bands.append((y0 + start, y0 + len(rows) - 1))
    return bands


def _read_band_text(
    image: np.ndarray,
    q: ExamQuestion,
    y0: int,
    y1: int,
    line_reader: Callable[[np.ndarray], str] | None,
    fallback_lines: list[OcrLine],
) -> str:
    h, w = image.shape[:2]
    if line_reader is None:
        text = ""
        for line in fallback_lines:
            if y0 - 5 <= line.cy <= y1 + 5 and line.x0 > q.start.x0 + 45:
                text = line.text
                break
        return _clean_option_text(text)

    left = max(0, q.start.x0 + 45)
    right = min(w, q.start.x0 + 880)
    if right <= left:
        return ""
    gray = cv2.cvtColor(image[max(0, y0 - 2):min(h, y1 + 2), left:right], cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 185)
    if len(xs) == 0:
        return ""
    x_min = left + int(xs.min())
    x_max = left + int(xs.max())
    y_top = max(0, y0 - 7)
    y_bottom = min(h, y1 + 8)
    attempts = [
        (max(0, x_min - 8), min(w, x_max + 12)),
        (max(0, x_min - 20), min(w, x_max + 35)),
        (left, min(w, left + 700)),
    ]
    best = ""
    for lx, rx in attempts:
        if rx - lx < 5:
            continue
        text = line_reader(image[y_top:y_bottom, lx:rx])
        text = _clean_option_text(text)
        if len(text) > len(best):
            best = text
    if not best:
        for line in fallback_lines:
            if y0 - 5 <= line.cy <= y1 + 5 and line.x0 > q.start.x0 + 45:
                best = _clean_option_text(line.text)
                if best:
                    break
    return best


def _option_selected(image: np.ndarray, q: ExamQuestion, center_y: int) -> bool:
    h, w = image.shape[:2]
    x0 = max(0, q.start.x0 + 2)
    x1 = min(w, q.start.x0 + 58)
    y0 = max(0, center_y - 18)
    y1 = min(h, center_y + 18)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return False
    b = patch[:, :, 0].astype(np.int16)
    g = patch[:, :, 1].astype(np.int16)
    r = patch[:, :, 2].astype(np.int16)
    blue = (b > 165) & (b > r + 55) & (g > r + 15) & (g < 235)
    return float(blue.mean()) > 0.10


def _build_exam_questions(
    image: np.ndarray,
    lines: list[OcrLine],
    width: int,
    height: int,
    line_reader: Callable[[np.ndarray], str] | None,
) -> list[ExamQuestion]:
    starts = _question_starts(lines, width, height)
    result: list[ExamQuestion] = []
    for i, q in enumerate(starts):
        next_y = starts[i + 1].start.y0 if i + 1 < len(starts) else height - 20
        bands = _row_bands(image, q, next_y, width)
        options: list[QuizOption] = []
        option_idx = 0
        for b0, b1 in bands:
            if b0 < q.start.y1 + 25:
                continue
            band_h = b1 - b0
            if band_h < 10 or band_h > 58:
                continue
            if q.kind == "判断题" and option_idx >= 2:
                break
            if q.kind != "判断题" and option_idx >= 6:
                break
            text = "" if q.kind == "判断题" else _read_band_text(
                image, q, b0, b1, line_reader, lines
            )
            if not text and q.kind == "判断题":
                text = "对" if option_idx == 0 else "错"
            if not text or any(
                k in compact_text(text)
                for k in ("暂时保存", "提交", "题量", "满分", "ai讲解", "上一节", "下一节")
            ):
                continue
            # 题型小标题（如“二、多选题（共1题）”）不是选项，不能占 E 位。
            if re.match(
                r"^[一二三四五六七八九十]+[、.．]?\s*(单选题|多选题|判断题|填空题|简答题)",
                text,
            ):
                # 遇到下一节题型标题，当前题选项到此为止，避免把下一题选项算进来。
                break
            if (
                any(k in text for k in ("单选题", "多选题", "判断题"))
                and "共" in text
                and "题" in text
            ):
                break
            letter = chr(ord("A") + option_idx)
            line = OcrLine(
                text=text,
                score=1.0,
                x0=q.start.x0 + 45,
                y0=b0,
                x1=min(width - 2, q.start.x0 + 650),
                y1=b1,
            )
            selected = _option_selected(image, q, (b0 + b1) // 2)
            options.append(QuizOption(letter, text, line, selected=selected))
            option_idx += 1
        if len(options) >= 2:
            q.options = options
            last_bottom = max(o.line.y1 for o in options)
            if last_bottom > height - 55:
                q.complete = False
            if q.kind in ("单选题", "多选题") and len(options) < 4 and last_bottom > height - 130:
                q.complete = False
            result.append(q)
    return result


def _find_confirm_submit(lines: list[OcrLine]) -> tuple[OcrLine | None, OcrLine | None, bool]:
    confirm_line = None
    for line in lines:
        if "确认提交" in compact_text(line.text):
            confirm_line = line
            break
    if confirm_line is None:
        return None, None, False
    submits = [l for l in lines if compact_text(l.text) == "提交"]
    cancels = [l for l in lines if compact_text(l.text) == "取消"]
    cancel = None
    if cancels:
        cancel = min(
            cancels,
            key=lambda l: (l.cx - confirm_line.cx) ** 2 + (l.cy - confirm_line.cy) ** 2,
        )
    submit = None
    if submits:
        # 优先选和“取消”同一行的弹窗按钮；页面底部的另一个“提交”通常离得很远
        if cancel is not None:
            same_row = [l for l in submits if abs(l.cy - cancel.cy) <= 35]
            if same_row:
                submit = min(same_row, key=lambda l: abs(l.cx - cancel.cx))
        if submit is None:
            below = [l for l in submits if 0 <= l.cy - confirm_line.cy <= 220]
            if below:
                submit = min(below, key=lambda l: abs(l.cy - confirm_line.cy))
        if submit is None:
            submit = min(
                submits,
                key=lambda l: (l.cx - confirm_line.cx) ** 2 + (l.cy - confirm_line.cy) ** 2,
            )
    return submit, cancel, True


def detect_exam_page(
    image: np.ndarray,
    lines: list[OcrLine],
    width: int,
    height: int,
    has_time: bool,
    line_reader: Callable[[np.ndarray], str] | None = None,
) -> ExamPage:
    exam = ExamPage()
    texts = [normalize_text(l.text) for l in lines]
    compact_all = "\n".join(compact_text(t) for t in texts)
    exam.quiz_tab = find_quiz_tab(lines, width, height)

    m = TOTAL_RE.search(normalize_text(" ".join(texts)))
    if m:
        try:
            exam.total = int(m.group(1))
            if exam.total <= 0:
                exam.total = None
        except ValueError:
            exam.total = None

    confirm_submit, cancel, confirm = _find_confirm_submit(lines)
    exam.confirm = confirm
    exam.confirm_submit = confirm_submit
    exam.cancel = cancel

    next_button = find_next_button(lines)
    exam.next_button = next_button

    has_result = any("我的答案" in compact_text(t) for t in texts)
    has_score = any(re.search(r"\d+(?:\.\d+)?\s*分", t) for t in texts)
    exam.result = bool(has_result and (next_button or has_score))

    alerts = [t for t in texts if any(k in t for k in ("请完成所有题目", "还有题目未作答", "请答完", "未作答"))]
    exam.alert = " ".join(alerts[:3])

    if exam.result or exam.confirm:
        exam.detected = True
        return exam

    questions = _build_exam_questions(image, lines, width, height, line_reader)
    exam.questions = questions
    exam.submit = find_button(lines, ["提交", "提交试卷", "交卷"], exact=True)
    if questions:
        if len(questions) >= 2:
            exam.detected = True
        elif not has_time and (exam.submit or exam.total or exam.quiz_tab):
            exam.detected = True
        elif exam.total is not None:
            exam.detected = True
    if not exam.detected and not has_time and exam.submit:
        # 只有存在“题量”或解析出了带编号的题目时才按整页测验处理。
        # 视频上的单题弹窗也有“提交”和“章节测验”标签，但它没有题量/编号，
        # 必须交给 detect_quiz 直接作答，不能进入会滚动的整页测验流程。
        if exam.total is not None or questions:
            exam.detected = True
    if not exam.detected and not has_time and exam.total is not None:
        # 有些章节测验页先加载出“题量:5/满分”，题目和提交按钮稍后才渲染；
        # 只要确认有题量，就按整页测验进入，由 _handle_exam 继续滚动识别。
        exam.detected = True
    return exam


def _extract_options(lines: list[OcrLine], kind: str) -> tuple[list[QuizOption], bool]:
    options: dict[str, QuizOption] = {}
    standalone: dict[str, OcrLine] = {}
    for line in lines:
        text = normalize_text(line.text)
        m = OPTION_RE.match(text) or OPTION_SPACE_RE.match(text)
        if m:
            letter = m.group(1).upper()
            body = m.group(2).strip(" .．、:：")
            if body and len(body) <= 500:
                old = options.get(letter)
                if old is None or len(body) > len(old.text):
                    options[letter] = QuizOption(letter, body, line)
            else:
                standalone[letter] = line
            continue
        m2 = OPTION_LETTER_RE.match(text)
        if m2:
            standalone[m2.group(1).upper()] = line

    # 有些页面把 A/B/C/D 单独识别成一个框，合并同一行右侧的文本
    for letter, letter_line in standalone.items():
        if letter in options:
            continue
        best_line = None
        best_dist = 10_000
        for line in lines:
            if line is letter_line:
                continue
            dy = abs(line.cy - letter_line.cy)
            # OCR 常把“A、”和“心理”拆成两个紧挨着的框，
            # 两者的 x 还可能重叠几像素，不能简单要求 x0 > x1。
            if dy > max(42, int(letter_line.height * 2.2)):
                continue
            if line.x0 < letter_line.x1 - 15:
                continue
            dist = max(0, line.x0 - letter_line.x1) + dy * 3
            if dist < best_dist and dist < 500:
                best_dist, best_line = dist, line
        if best_line is not None:
            options[letter] = QuizOption(letter, normalize_text(best_line.text), letter_line)

    # 少数播放器把复选框和选项文字分开渲染：例如字母 C 只在紫色方框里，
    # OCR 只读到“法律”。按行间距和字母顺序把这类“无字母的文字行”补回缺失选项。
    if 2 <= len(options) + len(standalone) <= 8:
        used = {id(o.line) for o in options.values()} | {
            id(line) for line in standalone.values()
        }
        known = {letter: line.cy for letter, line in standalone.items()}
        known.update({letter: opt.line.cy for letter, opt in options.items()})
        ordered_known = sorted(known.values())
        gaps = [
            ordered_known[i + 1] - ordered_known[i]
            for i in range(len(ordered_known) - 1)
            if ordered_known[i + 1] - ordered_known[i] > 20
        ]
        spacing = sorted(gaps)[len(gaps) // 2] if gaps else 90
        min_x = min((opt.line.x0 for opt in options.values()), default=0)
        min_x = min([min_x] + [line.x0 for line in standalone.values()])
        max_x = max((opt.line.x1 for opt in options.values()), default=0)
        for letter in "ABCDEFGH":
            if letter in options or letter in standalone:
                continue
            lower = [known[k] for k in known if k < letter]
            upper = [known[k] for k in known if k > letter]
            if lower and upper:
                expected = (max(lower) + min(upper)) / 2
            elif lower:
                expected = max(lower) + spacing
            elif upper:
                expected = min(upper) - spacing
            else:
                continue
            best_line = None
            best_dist = 999.0
            for line in lines:
                if id(line) in used:
                    continue
                if abs(line.cy - expected) > 30:
                    continue
                if line.x0 < min_x - 10 or line.x1 > max_x + 120:
                    continue
                text = normalize_text(line.text)
                if not text or len(text) > 80:
                    continue
                if any(k in text for k in ("提交", "回答", "下一", "上一", "知识点")):
                    continue
                dist = abs(line.cy - expected) + max(0, line.x0 - (min_x + 300))
                if dist < best_dist:
                    best_dist = dist
                    best_line = line
            if best_line is not None:
                options[letter] = QuizOption(
                    letter, normalize_text(best_line.text), best_line
                )
                used.add(id(best_line))
                known[letter] = best_line.cy

    if len(options) < 2 and kind == "判断题":
        right = find_button(lines, ["正确", "对", "√"], exact=False)
        wrong = find_button(lines, ["错误", "错", "×"], exact=False)
        if right and wrong:
            options = {
                "A": QuizOption("A", "正确", right),
                "B": QuizOption("B", "错误", wrong),
            }

    ordered = sorted(options.values(), key=lambda o: (o.line.cy, o.line.cx))
    letters = [o.letter for o in ordered]
    multiple_sets = len(set(letters)) != len(letters)
    return ordered, multiple_sets


def _quiz_kind(text: str) -> str:
    for kind in ("多选题", "单选题", "判断题", "填空题", "简答题", "不定项选择题"):
        if kind in text:
            return "多选题" if kind == "不定项选择题" else kind
    return "选择题"


def detect_quiz(lines: list[OcrLine], width: int, height: int) -> QuizView:
    compact_all = "\n".join(compact_text(l.text) for l in lines)
    has_marker = any(
        k in compact_all
        for k in ("单选题", "多选题", "判断题", "填空题", "简答题", "不定项", "开始答题", "答题卡")
    ) or bool(QUESTION_NO_RE.search(compact_all))
    submit = find_button(
        lines,
        ["提交答案", "提交试卷", "确认提交", "保存并提交", "提交", "交卷"],
        exact=True,
    )
    feedback_lines = [
        l
        for l in lines
        if any(
            k in compact_text(l.text)
            for k in (
                "回答正确",
                "答案正确",
                "答对了",
                "恭喜",
                "回答错误",
                "答案错误",
                "答错了",
                "不正确",
                "正确答案",
                "你的答案",
                "答案解析",
            )
        )
    ]
    feedback = " ".join(l.text for l in feedback_lines[:6])

    # 结果弹窗：提交后出现对错提示和“下一题/关闭”
    if feedback_lines and not submit:
        nb = find_button(lines, ["下一题", "继续答题", "继续", "关闭", "完成", "确定"], exact=False)
        if nb:
            return QuizView(
                detected=True,
                kind="结果",
                question="",
                feedback=feedback,
                next_button=nb,
                is_result=True,
                signature=hashlib.md5(feedback.encode("utf-8", "ignore")).hexdigest()[:12],
            )

    options, multiple_sets = _extract_options(lines, _quiz_kind(compact_all))
    if len(options) < 2:
        # 没有选项但存在明确题面 + 提交按钮，也可进入作答流程（填空题由人工处理）
        if submit and has_marker:
            question = _extract_question(lines, None, width, height)
            return QuizView(
                detected=True,
                kind=_quiz_kind(question + compact_all),
                question=question,
                submit=submit,
                multiple_sets=False,
                signature=_quiz_signature(question, []),
            )
        return QuizView()

    if not (submit or has_marker):
        return QuizView()
    # 只有选项、没有提交/题型标记的，多半是视频字幕，不做处理
    if not submit and not has_marker:
        return QuizView()

    kind = _quiz_kind(compact_all + " " + " ".join(o.text for o in options))
    question = _extract_question(lines, options, width, height)
    if not question and not has_marker:
        return QuizView()
    return QuizView(
        detected=True,
        kind=kind,
        question=question,
        options=options,
        submit=submit,
        multiple_sets=multiple_sets,
        signature=_quiz_signature(question, options),
    )


def _is_noise_line(text: str) -> bool:
    c = compact_text(text)
    if not c:
        return True
    if c in {compact_text(x) for x in UI_NOISE}:
        return True
    if parse_section_line(text):
        return True
    if TIME_RE.search(text):
        return True
    return False


def _extract_question(
    lines: list[OcrLine], options: list[QuizOption] | None, width: int, height: int
) -> str:
    if options:
        first_y = min(o.line.y0 for o in options)
        max_x = max(o.line.x1 for o in options)
        min_x = min(o.line.x0 for o in options)
        score_bottom = max(o.line.y1 for o in options)
    else:
        first_y = height
        max_x = width * 0.75
        min_x = width * 0.10
        score_bottom = height

    # 题面一般位于第一个选项上方；限制在合理区域，避免把视频字幕、侧栏带进来
    top = max(0, first_y - 520)
    left = max(0, min_x - 280)
    right = min(width, max_x + 300)
    region = []
    option_ids = {id(o.line) for o in options or []}
    for line in lines:
        if id(line) in option_ids:
            continue
        if not (top <= line.y0 <= first_y - 2):
            continue
        if not (left <= line.x0 <= right):
            continue
        if _is_noise_line(line.text):
            continue
        if line.x0 > width * 0.62:
            continue
        region.append(line)
    region.sort(key=lambda l: (l.y0, l.x0))
    if not region:
        # 有些弹窗题面与选项同一行，退化为取选项附近文本
        for line in lines:
            if id(line) in option_ids:
                continue
            if abs(line.cy - first_y) < 90 and min_x - 200 < line.x0 < max_x + 200:
                if not _is_noise_line(line.text):
                    region.append(line)
        region.sort(key=lambda l: (l.y0, l.x0))
    # 单题弹窗的题面通常以“判断题/单选题/多选题”开头；以此为界，
    # 排除上方页面标题、视频标签、完成条件等无关文字。
    marker_top = None
    for line in region:
        c = compact_text(line.text)
        if any(k in c for k in ("判断题", "单选题", "多选题", "填空题", "简答题")) and len(c) <= 8:
            marker_top = line.y0 if marker_top is None else min(marker_top, line.y0)
    if marker_top is not None:
        region = [l for l in region if l.y0 >= marker_top - 2]
    region = [
        l
        for l in region
        if "观看时长" not in l.text
        and "完成条件" not in l.text
        and compact_text(l.text) not in {"1视频", "2章节测验", "章节测验"}
    ]
    text = "\n".join(l.text for l in region[-30:]).strip()
    return text[:3500]


def _quiz_signature(question: str, options: list[QuizOption]) -> str:
    payload = question + "|" + "|".join(f"{o.letter}:{o.text}" for o in options)
    return hashlib.md5(payload.encode("utf-8", "ignore")).hexdigest()[:12]


def analyze_page(
    image: np.ndarray,
    lines: list[OcrLine],
    previous_section: str | None = None,
    line_reader: Callable[[np.ndarray], str] | None = None,
    preferred_time: OcrLine | None = None,
) -> PageView:
    h, w = image.shape[:2]
    view = PageView(width=w, height=h, lines=lines)
    view.times = [reading for reading in find_time_readings(lines) if reading.line.x0 < w * 0.72]
    view.videos = find_video_tasks(image, lines, view.times)
    view.active_video = select_video(view.videos, preferred_time)
    if view.active_video:
        view.time = view.active_video.time
        view.task_line = view.active_video.label
        view.task_completed = view.active_video.completed
    view.sections = find_sections(lines, w, h)
    if view.sections:
        mark_completed_sections(image, view.sections)
    view.page_title_line, title_score, title_idx = find_page_title(lines, w, view.sections)
    if view.page_title_line:
        view.page_title = view.page_title_line.text
    view.current_idx, view.current_evidence = find_current_section(
        image, view.sections, view.page_title_line, title_idx, title_score, previous_section
    )
    view.exam = detect_exam_page(
        image, lines, w, h, has_time=view.time is not None, line_reader=line_reader
    )
    if not view.exam.detected:
        view.quiz = detect_quiz(lines, w, h)
    view.next_button = find_next_button(lines)
    view.resume_button = find_resume_button(lines)
    view.start_button = find_start_button(lines)
    view.quiz_entry = find_quiz_entry(lines)
    view.completion_line = find_completion_line(lines)
    return view


def section_for_key(sections: list[SectionRow], key: str | None) -> int | None:
    if not key:
        return None
    for i, row in enumerate(sections):
        if row.key == key:
            return i
    return None


def quiz_answer_letters(text: str) -> list[str]:
    letters = re.findall(r"(?<![A-Z])[A-H](?![A-Z])", text.upper())
    result: list[str] = []
    for letter in letters:
        if letter not in result:
            result.append(letter)
    return result


def annotate_frame(
    image: np.ndarray,
    view: PageView | None,
    status: str = "",
    current_key: str | None = None,
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    if view:
        for index, video in enumerate(view.videos, 1):
            x0, y0, x1, y1 = video.bounds
            color = (60, 220, 120) if video is view.active_video else (220, 160, 60)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
            cv2.putText(canvas, f"VIDEO {index}", (x0 + 5, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if view.time:
            x0, y0, x1, y1 = view.time.line.expanded(8)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 220, 120), 3)
            cv2.putText(
                canvas,
                f"{view.time.current_text()}/{view.time.duration_text()}",
                (max(5, x0), max(24, y0 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (60, 220, 120),
                2,
                cv2.LINE_AA,
            )
        for idx, row in enumerate(view.sections):
            is_current = idx == view.current_idx or (
                current_key is not None and row.key == current_key
            )
            if is_current:
                color = (40, 170, 255)
            elif row.completed:
                color = (60, 190, 90)
            else:
                color = (140, 140, 90)
            thickness = 3 if is_current else 1
            x0, y0, x1, y1 = row.line.expanded(6)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, thickness)
            if is_current:
                cv2.putText(
                    canvas,
                    row.number,
                    (max(5, x0 - 90), y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        quiz = view.quiz
        if quiz.detected:
            for opt in quiz.options:
                x0, y0, x1, y1 = opt.line.expanded(6)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (235, 90, 210), 3)
                cv2.putText(
                    canvas,
                    opt.letter,
                    (max(5, x0 - 35), opt.line.cy + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (235, 90, 210),
                    2,
                    cv2.LINE_AA,
                )
            if quiz.submit:
                x0, y0, x1, y1 = quiz.submit.expanded(8)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 70, 245), 3)
        if view.exam.detected:
            for q in view.exam.questions:
                x0, y0, x1, y1 = q.start.expanded(4)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (190, 90, 250), 2)
                for opt in q.options:
                    ox0, oy0, ox1, oy1 = opt.line.expanded(4)
                    color = (40, 190, 255) if opt.selected else (200, 100, 220)
                    cv2.rectangle(canvas, (ox0, oy0), (ox1, oy1), color, 2)
            if view.exam.submit:
                x0, y0, x1, y1 = view.exam.submit.expanded(8)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (40, 70, 245), 3)
            if view.exam.quiz_tab:
                x0, y0, x1, y1 = view.exam.quiz_tab.expanded(6)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 210, 255), 2)
        if view.next_button:
            x0, y0, x1, y1 = view.next_button.expanded(8)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 210, 255), 3)

    if status:
        band = canvas[0:52, 0:w].copy()
        cv2.rectangle(band, (0, 0), (w, 52), (18, 22, 30), -1)
        cv2.addWeighted(band, 0.82, canvas[0:52, 0:w], 0.18, 0, canvas[0:52, 0:w])
        cv2.putText(
            canvas,
            status[:150],
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.86,
            (235, 242, 250),
            2,
            cv2.LINE_AA,
        )
    return canvas


def preview_jpeg(image: np.ndarray, max_width: int = 900, quality: int = 82) -> bytes:
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def ocr_debug_text(view: PageView) -> str:
    rows = [
        f"window={view.width}x{view.height}",
        f"time={view.time.raw if view.time else '-'}",
        f"visible_videos={len(view.videos)} times={len(view.times)}",
        f"title={view.page_title!r}",
        f"current_idx={view.current_idx} evidence={view.current_evidence}",
        f"completion={'yes' if view.completion_line else 'no'}",
        f"task_point={'completed' if view.task_completed else 'pending' if view.task_completed is False else 'unknown'}",
        f"quiz={view.quiz.detected} kind={view.quiz.kind} options={len(view.quiz.options)}",
        f"exam={view.exam.detected} questions={len(view.exam.questions)} confirm={view.exam.confirm} result={view.exam.result} total={view.exam.total}",
        f"quiz_tab={view.exam.quiz_tab.text if view.exam.quiz_tab else '-'}",
        f"next_button={view.next_button.text if view.next_button else '-'}",
    ]
    for index, video in enumerate(view.videos, 1):
        rows.append(f"video[{index}] bounds={video.bounds} completed={video.completed} time={video.time.raw if video.time else '-'} active={video is view.active_video}")
    rows.append("sections:")
    for i, row in enumerate(view.sections):
        if row.completed:
            status = "✓"
        elif row.pending_tasks is not None:
            status = str(row.pending_tasks)
        else:
            status = " "
        marks = ("*" if i == view.current_idx else " ") + status
        rows.append(f"  {i:02d} {marks} {row.number} {row.title}")
    if view.quiz.detected:
        rows.append("question:")
        rows.append(view.quiz.question)
        for opt in view.quiz.options:
            rows.append(f"  {opt.letter}. {opt.text}")
        rows.append(f"submit={view.quiz.submit.text if view.quiz.submit else '-'}")
    if view.exam.detected:
        rows.append("exam:")
        for q in view.exam.questions:
            rows.append(f"  {q.number} [{q.kind}] complete={q.complete} {q.text}")
            for opt in q.options:
                rows.append(f"    {opt.letter}. {opt.text} selected={opt.selected}")
        rows.append(f"  exam_submit={view.exam.submit.text if view.exam.submit else '-'}")
    return "\n".join(rows)
