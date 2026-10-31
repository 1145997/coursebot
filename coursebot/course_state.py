"""整页课程状态与决策；不访问桌面，不根据橙色数字推断任务类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(frozen=True)
class CourseVideo:
    key: str
    order: int
    completed: bool | None
    current: float | None = None
    duration: float | None = None
    paused: bool | None = None
    play_token: str = ""
    error_text: str = ""


@dataclass(frozen=True)
class PendingChapter:
    key: str
    title: str
    token: str


@dataclass
class CourseSnapshot:
    page_key: str
    videos: list[CourseVideo] = field(default_factory=list)
    pending: list[PendingChapter] = field(default_factory=list)
    complete: bool = True
    problem: str = ""
    video_tab_token: str = ""
    quiz_tab_token: str = ""
    popup_quiz: bool = False
    video_tab_selected: bool | None = None

    def inventory_text(self) -> str:
        done = [str(v.order) for v in self.videos if v.completed is True]
        pending = [str(v.order) for v in self.videos if v.completed is False]
        unknown = [str(v.order) for v in self.videos if v.completed is None]
        return (f"整页视频 {len(self.videos)} 个；已完成：{','.join(done) or '无'}；"
                f"未完成：{','.join(pending) or '无'}；状态未知：{','.join(unknown) or '无'}")


class Action(Enum):
    WAIT = "wait"
    PLAY = "play"
    OPEN_VIDEO_TAB = "open_video_tab"
    PAGE_DONE = "page_done"
    LEGACY_PAGE = "legacy_page"
    STOP = "stop"
    LOAD_ERROR = "load_error"
    PAUSE_COMPLETED = "pause_completed"


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    token: str = ""
    video_key: str = ""


@dataclass
class Progress:
    current: float | None = None
    advanced_at: float = 0.0
    observed_at: float = 0.0
    clicked_at: float | None = None
    attempts: int = 0


class CourseController:
    """每次重新观察后只产生一个动作；所有播放点击共用这一入口。"""

    def __init__(self):
        self.page_key = ""
        self.progress: dict[str, Progress] = {}
        self.page_seen_at = 0.0
        self.tab_clicked_at: float | None = None
        self.tab_attempts = 0
        self.pause_requests: dict[str, float] = {}

    def decide(self, snapshot: CourseSnapshot, now: float, auto_resume: bool = True) -> Decision:
        if not snapshot.complete or not snapshot.page_key:
            return Decision(Action.STOP, snapshot.problem or "整页清单不完整，不能确定视频总数")
        if snapshot.page_key != self.page_key:
            self.page_key = snapshot.page_key
            self.progress.clear()
            self.page_seen_at = now
            self.tab_clicked_at = None
            self.tab_attempts = 0
            self.pause_requests.clear()
        if not snapshot.videos:
            if snapshot.video_tab_token:
                if snapshot.video_tab_selected:
                    if now - self.page_seen_at > 30:
                        return Decision(Action.STOP, "视频标签已选中，但未能读取播放器")
                    return Decision(Action.WAIT, "视频标签已选中，等待播放器加载")
                if self.tab_clicked_at is not None and now - self.tab_clicked_at < 10:
                    return Decision(Action.WAIT, "等待视频标签切换生效")
                if self.tab_attempts >= 2:
                    return Decision(Action.STOP, "两次操作后视频标签仍未加载，停止重复点击")
                return Decision(Action.OPEN_VIDEO_TAB, "进入视频标签后重新盘点", snapshot.video_tab_token)
            return Decision(Action.LEGACY_PAGE, "当前不是视频页，交给测验/导读流程")
        if any(video.completed is None for video in snapshot.videos):
            return Decision(Action.STOP, "有播放器无法关联任务点状态，停止操作以免重播或漏播")
        pending = [video for video in snapshot.videos if video.completed is False]
        if not pending:
            # 任务点完成后不再把播放钮当成“继续”，禁止重播。
            return Decision(Action.PAGE_DONE, "整页所有视频任务点已完成")
        completed_playing = next((v for v in snapshot.videos if v.completed is True and v.paused is False), None)
        if completed_playing:
            if completed_playing.key in self.pause_requests:
                if now - self.pause_requests[completed_playing.key] > 5:
                    return Decision(Action.STOP, "已完成视频的暂停操作未生效，停止重复切换")
                return Decision(Action.WAIT, "等待已完成的视频暂停")
            if not completed_playing.play_token:
                return Decision(Action.STOP, "已完成的视频仍在播放，但没有确认到暂停控件")
            return Decision(Action.PAUSE_COMPLETED, "暂停仍在播放的已完成项，再处理未完成项",
                            completed_playing.play_token, completed_playing.key)
        # 用户已经启动了后面某段时，先监控它，不再启动另一段造成并行播放。
        video = next((v for v in pending if v.paused is False), pending[0])
        p = self.progress.setdefault(video.key, Progress(advanced_at=now, observed_at=now))
        advanced = (video.current is not None and video.current > 0 and
                    (p.current is None or video.current > p.current + 0.2))
        if advanced:
            p.advanced_at = now
            p.current = video.current
            p.attempts = 0
        if p.current is None and video.current is not None:
            p.current = video.current
        p.observed_at = now
        if video.error_text:
            return Decision(Action.LOAD_ERROR, video.error_text, video_key=video.key)
        if video.paused is False or advanced:
            return Decision(Action.WAIT, f"监控第 {video.order} 个未完成视频", video_key=video.key)
        if video.current is not None and video.duration and video.current >= video.duration - 1:
            if now - p.advanced_at > 45:
                return Decision(Action.STOP, "视频已到结尾，但平台任务点仍未完成，停止跳节")
            return Decision(Action.WAIT, "等待平台回写任务点完成状态", video_key=video.key)
        if not auto_resume:
            return Decision(Action.WAIT, "自动恢复播放已关闭", video_key=video.key)
        if not video.play_token or video.paused is not True:
            if now - p.advanced_at > 45:
                return Decision(Action.STOP, "未能确认待播视频的播放按钮或播放状态")
            return Decision(Action.WAIT, "等待播放器控件加载", video_key=video.key)
        # 新视频零进度/未加载可立即开始；中途暂停必须先确认时间停止。
        if video.current and now - p.advanced_at < 8:
            return Decision(Action.WAIT, "复核时间是否继续增长", video_key=video.key)
        if p.clicked_at is not None and now - p.clicked_at < 15:
            return Decision(Action.WAIT, "等待上一次播放操作生效", video_key=video.key)
        if p.attempts >= 2:
            return Decision(Action.STOP, "两次播放操作后仍未确认进度，停止继续点击")
        return Decision(Action.PLAY, f"播放第 {video.order} 个未完成视频", video.play_token, video.key)

    def acted(self, decision: Decision, now: float) -> None:
        """只有实际输入成功后才更新点击冷却，试运行不会改变状态。"""
        if decision.action is Action.PLAY:
            p = self.progress[decision.video_key]
            p.clicked_at = now
            p.attempts += 1
        elif decision.action is Action.OPEN_VIDEO_TAB:
            self.tab_clicked_at = now
            self.tab_attempts += 1
        elif decision.action is Action.PAUSE_COMPLETED:
            self.pause_requests[decision.video_key] = now
