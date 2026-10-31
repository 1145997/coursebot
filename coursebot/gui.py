from __future__ import annotations

import io
import json
import os
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from . import __version__
from .automation import AutomationWorker
from .accessibility import BrowserAccessibility
from .capture import ScreenCapture
from .config import CONFIG_PATH, LOG_DIR, Settings, masked_key
from .llm import LLMClient
from .ocr import TextOCR
from .vision import analyze_page, annotate_frame, ocr_debug_text, preview_jpeg
from .windows import (
    Clicker,
    HotkeyListener,
    MouseInputError,
    WindowInfo,
    focus_window,
    get_window_info,
    list_windows,
)

# 配色
BG = "#0b1220"
PANEL = "#101a2b"
CARD = "#0f172a"
CARD_ALT = "#121f35"
BORDER = "#23324d"
TEXT = "#e7eef9"
MUTED = "#8fa4c2"
ACCENT = "#22c55e"
ACCENT_HOVER = "#34d399"
BLUE = "#38bdf8"
WARN = "#f59e0b"
DANGER = "#ef4444"
PURPLE = "#a78bfa"
INPUT_BG = "#091221"

STATE_COLORS = {
    "starting": BLUE,
    "scanning": BLUE,
    "playing": ACCENT,
    "paused": WARN,
    "quiz": PURPLE,
    "exam": PURPLE,
    "navigating": BLUE,
    "completed": ACCENT,
    "finished": ACCENT,
    "waiting_focus": WARN,
    "error": DANGER,
    "idle": MUTED,
}


class CourseBotApp:
    def __init__(self, root: tk.Tk, settings: Settings):
        self.root = root
        self.settings = settings
        self.running = False
        self.worker: AutomationWorker | None = None
        self.stop_event = threading.Event()
        self.queue: queue.Queue = queue.Queue()
        self.hotkey: HotkeyListener | None = None
        self.window_infos: list[WindowInfo] = []
        self.selected_window: WindowInfo | None = None
        self.queue_ok = True
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._preview_image: Image.Image | None = None
        self._preview_ocr: TextOCR | None = None
        self._scan_busy = False
        self._last_error_reason = ""

        self.dpi = max(96, int(root.winfo_fpixels("1i")))
        self.s = max(1.0, self.dpi / 96.0)
        root.tk.call("tk", "scaling", self.dpi / 72.0)
        self._setup_fonts()
        self._setup_window()
        self._setup_style()
        self._build_vars()
        self._build_ui()
        self._populate_settings()
        self.refresh_windows(initial=True)
        self._apply_always_on_top()
        self._append_log("info", "欢迎使用网课自动学习助手。OpenCV 截图 + RapidOCR 识别 + LLM 答题。")
        self._append_log("info", f"按 F8 可随时紧急停止；配置保存在 {CONFIG_PATH}")
        self._append_log("info", "操作顺序：选择浏览器窗口 → 扫描预览确认识别 → 开始自动学习。")
        self.root.after(150, self._poll_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._setup_hotkey()

    # ------------------------------------------------------------------ 基础
    def px(self, value: float) -> int:
        return max(1, int(round(value * self.s)))

    def _setup_fonts(self) -> None:
        family = "Microsoft YaHei UI"
        self.font_title = tkfont.Font(family=family, size=17, weight="bold")
        self.font_sub = tkfont.Font(family=family, size=9)
        self.font_h1 = tkfont.Font(family=family, size=11, weight="bold")
        self.font_body = tkfont.Font(family=family, size=10)
        self.font_body_bold = tkfont.Font(family=family, size=10, weight="bold")
        self.font_small = tkfont.Font(family=family, size=9)
        self.font_tiny = tkfont.Font(family=family, size=8)
        self.font_button = tkfont.Font(family=family, size=10, weight="bold")
        self.font_time = tkfont.Font(family="Cascadia Mono", size=10, weight="bold")
        self.font_log = tkfont.Font(family="Cascadia Mono", size=9)
        self.font_stat = tkfont.Font(family="Cascadia Mono", size=11, weight="bold")

    def _setup_window(self) -> None:
        self.root.title(f"网课自动学习助手 · CourseBot v{__version__}")
        self.root.configure(bg=BG)
        w, h = self.px(1250), self.px(830)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2 - self.px(12))
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(self.px(1080), self.px(720))
        try:
            icon = tk.PhotoImage(width=32, height=32)
            icon.put("#22c55e", to=(0, 0, 32, 32))
            icon.put("#0b1220", to=(6, 6, 26, 26))
            icon.put("#22c55e", to=(12, 12, 20, 20))
            self.root.iconphoto(True, icon)
            self._icon = icon
        except Exception:
            pass

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=BG, foreground=TEXT, font=self.font_body)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Card.TLabel", background=CARD, foreground=TEXT)
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=self.font_small)
        style.configure("TNotebook", background=CARD, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "TNotebook.Tab",
            background=PANEL,
            foreground=MUTED,
            padding=(self.px(13), self.px(8)),
            font=self.font_body,
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", CARD), ("active", CARD_ALT)],
            foreground=[("selected", ACCENT), ("active", TEXT)],
        )
        style.configure(
            "TCombobox",
            fieldbackground=INPUT_BG,
            background=PANEL,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=self.px(6),
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", INPUT_BG), ("disabled", CARD)],
            foreground=[("readonly", TEXT), ("disabled", MUTED)],
            bordercolor=[("focus", ACCENT), ("hover", BLUE)],
        )
        style.configure(
            "Horizontal.TProgressbar",
            background=ACCENT,
            troughcolor="#1b2b44",
            bordercolor="#1b2b44",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=self.px(8),
        )
        style.configure("Vertical.TScrollbar", background=PANEL, troughcolor=CARD, arrowcolor=MUTED, borderwidth=0)
        style.configure(
            "TSpinbox",
            fieldbackground=INPUT_BG,
            background=PANEL,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=BORDER,
        )

    def _build_vars(self) -> None:
        s = self.settings
        self.var_window = tk.StringVar()
        self.var_auto_next = tk.BooleanVar(value=s.auto_next)
        self.var_auto_quiz = tk.BooleanVar(value=s.auto_quiz)
        self.var_auto_resume = tk.BooleanVar(value=s.auto_resume)
        self.var_dry_run = tk.BooleanVar(value=s.dry_run)
        self.var_minimize = tk.BooleanVar(value=s.minimize_on_start)
        self.var_topmost = tk.BooleanVar(value=s.always_on_top)
        self.var_preview = tk.BooleanVar(value=s.preview_enabled)
        self.var_debug = tk.BooleanVar(value=s.save_debug_frames)
        self.var_max_sections = tk.StringVar(value=str(s.max_sections))
        self.var_stall = tk.StringVar(value=f"{s.stall_seconds:g}")
        self.var_margin = tk.StringVar(value=f"{s.completion_margin:g}")
        self.var_poll = tk.StringVar(value=f"{s.poll_interval:g}")
        self.var_full_scan = tk.StringVar(value=f"{s.full_scan_interval:g}")
        self.var_quiz_retry = tk.StringVar(value=str(s.quiz_retry))
        self.var_llm_mode = tk.StringVar(value=s.llm_mode)
        self.var_api_url = tk.StringVar(value=s.api_url)
        self.var_api_key = tk.StringVar(value=s.api_key)
        self.var_model = tk.StringVar(value=s.model)
        self.var_local_url = tk.StringVar(value=s.local_api_url)
        self.var_local_model = tk.StringVar(value=s.local_model)
        self.var_local_timeout = tk.StringVar(value=str(s.local_timeout))
        self.var_timeout = tk.StringVar(value=str(s.llm_timeout))
        self.var_max_tokens = tk.StringVar(value=str(s.max_tokens))
        self.var_temperature = tk.StringVar(value=f"{s.temperature:g}")
        self.var_thinking = tk.BooleanVar(value=s.enable_thinking)
        self.show_key = tk.BooleanVar(value=False)

    # ------------------------------------------------------------------ 组件
    def card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )

    def label(
        self,
        parent: tk.Widget,
        text: str = "",
        font=None,
        fg: str = TEXT,
        bg: str = CARD,
        **kwargs,
    ) -> tk.Label:
        return tk.Label(parent, text=text, font=font or self.font_body, fg=fg, bg=bg, **kwargs)

    def button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        kind: str = "primary",
        width: int | None = None,
        state: str = "normal",
        font=None,
    ) -> tk.Button:
        palette = {
            "primary": (ACCENT, "#07130c", ACCENT_HOVER),
            "secondary": (CARD_ALT, TEXT, "#1b2c47"),
            "danger": ("#3a1720", "#fecaca", "#4c1d2b"),
            "ghost": (PANEL, MUTED, CARD_ALT),
        }
        bg, fg, hover = palette.get(kind, palette["secondary"])
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=font or self.font_button,
            bg=bg,
            fg=fg,
            activebackground=hover,
            activeforeground=fg,
            disabledforeground="#6c7f99",
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=self.px(16),
            pady=self.px(9),
            cursor="hand2" if state == "normal" else "arrow",
            state=state,
        )
        btn._palette = (bg, fg, hover)  # type: ignore[attr-defined]
        if width:
            btn.configure(width=width)
        if state == "normal":
            btn.bind("<Enter>", lambda _e, b=btn, c=hover: b.configure(bg=c) if b["state"] == "normal" else None)
            btn.bind("<Leave>", lambda _e, b=btn, c=bg: b.configure(bg=c) if b["state"] == "normal" else None)
        return btn

    def set_button_enabled(self, btn: tk.Button, enabled: bool) -> None:
        bg, _fg, _hover = getattr(btn, "_palette", (CARD_ALT, TEXT, CARD_ALT))
        btn.configure(state="normal" if enabled else "disabled", cursor="hand2" if enabled else "arrow")
        if not enabled:
            btn.configure(bg=bg)
        else:
            btn.configure(bg=bg)

    def _build_ui(self) -> None:
        pad = self.px(14)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        # 顶栏
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=pad, pady=(self.px(12), self.px(4)))
        header.grid_columnconfigure(1, weight=1)
        accent_bar = tk.Frame(header, bg=ACCENT, width=self.px(4), height=self.px(46))
        accent_bar.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, self.px(12)))
        accent_bar.grid_propagate(False)
        self.label(header, "网课自动学习助手", font=self.font_title, bg=BG).grid(
            row=0, column=1, sticky="w"
        )
        self.label(
            header,
            "OpenCV 截图定位  ·  RapidOCR 识别  ·  LLM 答题  ·  自动进入下一节",
            font=self.font_sub,
            fg=MUTED,
            bg=BG,
        ).grid(row=1, column=1, sticky="w")
        status_wrap = tk.Frame(header, bg=BG)
        status_wrap.grid(row=0, column=2, rowspan=2, sticky="e")
        self.status_dot = self.label(status_wrap, "●", font=self.font_body, fg=MUTED, bg=BG)
        self.status_dot.pack(side="left", padx=(0, self.px(6)))
        self.header_status = self.label(status_wrap, "待机", font=self.font_body_bold, fg=TEXT, bg=BG)
        self.header_status.pack(side="left")

        # 目标窗口工具栏
        toolbar = self.card(self.root)
        toolbar.grid(row=1, column=0, sticky="ew", padx=pad, pady=(self.px(6), self.px(10)))
        toolbar.grid_columnconfigure(1, weight=1)
        self.label(toolbar, "目标窗口", font=self.font_small, fg=MUTED).grid(
            row=0, column=0, padx=(self.px(14), self.px(10)), pady=self.px(10)
        )
        self.window_combo = ttk.Combobox(
            toolbar,
            textvariable=self.var_window,
            state="readonly",
            font=self.font_body,
            height=self.px(20),
        )
        self.window_combo.grid(row=0, column=1, sticky="ew", pady=self.px(8))
        self.window_combo.bind("<<ComboboxSelected>>", self._on_window_selected)
        self.refresh_btn = self.button(toolbar, "刷新窗口", self.refresh_windows, "secondary", font=self.font_small)
        self.refresh_btn.grid(row=0, column=2, padx=(self.px(8), self.px(6)), pady=self.px(8))
        self.scan_btn = self.button(toolbar, "扫描预览", self.scan_preview, "secondary", font=self.font_small)
        self.scan_btn.grid(row=0, column=3, padx=(0, self.px(10)), pady=self.px(8))

        # 主体
        body = tk.Frame(self.root, bg=BG)
        body.grid(row=2, column=0, sticky="nsew", padx=pad)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=self.px(405), weight=0)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, self.px(10)))
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)

        preview_card = self.card(left)
        preview_card.grid(row=0, column=0, sticky="nsew")
        preview_card.grid_columnconfigure(0, weight=1)
        preview_card.grid_rowconfigure(1, weight=1)
        ph = tk.Frame(preview_card, bg=CARD)
        ph.grid(row=0, column=0, sticky="ew", padx=self.px(12), pady=(self.px(9), self.px(4)))
        ph.grid_columnconfigure(1, weight=1)
        self.label(ph, "实时识别预览", font=self.font_h1).grid(row=0, column=0, sticky="w")
        self.preview_hint = self.label(ph, "OCR 叠加框", font=self.font_tiny, fg=MUTED)
        self.preview_hint.grid(row=0, column=1, sticky="e")
        self.preview_canvas = tk.Canvas(
            preview_card,
            bg="#050a13",
            highlightthickness=0,
            bd=0,
            height=self.px(300),
        )
        self.preview_canvas.grid(row=1, column=0, sticky="nsew", padx=self.px(10), pady=(0, self.px(10)))
        self.preview_canvas.bind("<Configure>", lambda _e: self._render_preview())
        self._draw_preview_placeholder()

        # 状态卡
        status_card = self.card(left)
        status_card.grid(row=1, column=0, sticky="ew", pady=(self.px(10), 0))
        status_card.grid_columnconfigure(0, weight=1)
        row1 = tk.Frame(status_card, bg=CARD)
        row1.grid(row=0, column=0, sticky="ew", padx=self.px(14), pady=(self.px(10), self.px(4)))
        row1.grid_columnconfigure(0, weight=1)
        self.state_text = self.label(row1, "等待开始", font=self.font_h1)
        self.state_text.grid(row=0, column=0, sticky="w")
        self.time_text = self.label(row1, "--:-- / --:--", font=self.font_time, fg=MUTED)
        self.time_text.grid(row=0, column=1, sticky="e")
        self.progress = ttk.Progressbar(status_card, style="Horizontal.TProgressbar", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", padx=self.px(14), pady=(0, self.px(6)))
        row2 = tk.Frame(status_card, bg=CARD)
        row2.grid(row=2, column=0, sticky="ew", padx=self.px(14), pady=(0, self.px(10)))
        row2.grid_columnconfigure(0, weight=1)
        self.stats_text = self.label(row2, "本次完成 0 节 · 自动答题 0 题", font=self.font_small, fg=MUTED)
        self.stats_text.grid(row=0, column=0, sticky="w")
        self.quiz_short = self.label(row2, "", font=self.font_tiny, fg=PURPLE)
        self.quiz_short.grid(row=0, column=1, sticky="e")

        # 右侧设置
        right = self.card(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        setting_head = tk.Frame(right, bg=CARD)
        setting_head.grid(row=0, column=0, sticky="ew", padx=self.px(14), pady=(self.px(10), self.px(2)))
        self.label(setting_head, "运行设置", font=self.font_h1).pack(side="left")
        self.label(setting_head, "改动在下次开始时生效", font=self.font_tiny, fg=MUTED).pack(
            side="right", pady=self.px(3)
        )
        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=self.px(8), pady=(0, self.px(8)))
        self.tab_run = tk.Frame(self.notebook, bg=CARD)
        self.tab_llm = tk.Frame(self.notebook, bg=CARD)
        self.tab_adv = tk.Frame(self.notebook, bg=CARD)
        self.notebook.add(self.tab_run, text=" 运行 ")
        self.notebook.add(self.tab_llm, text=" LLM ")
        self.notebook.add(self.tab_adv, text=" 高级 ")
        self._build_run_tab()
        self._build_llm_tab()
        self._build_adv_tab()

        # 日志
        log_card = self.card(self.root)
        log_card.grid(row=3, column=0, sticky="ew", padx=pad, pady=(self.px(10), 0))
        log_card.grid_columnconfigure(0, weight=1)
        log_head = tk.Frame(log_card, bg=CARD)
        log_head.grid(row=0, column=0, sticky="ew", padx=self.px(12), pady=(self.px(7), self.px(2)))
        log_head.grid_columnconfigure(1, weight=1)
        self.label(log_head, "运行日志", font=self.font_h1).grid(row=0, column=0, sticky="w")
        self.label(log_head, "F8 紧急停止", font=self.font_tiny, fg=WARN).grid(row=0, column=1, sticky="e")
        self.log_text = tk.Text(
            log_card,
            height=5,
            bg="#070d18",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=self.font_log,
            wrap="word",
            padx=self.px(10),
            pady=self.px(6),
            state="disabled",
        )
        self.log_text.grid(row=1, column=0, sticky="ew", padx=self.px(10), pady=(0, self.px(8)))
        self.log_text.tag_configure("info", foreground=TEXT)
        self.log_text.tag_configure("success", foreground=ACCENT)
        self.log_text.tag_configure("warning", foreground=WARN)
        self.log_text.tag_configure("error", foreground="#fca5a5")
        self.log_text.tag_configure("dry", foreground=BLUE)
        self.log_text.tag_configure("muted", foreground=MUTED)

        # 底栏
        footer = tk.Frame(self.root, bg=BG)
        footer.grid(row=4, column=0, sticky="ew", padx=pad, pady=(self.px(10), self.px(10)))
        footer.grid_columnconfigure(0, weight=1)
        hint = self.label(
            footer,
            "请保持浏览器在前台，不要最小化；自动化会保护性暂停。",
            font=self.font_tiny,
            fg=MUTED,
            bg=BG,
        )
        hint.grid(row=0, column=0, sticky="w")
        self.save_btn = self.button(footer, "保存配置", self.save_config, "secondary", font=self.font_small)
        self.save_btn.grid(row=0, column=1, padx=(0, self.px(8)))
        self.stop_btn = self.button(footer, "停止  F8", self.stop, "danger", font=self.font_small, state="disabled")
        self.stop_btn.grid(row=0, column=2, padx=(0, self.px(8)))
        self.start_btn = self.button(footer, "开始自动学习", self.start, "primary", font=self.font_body)
        self.start_btn.grid(row=0, column=3)

    def _build_run_tab(self) -> None:
        f = self.tab_run
        f.configure(padx=self.px(12), pady=self.px(10))
        f.grid_columnconfigure(0, weight=1)
        checks = [
            ("视频结束后自动进入下一节", self.var_auto_next),
            ("自动答题：弹题 + 章节测验（LLM）", self.var_auto_quiz),
            ("视频暂停时自动恢复播放", self.var_auto_resume),
            ("试运行：只识别，不点击鼠标", self.var_dry_run),
            ("开始时最小化本窗口", self.var_minimize),
            ("窗口置顶", self.var_topmost),
        ]
        for i, (text, var) in enumerate(checks):
            cb = tk.Checkbutton(
                f,
                text=text,
                variable=var,
                font=self.font_body,
                bg=CARD,
                fg=TEXT,
                activebackground=CARD,
                activeforeground=TEXT,
                selectcolor=INPUT_BG,
                highlightthickness=0,
                bd=0,
                anchor="w",
                padx=0,
                pady=self.px(1),
                command=self._on_check_change if var is self.var_topmost else None,
            )
            cb.grid(row=i, column=0, sticky="w", pady=self.px(2))
        self._separator(f, 6)
        self._field_row(f, 7, "最多连续学习", self.var_max_sections, "节")
        self._field_row(f, 8, "无进度判定", self.var_stall, "秒后尝试恢复")
        self._field_row(f, 9, "结束判定余量", self.var_margin, "秒")
        self._field_row(f, 10, "答题重试次数", self.var_quiz_retry, "次")
        self.label(
            f,
            "提示：首次使用建议勾选“试运行”，先观察预览与日志里的定位是否正确。",
            font=self.font_tiny,
            fg=MUTED,
            wraplength=self.px(330),
            justify="left",
        ).grid(row=11, column=0, sticky="w", pady=(self.px(10), 0))

    def _build_llm_tab(self) -> None:
        f = self.tab_llm
        f.configure(padx=self.px(12), pady=self.px(8))
        f.grid_columnconfigure(0, weight=1)
        mode_row = tk.Frame(f, bg=CARD)
        mode_row.grid(row=0, column=0, sticky="w", pady=(0, self.px(6)))
        for text, value in (("云端 API", "cloud"), ("本地模型（Ollama）", "local")):
            rb = tk.Radiobutton(
                mode_row,
                text=text,
                variable=self.var_llm_mode,
                value=value,
                command=self._toggle_llm_mode,
                font=self.font_small,
                bg=CARD,
                fg=TEXT,
                activebackground=CARD,
                activeforeground=ACCENT,
                selectcolor=INPUT_BG,
                highlightthickness=0,
                bd=0,
            )
            rb.pack(side="left", padx=(0, self.px(16)))

        self.llm_cloud_frame = tk.Frame(f, bg=CARD)
        self.llm_cloud_frame.grid(row=1, column=0, sticky="ew")
        self.llm_cloud_frame.grid_columnconfigure(1, weight=1)
        cloud = self.llm_cloud_frame
        self._entry_row(cloud, 0, "API 地址", self.var_api_url)
        self.key_entry = self._entry_row(cloud, 1, "API Key", self.var_api_key, show="•")
        show_cb = tk.Checkbutton(
            cloud,
            text="显示密钥",
            variable=self.show_key,
            command=self._toggle_key,
            font=self.font_tiny,
            bg=CARD,
            fg=MUTED,
            activebackground=CARD,
            activeforeground=TEXT,
            selectcolor=INPUT_BG,
            highlightthickness=0,
            bd=0,
        )
        show_cb.grid(row=2, column=1, sticky="w", pady=(0, self.px(3)))
        self._entry_row(cloud, 3, "模型名称", self.var_model)
        self._entry_row(cloud, 4, "请求超时", self.var_timeout, suffix="秒")
        self._entry_row(cloud, 5, "最大输出", self.var_max_tokens, suffix="tokens")
        self._entry_row(cloud, 6, "温度", self.var_temperature)
        think_cb = tk.Checkbutton(
            cloud,
            text="启用深度思考（更慢，通常不需要）",
            variable=self.var_thinking,
            font=self.font_tiny,
            bg=CARD,
            fg=TEXT,
            activebackground=CARD,
            activeforeground=TEXT,
            selectcolor=INPUT_BG,
            highlightthickness=0,
            bd=0,
            anchor="w",
        )
        think_cb.grid(row=7, column=0, columnspan=2, sticky="w", pady=self.px(2))

        self.llm_local_frame = tk.Frame(f, bg=CARD)
        self.llm_local_frame.grid(row=1, column=0, sticky="ew")
        self.llm_local_frame.grid_columnconfigure(1, weight=1)
        local = self.llm_local_frame
        self._entry_row(local, 0, "本地接口", self.var_local_url)
        self._entry_row(local, 1, "本地模型", self.var_local_model)
        self._entry_row(local, 2, "请求超时", self.var_local_timeout, suffix="秒")
        self.button(
            local,
            "检测本地模型",
            self.discover_local_models,
            "ghost",
            font=self.font_tiny,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(self.px(4), 0))
        self.label(
            local,
            "默认 Ollama：http://127.0.0.1:11434/api/chat，需先启动 Ollama；"
            "本地模式不需要 API Key。",
            font=self.font_tiny,
            fg=MUTED,
            wraplength=self.px(330),
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(self.px(4), 0))

        self.test_llm_btn = self.button(
            f, "测试 LLM 连接", self.test_llm, "secondary", font=self.font_small
        )
        self.test_llm_btn.grid(row=2, column=0, sticky="w", pady=(self.px(8), self.px(4)))
        self.llm_result = self.label(
            f,
            "尚未测试。",
            font=self.font_tiny,
            fg=MUTED,
            wraplength=self.px(330),
            justify="left",
        )
        self.llm_result.grid(row=3, column=0, sticky="w")
        self.label(
            f,
            "云端走 OpenAI 兼容接口；本地走 Ollama /api/chat，也可填 /v1/chat/completions。",
            font=self.font_tiny,
            fg=MUTED,
            wraplength=self.px(330),
            justify="left",
        ).grid(row=4, column=0, sticky="w", pady=(self.px(8), 0))
        self._toggle_llm_mode()

    def _toggle_llm_mode(self) -> None:
        local = self.var_llm_mode.get() == "local"
        if local:
            self.llm_cloud_frame.grid_remove()
            self.llm_local_frame.grid()
        else:
            self.llm_local_frame.grid_remove()
            self.llm_cloud_frame.grid()

    def discover_local_models(self) -> None:
        url = self.var_local_url.get().strip()
        self.llm_result.configure(text="正在检测本地模型…", fg=MUTED)

        def worker() -> None:
            try:
                from .llm import list_local_models

                models = list_local_models(url)
                self.queue.put({"type": "local_models", "ok": True, "models": models})
            except Exception as exc:
                self.queue.put({"type": "local_models", "ok": False, "error": str(exc)})

        threading.Thread(target=worker, daemon=True).start()

    def _build_adv_tab(self) -> None:
        f = self.tab_adv
        f.configure(padx=self.px(12), pady=self.px(10))
        f.grid_columnconfigure(1, weight=1)
        self._entry_row(f, 0, "轮询间隔", self.var_poll, suffix="秒")
        self._entry_row(f, 1, "定期整页扫描", self.var_full_scan, suffix="秒")
        cb1 = tk.Checkbutton(
            f, text="运行中显示实时预览", variable=self.var_preview, font=self.font_body,
            bg=CARD, fg=TEXT, activebackground=CARD, activeforeground=TEXT,
            selectcolor=INPUT_BG, highlightthickness=0, bd=0, anchor="w"
        )
        cb1.grid(row=2, column=0, columnspan=2, sticky="w", pady=self.px(2))
        cb2 = tk.Checkbutton(
            f, text="保存答题/失败调试截图", variable=self.var_debug, font=self.font_body,
            bg=CARD, fg=TEXT, activebackground=CARD, activeforeground=TEXT,
            selectcolor=INPUT_BG, highlightthickness=0, bd=0, anchor="w"
        )
        cb2.grid(row=3, column=0, columnspan=2, sticky="w", pady=self.px(2))
        self._separator(f, 4)
        self.label(
            f,
            f"配置文件：{CONFIG_PATH}",
            font=self.font_tiny,
            fg=MUTED,
            wraplength=self.px(330),
            justify="left",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(self.px(8), self.px(2)))
        btnrow = tk.Frame(f, bg=CARD)
        btnrow.grid(row=6, column=0, columnspan=2, sticky="w", pady=self.px(6))
        self.button(btnrow, "打开日志目录", self.open_logs, "ghost", font=self.font_tiny).pack(side="left", padx=(0, self.px(6)))
        self.button(btnrow, "打开配置", self.open_config, "ghost", font=self.font_tiny).pack(side="left", padx=(0, self.px(6)))
        self.button(btnrow, "恢复默认", self.reset_settings, "ghost", font=self.font_tiny).pack(side="left")
        self.button(
            f,
            "测试鼠标点击",
            self.test_mouse_input,
            "secondary",
            font=self.font_tiny,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(self.px(8), 0))
        self.label(
            f,
            "识别使用 RapidOCR 本地模型，首次加载约 2~5 秒；默认每 25 秒整页复核一次。",
            font=self.font_tiny,
            fg=MUTED,
            wraplength=self.px(330),
            justify="left",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(self.px(8), 0))

    def _separator(self, parent: tk.Widget, row: int) -> None:
        line = tk.Frame(parent, bg=BORDER, height=1)
        line.grid(row=row, column=0, columnspan=2, sticky="ew", pady=self.px(8))
        line.grid_propagate(False)

    def _field_row(self, parent: tk.Widget, row: int, label: str, var: tk.StringVar, suffix: str = "") -> None:
        parent.grid_columnconfigure(1, weight=1)
        self.label(parent, label, font=self.font_small, fg=MUTED).grid(
            row=row, column=0, sticky="w", pady=self.px(3)
        )
        wrap = tk.Frame(parent, bg=CARD)
        wrap.grid(row=row, column=1, sticky="ew", pady=self.px(3))
        wrap.grid_columnconfigure(0, weight=1)
        entry = tk.Entry(
            wrap,
            textvariable=var,
            font=self.font_small,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        entry.grid(row=0, column=0, sticky="ew", ipady=self.px(3))
        self.label(wrap, suffix, font=self.font_tiny, fg=MUTED, bg=CARD).grid(
            row=0, column=1, padx=(self.px(5), 0)
        )

    def _entry_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        var: tk.StringVar,
        show: str | None = None,
        suffix: str = "",
    ) -> tk.Entry:
        parent.grid_columnconfigure(1, weight=1)
        self.label(parent, label, font=self.font_small, fg=MUTED).grid(
            row=row, column=0, sticky="w", pady=self.px(3), padx=(0, self.px(8))
        )
        wrap = tk.Frame(parent, bg=CARD)
        wrap.grid(row=row, column=1, sticky="ew", pady=self.px(3))
        wrap.grid_columnconfigure(0, weight=1)
        entry = tk.Entry(
            wrap,
            textvariable=var,
            show=show or "",
            font=self.font_small,
            bg=INPUT_BG,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=ACCENT,
        )
        entry.grid(row=0, column=0, sticky="ew", ipady=self.px(3))
        if suffix:
            self.label(wrap, suffix, font=self.font_tiny, fg=MUTED, bg=CARD).grid(
                row=0, column=1, padx=(self.px(5), 0)
            )
        return entry

    # ------------------------------------------------------------------ 配置
    def _populate_settings(self) -> None:
        s = self.settings
        self.var_auto_next.set(s.auto_next)
        self.var_auto_quiz.set(s.auto_quiz)
        self.var_auto_resume.set(s.auto_resume)
        self.var_dry_run.set(s.dry_run)
        self.var_minimize.set(s.minimize_on_start)
        self.var_topmost.set(s.always_on_top)
        self.var_preview.set(s.preview_enabled)
        self.var_debug.set(s.save_debug_frames)
        self.var_max_sections.set(str(s.max_sections))
        self.var_stall.set(f"{s.stall_seconds:g}")
        self.var_margin.set(f"{s.completion_margin:g}")
        self.var_poll.set(f"{s.poll_interval:g}")
        self.var_full_scan.set(f"{s.full_scan_interval:g}")
        self.var_quiz_retry.set(str(s.quiz_retry))
        self.var_llm_mode.set(getattr(s, "llm_mode", "cloud"))
        self.var_api_url.set(s.api_url)
        self.var_api_key.set(s.api_key)
        self.var_model.set(s.model)
        self.var_local_url.set(getattr(s, "local_api_url", ""))
        self.var_local_model.set(getattr(s, "local_model", ""))
        self.var_local_timeout.set(str(getattr(s, "local_timeout", 180)))
        if hasattr(self, "llm_local_frame"):
            self._toggle_llm_mode()
        self.var_timeout.set(str(s.llm_timeout))
        self.var_max_tokens.set(str(s.max_tokens))
        self.var_temperature.set(f"{s.temperature:g}")
        self.var_thinking.set(s.enable_thinking)

    def _collect_settings(self, strict: bool = False) -> bool:
        errors: list[str] = []

        def num(var: tk.StringVar, name: str, default, lo, hi, cast=float):
            raw = var.get().strip()
            try:
                value = cast(raw)
            except (TypeError, ValueError):
                errors.append(f"{name} 必须是数字")
                return default
            if not (lo <= value <= hi):
                errors.append(f"{name} 需在 {lo} ~ {hi} 之间")
                return default
            return value

        s = self.settings
        s.auto_next = bool(self.var_auto_next.get())
        s.auto_quiz = bool(self.var_auto_quiz.get())
        s.auto_resume = bool(self.var_auto_resume.get())
        s.dry_run = bool(self.var_dry_run.get())
        s.minimize_on_start = bool(self.var_minimize.get())
        s.always_on_top = bool(self.var_topmost.get())
        s.preview_enabled = bool(self.var_preview.get())
        s.save_debug_frames = bool(self.var_debug.get())
        s.max_sections = int(num(self.var_max_sections, "最多连续学习", s.max_sections, 1, 500, int))
        s.stall_seconds = num(self.var_stall, "无进度判定", s.stall_seconds, 4, 300)
        s.completion_margin = num(self.var_margin, "结束判定余量", s.completion_margin, 0.5, 60)
        s.poll_interval = num(self.var_poll, "轮询间隔", s.poll_interval, 0.2, 5)
        s.full_scan_interval = num(self.var_full_scan, "整页扫描间隔", s.full_scan_interval, 5, 300)
        s.quiz_retry = int(num(self.var_quiz_retry, "答题重试次数", s.quiz_retry, 0, 5, int))
        s.llm_mode = self.var_llm_mode.get().strip() or "cloud"
        s.api_url = self.var_api_url.get().strip()
        s.api_key = self.var_api_key.get().strip()
        s.model = self.var_model.get().strip()
        s.local_api_url = self.var_local_url.get().strip()
        s.local_model = self.var_local_model.get().strip()
        s.local_timeout = int(
            num(self.var_local_timeout, "本地请求超时", s.local_timeout, 10, 600, int)
        )
        s.llm_timeout = int(num(self.var_timeout, "请求超时", s.llm_timeout, 10, 300, int))
        s.max_tokens = int(num(self.var_max_tokens, "最大输出", s.max_tokens, 256, 16384, int))
        s.temperature = num(self.var_temperature, "温度", s.temperature, 0, 2)
        s.enable_thinking = bool(self.var_thinking.get())
        if s.llm_mode == "cloud" and not s.api_url.startswith(("http://", "https://")):
            errors.append("API 地址需以 http:// 或 https:// 开头")
        if s.llm_mode == "local" and not s.local_api_url.startswith(("http://", "https://")):
            errors.append("本地接口需以 http:// 或 https:// 开头")
        if s.llm_mode == "local" and not s.local_model:
            errors.append("本地模式需要填写模型名称")
        if self.selected_window:
            s.window_title = self.selected_window.title
            s.window_class = self.selected_window.class_name
        if errors and strict:
            messagebox.showerror("配置有误", "\n".join(errors), parent=self.root)
            return False
        if errors:
            self._append_log("warning", "部分配置无效，已使用默认值：" + "；".join(errors))
        return True

    def save_config(self) -> None:
        if not self._collect_settings(strict=False):
            return
        try:
            self.settings.save()
            self._append_log("success", f"配置已保存：{CONFIG_PATH}")
        except Exception as exc:
            self._append_log("error", f"保存配置失败：{exc}")
            messagebox.showerror("保存失败", str(exc), parent=self.root)

    def reset_settings(self) -> None:
        if not messagebox.askyesno("恢复默认", "将恢复所有设置（包括 API Key 为默认值），继续？", parent=self.root):
            return
        self.settings = Settings()
        self._populate_settings()
        self._apply_always_on_top()
        self._append_log("warning", "已恢复默认设置（尚未保存到文件）")

    def _toggle_key(self) -> None:
        try:
            self.key_entry.configure(show="" if self.show_key.get() else "•")
        except Exception:
            pass

    def _apply_always_on_top(self) -> None:
        try:
            self.root.attributes("-topmost", bool(self.var_topmost.get()))
        except Exception:
            pass

    def _on_check_change(self) -> None:
        self._apply_always_on_top()

    # ------------------------------------------------------------------ 窗口
    def refresh_windows(self, initial: bool = False) -> None:
        old_hwnd = self.selected_window.hwnd if self.selected_window else 0
        try:
            infos = list_windows()
        except Exception as exc:
            self._append_log("error", f"枚举窗口失败：{exc}")
            return
        browsers = [w for w in infos if w.class_name == "Chrome_WidgetWin_1"]
        others = [w for w in infos if w not in browsers]
        self.window_infos = browsers + others
        labels = [w.label() for w in self.window_infos]
        self.window_combo.configure(values=labels)
        chosen = None
        if old_hwnd:
            for w in self.window_infos:
                if w.hwnd == old_hwnd:
                    chosen = w
                    break
        if chosen is None and self.settings.window_title:
            for w in self.window_infos:
                if self.settings.window_title in w.title and w.title.startswith("学生学习页面"):
                    chosen = w
                    break
        if chosen is None and self.window_infos:
            for w in self.window_infos:
                if w.title.startswith("学生学习页面"):
                    chosen = w
                    break
        if chosen:
            self.selected_window = chosen
            self.var_window.set(chosen.label())
            self.window_combo.current(self.window_infos.index(chosen))
            if initial:
                self._append_log("info", f"自动选择窗口：{chosen.label()}")
        else:
            self.selected_window = None
            self.var_window.set("")
            self._append_log("warning", "未找到课程学习页，请在独立浏览器打开课程后刷新并选择窗口")

    def _on_window_selected(self, _event=None) -> None:
        idx = self.window_combo.current()
        if 0 <= idx < len(self.window_infos):
            self.selected_window = self.window_infos[idx]
            self._append_log("info", f"已选择窗口：{self.selected_window.label()}")

    # ------------------------------------------------------------------ 预览
    def _draw_preview_placeholder(self) -> None:
        c = self.preview_canvas
        c.delete("all")
        w = max(300, c.winfo_width())
        h = max(180, c.winfo_height())
        c.create_text(
            w // 2,
            h // 2,
            text="选择窗口后点击「扫描预览」\n这里会显示 OCR 定位框",
            fill="#476080",
            font=self.font_body,
            justify="center",
        )

    def _show_preview(self, jpeg: bytes) -> None:
        try:
            img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        except Exception:
            return
        self._preview_image = img
        self._render_preview()

    def _render_preview(self) -> None:
        if not self._preview_image:
            return
        c = self.preview_canvas
        cw = max(50, c.winfo_width())
        ch = max(50, c.winfo_height())
        img = self._preview_image.copy()
        scale = min(cw / img.width, ch / img.height)
        nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(img)
        c.delete("all")
        c.create_image(cw // 2, ch // 2, image=self._preview_photo, anchor="center")

    def scan_preview(self) -> None:
        if self.running:
            self._append_log("warning", "自动学习运行中已有实时预览，无需重复扫描")
            return
        if self._scan_busy:
            return
        win = self.selected_window
        if not win:
            messagebox.showwarning("没有窗口", "请先选择目标浏览器窗口。", parent=self.root)
            return
        self._scan_busy = True
        self.scan_btn.configure(state="disabled")
        self._append_log("info", f"开始扫描预览：{win.label()}")
        self._collect_settings(strict=False)
        max_width = self.settings.ocr_max_width
        # 截图会抓真实屏幕；先最小化本窗口，避免预览/识别被本窗口遮挡。
        self._preview_restore = True
        try:
            self.root.iconify()
        except Exception:
            pass
        self.root.after(300, lambda: focus_window(win.hwnd))
        self.root.after(
            650,
            lambda: threading.Thread(
                target=self._scan_preview_worker, args=(win, max_width), daemon=True
            ).start(),
        )

    def _scan_preview_worker(self, win: WindowInfo, max_width: int) -> None:
        cap = None

        def log(level: str, text: str) -> None:
            self.queue.put({"type": "log", "level": level, "text": text})

        try:
            reader = BrowserAccessibility(win)
            try:
                inventory = reader.read()
            finally:
                reader.close()
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            from dataclasses import asdict
            (LOG_DIR / "preview_inventory.json").write_text(
                json.dumps(asdict(inventory), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if inventory.complete and inventory.page_key:
                log("info", inventory.inventory_text())
            else:
                log("warning", f"整页盘点未完成：{inventory.problem or '无法确认章节身份'}")
            if self._preview_ocr is None:
                self._preview_ocr = TextOCR(max_width)
                self._preview_ocr.warmup()
            # 播放器控制栏会自动隐藏，先扫过视频区域再截图，确保时间可识别。
            try:
                import ctypes

                for fx, fy in ((0.34, 0.56), (0.56, 0.62), (0.43, 0.58)):
                    ctypes.windll.user32.SetCursorPos(
                        win.left + int(win.width * fx), win.top + int(win.height * fy)
                    )
                    time.sleep(0.12)
                time.sleep(0.15)
            except Exception:
                pass
            cap = ScreenCapture(win)
            frame = cap.grab()
            lines = self._preview_ocr.full(frame.image)
            view = analyze_page(
                frame.image, lines, line_reader=self._preview_ocr.line
            )
            annotated = annotate_frame(
                frame.image,
                view,
                f"PREVIEW  {view.time.raw if view.time else 'no time'}  sections={len(view.sections)}",
            )
            data = preview_jpeg(annotated, 1100, 86)
            self.queue.put({"type": "preview", "jpeg": data})
            self.queue.put(
                {
                    "type": "status",
                    "state": "idle",
                    "text": "预览扫描完成",
                    "current": view.time.current if view.time else None,
                    "duration": view.time.duration if view.time else None,
                    "percent": view.time.percent * 100 if view.time else 0,
                }
            )
            try:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                cv2_ok = cv2_imwrite(LOG_DIR / "preview_last.jpg", annotated)
                (LOG_DIR / "preview_last_ocr.txt").write_text(ocr_debug_text(view), encoding="utf-8")
                if not cv2_ok:
                    log("warning", "预览截图保存失败")
            except Exception:
                pass
            title = view.page_title or "-"
            t = f"{view.time.current_text()} / {view.time.duration_text()}" if view.time else "未检测到时间"
            cur = "-"
            if view.current_idx is not None and view.sections:
                cur = view.sections[view.current_idx].display
            log("success", f"预览完成：时间 {t}；当前小节 {cur}；目录 {len(view.sections)} 项；页面标题 {title}")
            pending = [r for r in view.sections if r.pending_tasks == 1 and not r.completed]
            if pending:
                log(
                    "warning",
                    "发现待补章节测验："
                    + "、".join(f"{r.number} {r.title}" for r in pending[:6]),
                )
            if view.quiz.detected:
                log("warning", f"检测到题目：{view.quiz.kind}，{len(view.quiz.options)} 个选项")
            if view.task_completed is False:
                log("warning", "当前小节任务点为黄色（视频未完成），运行时会自动开始/继续播放")
            if view.exam.detected:
                log(
                    "warning",
                    f"检测到章节测验：{len(view.exam.questions)} 题可见，"
                    f"本章总题量 {view.exam.total or '未知'}，提交按钮"
                    f"{'可见' if view.exam.submit else '未可见'}",
                )
            if not view.sections:
                log("warning", "未识别到右侧目录，请确认窗口选择正确且页面已加载")
        except Exception as exc:
            log("error", f"预览失败：{exc}")
        finally:
            if cap:
                cap.close()
            self.queue.put({"type": "scan_done"})

    # ------------------------------------------------------------------ LLM
    def test_llm(self) -> None:
        self._collect_settings(strict=False)
        self.test_llm_btn.configure(state="disabled", text="测试中…")
        self.llm_result.configure(text="正在请求接口…", fg=MUTED)
        snapshot = Settings()
        for key, value in self.settings.to_dict().items():
            try:
                setattr(snapshot, key, value)
            except Exception:
                pass
        threading.Thread(target=self._test_llm_worker, args=(snapshot,), daemon=True).start()

    def _test_llm_worker(self, settings: Settings) -> None:
        try:
            reply, latency = LLMClient(settings).test_connection()
            self.queue.put({"type": "llm_test", "ok": True, "reply": reply, "latency": latency})
        except Exception as exc:
            self.queue.put({"type": "llm_test", "ok": False, "error": str(exc)})

    # ------------------------------------------------------------------ 运行控制
    def start(self) -> None:
        if self.running:
            return
        win = self.selected_window
        if not win:
            messagebox.showwarning("没有窗口", "请先选择目标浏览器窗口，建议先扫描预览。", parent=self.root)
            return
        if not self._collect_settings(strict=True):
            return
        if (
            self.settings.auto_quiz
            and self.settings.llm_mode != "local"
            and not self.settings.api_key
        ):
            if not messagebox.askyesno(
                "未配置 API Key",
                "自动答题需要 LLM API Key，当前为空。是否仍要开始（遇到题目会停止）？",
                parent=self.root,
            ):
                return
        try:
            self.settings.save()
        except Exception:
            pass
        self.stop_event = threading.Event()
        self._last_error_reason = ""
        self.stop_btn.configure(state="normal", bg="#3a1720")
        self.start_btn.configure(state="disabled", bg="#244b36", text="运行中…")
        self.running = True
        self._append_log(
            "info",
            f"开始自动学习：{win.label()}；{'试运行（不点击）' if self.settings.dry_run else '真实点击模式'}",
        )
        focus_window(win.hwnd)
        overlap = self._root_overlaps(win)
        self._force_minimize = bool(self.settings.minimize_on_start or overlap)
        if overlap and not self.settings.minimize_on_start:
            self._append_log("warning", "本窗口与目标区域重叠，运行时将自动最小化，避免遮挡截图。")
        if self._force_minimize:
            self.root.after(900, self._minimize_self)
        if self.settings.dry_run:
            self._append_log("dry", "试运行：日志中的“将点击”不会实际移动鼠标。")
        self.worker = AutomationWorker(self.settings, win, self.queue, self.stop_event)
        self.worker.start()

    def _minimize_self(self) -> None:
        if self.running and getattr(self, "_force_minimize", False):
            try:
                self.root.iconify()
            except Exception:
                pass

    def _root_overlaps(self, win: WindowInfo) -> bool:
        try:
            self.root.update_idletasks()
            rx = self.root.winfo_rootx()
            ry = self.root.winfo_rooty()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            return not (
                rx + rw <= win.left
                or win.right <= rx
                or ry + rh <= win.top
                or win.bottom <= ry
            )
        except Exception:
            return True

    def stop(self) -> None:
        if not self.running:
            return
        self._append_log("warning", "正在停止自动化…")
        self.stop_event.set()
        self.stop_btn.configure(state="disabled", text="停止中…")
        self.header_status.configure(text="正在停止…", fg=WARN)
        self.status_dot.configure(fg=WARN)

    def _on_worker_stopped(self, reason: str) -> None:
        self.running = False
        self._last_error_reason = reason
        self.start_btn.configure(state="normal", bg=ACCENT, text="开始自动学习")
        self.stop_btn.configure(state="disabled", text="停止  F8")
        self.header_status.configure(text="已停止" if "出错" not in reason else "异常停止", fg=MUTED)
        self.status_dot.configure(fg=MUTED)
        try:
            self.root.deiconify()
            self.root.lift()
        except Exception:
            pass
        self._append_log("warning" if reason else "info", f"自动化结束：{reason}")
        self.worker = None

    def _setup_hotkey(self) -> None:
        try:
            self.hotkey = HotkeyListener(self._hotkey_stop)
            self.hotkey.start()
            self.hotkey.ready.wait(0.6)
            if self.hotkey.ok:
                self._append_log("info", "F8 紧急停止热键已注册")
            else:
                self._append_log("warning", "F8 热键注册失败，可能被其他程序占用；请使用界面停止按钮")
        except Exception as exc:
            self._append_log("warning", f"F8 热键不可用：{exc}")

    def _hotkey_stop(self) -> None:
        if self.running:
            self.stop_event.set()
            self.queue.put({"type": "log", "level": "warning", "text": "收到 F8：正在紧急停止…"})

    # ------------------------------------------------------------------ 队列
    def _append_log(self, level: str, text: str) -> None:
        if not hasattr(self, "log_text"):
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        tag = level if level in {"info", "success", "warning", "error", "dry", "muted"} else "info"
        prefix = {"success": "✓ ", "warning": "! ", "error": "✕ ", "dry": "· "}.get(tag, "")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{stamp}] {prefix}{text}\n", tag)
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 900:
            self.log_text.delete("1.0", "200.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        except Exception as exc:
            self._append_log("error", f"界面消息处理异常：{exc}")
        if self.queue_ok:
            self.root.after(120, self._poll_queue)

    def _handle_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "log":
            self._append_log(msg.get("level", "info"), msg.get("text", ""))
        elif kind == "status":
            state = msg.get("state", "idle")
            color = STATE_COLORS.get(state, MUTED)
            self.header_status.configure(text=msg.get("text", ""), fg=color)
            self.status_dot.configure(fg=color)
            self.state_text.configure(text=msg.get("text", ""), fg=TEXT)
            current = msg.get("current")
            duration = msg.get("duration")
            if current is not None and duration:
                self.time_text.configure(
                    text=f"{format_clock(current)} / {format_clock(duration)}", fg=TEXT
                )
            elif current is None:
                self.time_text.configure(text="--:-- / --:--", fg=MUTED)
            pct = float(msg.get("percent") or 0)
            self.progress.configure(value=max(0, min(100, pct)))
        elif kind == "preview":
            data = msg.get("jpeg")
            if data:
                self._show_preview(data)
        elif kind == "stats":
            self.stats_text.configure(
                text=f"本次完成 {msg.get('sections', 0)} 节 · 自动答题 {msg.get('quizzes', 0)} 题"
            )
        elif kind == "quiz_answer":
            ans = ",".join(msg.get("answer") or [])
            reason = str(msg.get("reason") or "")[:80]
            self.quiz_short.configure(text=f"LLM: {ans} {reason}")
            self._append_log("info", f"LLM 答案 {ans}；{reason}")
        elif kind == "local_models":
            if msg.get("ok"):
                models = list(msg.get("models") or [])
                current = self.var_local_model.get().strip()
                if models and (not current or current not in models):
                    self.var_local_model.set(models[0])
                self.llm_result.configure(
                    text="发现本地模型：" + "、".join(models[:8]), fg=ACCENT
                )
                self._append_log("success", "发现本地模型：" + "、".join(models))
            else:
                err = str(msg.get("error", "未知错误"))
                self.llm_result.configure(text=f"检测失败：{err[:220]}", fg="#fca5a5")
                self._append_log("error", f"检测本地模型失败：{err}")
        elif kind == "llm_test":
            self.test_llm_btn.configure(state="normal", text="测试 LLM 连接")
            if msg.get("ok"):
                self.llm_result.configure(
                    text=f"连接成功（{msg.get('latency', 0):.1f}s）：{str(msg.get('reply', ''))[:120]}",
                    fg=ACCENT,
                )
                self._append_log("success", f"LLM 连接成功，耗时 {msg.get('latency', 0):.1f}s")
            else:
                err = str(msg.get("error", "未知错误"))
                self.llm_result.configure(text=f"连接失败：{err[:220]}", fg="#fca5a5")
                self._append_log("error", f"LLM 连接失败：{err}")
        elif kind == "scan_done":
            self._scan_busy = False
            self.scan_btn.configure(state="normal")
            if getattr(self, "_preview_restore", False):
                self._preview_restore = False
                try:
                    self.root.deiconify()
                    self.root.lift()
                except Exception:
                    pass
        elif kind == "stopped":
            self._on_worker_stopped(str(msg.get("reason", "")))

    # ------------------------------------------------------------------ 其它
    def test_mouse_input(self) -> None:
        """在独立测试窗口里让程序自己点击按钮，验证 SendInput 是否可用。"""
        if self.running:
            messagebox.showwarning("鼠标测试", "请先停止自动学习。", parent=self.root)
            return
        test = tk.Toplevel(self.root)
        test.title("鼠标点击测试")
        test.configure(bg=CARD)
        test.attributes("-topmost", True)
        w, h = self.px(430), self.px(230)
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - w) // 2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height() - h) // 2)
        test.geometry(f"{w}x{h}+{x}+{y}")
        test.lift()
        test.focus_force()
        test.update_idletasks()
        test.update()
        self.label(test, "鼠标输入测试", font=self.font_h1).pack(pady=(self.px(16), self.px(4)))
        self.label(
            test,
            "1 秒后程序会移动鼠标并点击下方按钮",
            font=self.font_small,
            fg=MUTED,
        ).pack()
        state = {"clicked": False}

        def on_hit():
            state["clicked"] = True
            result.configure(text="点击成功", fg=ACCENT)

        target = self.button(test, "测试目标（自动点击）", on_hit, "primary", font=self.font_body)
        target.pack(pady=self.px(16), fill="x", padx=self.px(40))
        result = self.label(test, "等待自动点击…", font=self.font_small, fg=WARN)
        result.pack(pady=(self.px(2), 0))

        def finish():
            ok = state["clicked"]
            try:
                test.destroy()
            except Exception:
                pass
            if ok:
                messagebox.showinfo(
                    "鼠标测试",
                    "自动点击成功。程序的鼠标控制正常；如果网页仍无反应，"
                    "请检查浏览器是否在前台、页面是否被其他窗口遮挡。",
                    parent=self.root,
                )
                self._append_log("success", "鼠标点击测试成功")
            else:
                messagebox.showerror(
                    "鼠标测试",
                    "自动点击没有到达测试按钮。可能是安全软件、远程控制、"
                    "权限级别或输入桌面拦截了模拟鼠标。请查看日志。",
                    parent=self.root,
                )
                self._append_log("error", "鼠标点击测试失败：模拟点击未到达目标按钮")

        def do_click():
            try:
                test.update_idletasks()
                test.update()
                bx = target.winfo_rootx() + target.winfo_width() // 2
                by = target.winfo_rooty() + target.winfo_height() // 2
                clicker = Clicker(dry_run=False)
                clicker.click(bx, by)
                self._append_log(
                    "info",
                    f"鼠标测试：目标 ({bx},{by})，鼠标 {clicker.last_before}->{clicker.last_after}，"
                    f"方式 {clicker.last_method}",
                )
            except MouseInputError as exc:
                self._append_log("error", f"鼠标测试失败：{exc}")
            except Exception as exc:
                self._append_log("error", f"鼠标测试异常：{exc}")
            test.after(self.px(600), finish)

        test.after(self.px(1000), do_click)

    def open_logs(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(LOG_DIR))  # type: ignore[attr-defined]

    def open_config(self) -> None:
        try:
            if not CONFIG_PATH.exists():
                self.settings.save()
            os.startfile(str(CONFIG_PATH))  # type: ignore[attr-defined]
        except Exception as exc:
            self._append_log("error", f"打开配置失败：{exc}")

    def on_close(self) -> None:
        if self.running:
            if not messagebox.askyesno("退出", "自动化正在运行，确定停止并退出？", parent=self.root):
                return
            self.stop_event.set()
        self.queue_ok = False
        try:
            if self.hotkey:
                self.hotkey.stop()
        except Exception:
            pass
        try:
            self._collect_settings(strict=False)
            self.settings.save()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


def format_clock(value: float) -> str:
    total = max(0, int(round(float(value))))
    m, s = divmod(total, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def cv2_imwrite(path: Path, image) -> bool:
    import cv2

    return bool(cv2.imwrite(str(path), image))


def run_app(settings: Settings) -> None:
    root = tk.Tk()
    app = CourseBotApp(root, settings)
    root.mainloop()
