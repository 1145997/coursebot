"""独立于主窗口的运行进度悬浮窗；仅由 Tk 主线程更新。"""
from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from tkinter import ttk
from collections.abc import Callable

from .capture import register_capture_overlay, unregister_capture_overlay


class ProgressOverlay:
    def __init__(self, root: tk.Tk, scale: float, on_open: Callable,
                 on_stop: Callable, on_close: Callable):
        self.root = root
        self.scale = scale
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.title("CourseBot · 运行进度")
        # 不设 transient：Windows 会随主窗口最小化 transient 窗口。
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        self.window.configure(bg="#23324d", padx=1, pady=1)
        self.window.protocol("WM_DELETE_WINDOW", on_close)
        self.window.bind("<Destroy>", self._on_destroy)
        self._drag_origin: tuple[int, int, int, int] | None = None
        self._positioned = False
        self._hwnd = 0
        self.capture_excluded = False

        card = tk.Frame(self.window, bg="#0f172a", padx=self.px(12), pady=self.px(10))
        card.pack(fill="both", expand=True)
        card.grid_columnconfigure(0, weight=1)
        header = tk.Frame(card, bg="#0f172a", cursor="fleur")
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        title = tk.Label(header, text="运行进度", bg="#0f172a", fg="#8fa4c2",
                         font=("Microsoft YaHei UI", 9, "bold"), cursor="fleur")
        title.pack(side="left")
        close = self._button(header, "×", on_close)
        close.pack(side="right")
        for widget in (header, title):
            widget.bind("<ButtonPress-1>", self._begin_drag)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<ButtonRelease-1>", lambda _event: self._end_drag())

        self.status = tk.Label(card, text="等待开始", bg="#0f172a", fg="#e7eef9",
                               font=("Microsoft YaHei UI", 10, "bold"), anchor="w",
                               justify="left", wraplength=self.px(270), height=2)
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(self.px(2), 0))
        self.clock = tk.Label(card, text="--:-- / --:--", bg="#0f172a", fg="#e7eef9",
                              font=("Cascadia Mono", 10, "bold"), anchor="w")
        self.clock.grid(row=2, column=0, sticky="w")
        self.percent = tk.Label(card, text="--", bg="#0f172a", fg="#8fa4c2",
                                font=("Microsoft YaHei UI", 9))
        self.percent.grid(row=2, column=1, sticky="e")
        self.progress = ttk.Progressbar(card, maximum=100, style="Horizontal.TProgressbar")
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=self.px(6))
        self.stats = tk.Label(card, text="本次完成 0 节 · 自动答题 0 题", bg="#0f172a",
                              fg="#8fa4c2", font=("Microsoft YaHei UI", 8), anchor="w")
        self.stats.grid(row=4, column=0, columnspan=2, sticky="ew")
        footer = tk.Frame(card, bg="#0f172a")
        footer.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(self.px(8), 0))
        self._button(footer, "主窗口", on_open).pack(side="left")
        self.stop_button = self._button(footer, "停止 F8", on_stop, fg="#fca5a5")
        self.stop_button.pack(side="right")
        self.stop_button.configure(state="disabled")
        self.window.update_idletasks()

    def px(self, value: int) -> int:
        return max(1, round(value * self.scale))

    def _button(self, parent: tk.Widget, text: str, command: Callable,
                fg: str = "#e7eef9") -> tk.Button:
        return tk.Button(parent, text=text, command=command, bg="#121f35", fg=fg,
                         activebackground="#23324d", activeforeground=fg,
                         disabledforeground="#6c7f99", font=("Microsoft YaHei UI", 8),
                         relief="flat", bd=0, highlightthickness=0, cursor="hand2",
                         padx=self.px(8), pady=self.px(2), takefocus=False)

    def _configure_native_window(self) -> None:
        if sys.platform != "win32":
            return
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.GetWindowLongW.restype = wintypes.LONG
        user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
        user32.SetWindowLongW.restype = wintypes.LONG
        user32.SetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPCWSTR]
        user32.SetWindowTextW.restype = wintypes.BOOL
        hwnd = user32.GetAncestor(self.window.winfo_id(), 2)  # GA_ROOT
        if self._hwnd and self._hwnd != hwnd:
            unregister_capture_overlay(self._hwnd)
        self._hwnd = hwnd
        # NOACTIVATE 窗口默认不占任务栏，鼠标操作不抢浏览器焦点。
        style = user32.GetWindowLongW(self._hwnd, -20)
        user32.SetWindowLongW(self._hwnd, -20, (style | 0x08000000) & ~0x00000080)
        user32.SetWindowTextW(self._hwnd, "CourseBot · 运行进度")
        self.capture_excluded = register_capture_overlay(self._hwnd)

    def _on_destroy(self, event) -> None:
        if event.widget is self.window:
            unregister_capture_overlay(self._hwnd)

    def show(self) -> None:
        if not self._positioned:
            self.window.update_idletasks()
            x = self.root.winfo_screenwidth() - self.window.winfo_reqwidth() - self.px(20)
            self.window.geometry(f"+{max(0, x)}+{self.px(72)}")
            self._positioned = True
        self.window.deiconify()
        self.window.update_idletasks()
        self._configure_native_window()

    def hide(self) -> None:
        self.window.withdraw()

    def update(self, *, text: str, color: str, clock: str, percent: float,
               stats: str, running: bool, stopping: bool) -> None:
        self.status.configure(text=text, fg=color)
        self.clock.configure(text=clock)
        self.progress.configure(value=percent)
        self.percent.configure(text=f"{percent:.0f}%" if clock != "--:-- / --:--" else "--")
        self.stats.configure(text=stats)
        self.stop_button.configure(state="normal" if running and not stopping else "disabled",
                                   text="停止中…" if stopping else "停止 F8")

    def _begin_drag(self, event) -> None:
        self._drag_origin = (event.x_root, event.y_root, self.window.winfo_x(), self.window.winfo_y())

    def _drag(self, event) -> None:
        if self._drag_origin is None:
            return
        mx, my, x, y = self._drag_origin
        left, top = self.window.winfo_vrootx(), self.window.winfo_vrooty()
        right = left + self.window.winfo_vrootwidth() - self.window.winfo_width()
        bottom = top + self.window.winfo_vrootheight() - self.window.winfo_height()
        if sys.platform == "win32":
            metrics = ctypes.windll.user32.GetSystemMetrics
            left, top = metrics(76), metrics(77)
            right = left + metrics(78) - self.window.winfo_width()
            bottom = top + metrics(79) - self.window.winfo_height()
        x = max(left, min(x + event.x_root - mx, max(left, right)))
        y = max(top, min(y + event.y_root - my, max(top, bottom)))
        if sys.platform == "win32" and self._hwnd:
            from ctypes import wintypes
            move = ctypes.windll.user32.SetWindowPos
            move.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, wintypes.UINT]
            move.restype = wintypes.BOOL
            move(self._hwnd, None, x, y, 0, 0, 0x0015)  # NOSIZE | NOZORDER | NOACTIVATE
        else:
            self.window.geometry(f"{x:+d}{y:+d}")

    def _end_drag(self) -> None:
        self._drag_origin = None
