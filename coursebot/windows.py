from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Iterable

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
shell32 = ctypes.windll.shell32

SW_RESTORE = 9
SW_MINIMIZE = 6
VK_F8 = 0x77
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
KEYEVENTF_KEYUP = 0x0002
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


try:
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
except Exception:
    pass


class MouseInputError(RuntimeError):
    pass


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    class_name: str
    left: int
    top: int
    width: int
    height: int
    pid: int = 0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def label(self) -> str:
        title = self.title.strip() or "(无标题)"
        if len(title) > 52:
            title = title[:49] + "…"
        return f"{title}  [{self.class_name}]  {self.width}×{self.height}"


def enable_dpi_awareness() -> int:
    """开启 Per-Monitor DPI 感知，确保截图像素与鼠标坐标一致。返回 DPI。"""
    try:
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    try:
        dpi = int(user32.GetDpiForSystem())
    except Exception:
        try:
            hdc = user32.GetDC(0)
            dpi = int(ctypes.windll.gdi32.GetDeviceCaps(hdc, 88))  # LOGPIXELSX
            user32.ReleaseDC(0, hdc)
        except Exception:
            dpi = 96
    return dpi if dpi > 0 else 96


def virtual_screen() -> tuple[int, int, int, int]:
    x = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
    y = user32.GetSystemMetrics(77)
    w = user32.GetSystemMetrics(78)
    h = user32.GetSystemMetrics(79)
    return int(x), int(y), int(w), int(h)


def get_window_title(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def get_class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def client_rect_on_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    w, h = int(rect.right - rect.left), int(rect.bottom - rect.top)
    if w <= 0 or h <= 0:
        return None
    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return int(pt.x), int(pt.y), w, h


def get_window_info(hwnd: int) -> WindowInfo | None:
    if not hwnd or not user32.IsWindow(hwnd) or not user32.IsWindowVisible(hwnd):
        return None
    if user32.IsIconic(hwnd):
        return None
    rect = client_rect_on_screen(hwnd)
    if not rect:
        return None
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return WindowInfo(
        hwnd=int(hwnd),
        title=get_window_title(hwnd),
        class_name=get_class_name(hwnd),
        left=rect[0],
        top=rect[1],
        width=rect[2],
        height=rect[3],
        pid=int(pid.value),
    )


def list_windows(min_width: int = 480, min_height: int = 320) -> list[WindowInfo]:
    result: list[WindowInfo] = []
    own_pid = kernel32.GetCurrentProcessId()
    cb_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        info = get_window_info(hwnd)
        if not info or info.pid == own_pid:
            return True
        if info.width < min_width or info.height < min_height:
            return True
        if not info.title.strip():
            return True
        # 跳过桌面/任务栏等外壳窗口
        if info.class_name in {"Progman", "WorkerW", "Shell_TrayWnd", "Windows.UI.Core.CoreWindow"}:
            return True
        result.append(info)
        return True

    user32.EnumWindows(cb_type(callback), 0)
    result.sort(key=lambda w: (w.area, w.title), reverse=True)
    return result


BROWSER_CLASSES = {
    "Chrome_WidgetWin_1",
    "MozillaWindowClass",
    "OperaWindowClass",
    "ApplicationFrameWindow",
    "CEF-OSC-WIDGET",
    "Qt51514QWindowIcon",
}


def is_browser_window(info: WindowInfo) -> bool:
    return info.class_name in BROWSER_CLASSES


def find_window(title: str = "", class_name: str = "") -> WindowInfo | None:
    windows = list_windows()
    if title:
        for w in windows:
            if w.title == title and (not class_name or w.class_name == class_name):
                return w
        for w in windows:
            if title.lower() in w.title.lower() and (not class_name or w.class_name == class_name):
                return w
    if class_name:
        for w in windows:
            if w.class_name == class_name:
                return w
    # 退而求其次：当前前台窗口
    hwnd = user32.GetForegroundWindow()
    info = get_window_info(hwnd) if hwnd else None
    if info and info.width >= 700 and info.height >= 500:
        return info
    for w in windows:
        if is_browser_window(w):
            return w
    return windows[0] if windows else None


def is_foreground(hwnd: int) -> bool:
    fg = user32.GetForegroundWindow()
    if not fg:
        return False
    if int(fg) == int(hwnd):
        return True
    try:
        return bool(user32.IsChild(hwnd, fg))
    except Exception:
        return False


def focus_window(hwnd: int) -> bool:
    if not hwnd or not user32.IsWindow(hwnd):
        return False
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    if is_foreground(hwnd):
        return True
    fg = user32.GetForegroundWindow()
    current_tid = kernel32.GetCurrentThreadId()
    target_tid = user32.GetWindowThreadProcessId(hwnd, None)
    fg_tid = user32.GetWindowThreadProcessId(fg, None) if fg else 0
    attached: list[int] = []
    try:
        if fg_tid and fg_tid != current_tid:
            if user32.AttachThreadInput(current_tid, fg_tid, True):
                attached.append(fg_tid)
        if target_tid and target_tid != current_tid and target_tid not in attached:
            if user32.AttachThreadInput(current_tid, target_tid, True):
                attached.append(target_tid)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetActiveWindow(hwnd)
        time.sleep(0.15)
    except Exception:
        pass
    finally:
        for tid in attached:
            try:
                user32.AttachThreadInput(current_tid, tid, False)
            except Exception:
                pass
    return is_foreground(hwnd)


def minimize_window(hwnd: int) -> None:
    if hwnd and user32.IsWindow(hwnd):
        user32.ShowWindow(hwnd, SW_MINIMIZE)


def maximize_window(hwnd: int) -> None:
    if not hwnd or not user32.IsWindow(hwnd):
        return
    if not user32.IsZoomed(hwnd):
        user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
        time.sleep(0.35)


def cursor_position() -> tuple[int, int]:
    pt = wintypes.POINT()
    if user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return -1, -1


def _mount_mouse_input(flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> INPUT:
    inp = INPUT()
    inp.type = 0  # INPUT_MOUSE
    inp.mi = MOUSEINPUT(dx, dy, data, flags, 0, 0)
    return inp


def _send_mouse_inputs(inputs: list[INPUT]) -> int:
    if not inputs:
        return 0
    arr = (INPUT * len(inputs))(*inputs)
    return int(user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT)))


def _send_absolute_move(x: int, y: int) -> bool:
    vx = user32.GetSystemMetrics(76)  # SM_XVIRTUALSCREEN
    vy = user32.GetSystemMetrics(77)
    vw = max(1, user32.GetSystemMetrics(78))
    vh = max(1, user32.GetSystemMetrics(79))
    nx = int(round((int(x) - vx) * 65535 / max(1, vw - 1)))
    ny = int(round((int(y) - vy) * 65535 / max(1, vh - 1)))
    flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
    return _send_mouse_inputs([_mount_mouse_input(flags, nx, ny)]) == 1


class Clicker:
    """物理像素坐标点击。进程必须已经开启 DPI 感知。"""

    def __init__(self, dry_run: bool = False, delay: float = 0.04):
        self.dry_run = bool(dry_run)
        self.delay = max(0.01, float(delay))
        self.last_before = (-1, -1)
        self.last_after = (-1, -1)
        self.last_method = "none"
        self.last_error = ""

    def move(self, x: int, y: int) -> bool:
        x, y = int(x), int(y)
        ok = bool(user32.SetCursorPos(x, y))
        time.sleep(0.025)
        ax, ay = cursor_position()
        if abs(ax - x) <= 2 and abs(ay - y) <= 2:
            self.last_method = "SetCursorPos"
            return True
        # 光标没动，尝试 SendInput 绝对移动（可能 SetCursorPos 被策略拦截）
        sent = _send_absolute_move(x, y)
        time.sleep(0.03)
        ax, ay = cursor_position()
        if sent and abs(ax - x) <= 3 and abs(ay - y) <= 3:
            self.last_method = "SendInput-Move"
            return True
        self.last_error = f"SetCursorPos={ok}, SendInput={sent}, cursor=({ax},{ay}), target=({x},{y})"
        return False

    def click(self, x: int, y: int) -> None:
        if self.dry_run:
            time.sleep(0.03)
            return
        x, y = int(x), int(y)
        self.last_before = cursor_position()
        self.last_after = self.last_before
        self.last_method = "none"
        self.last_error = ""
        if not self.move(x, y):
            raise MouseInputError(f"鼠标无法移动到目标坐标：{self.last_error}")
        time.sleep(0.03)
        down = _mount_mouse_input(MOUSEEVENTF_LEFTDOWN)
        up = _mount_mouse_input(MOUSEEVENTF_LEFTUP)
        sent = _send_mouse_inputs([down, up])
        if sent != 2:
            # 旧 API 兜底，并记录错误供日志诊断
            self.last_error = f"SendInput 点击返回 {sent}/2，已回退 mouse_event"
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.03)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            self.last_method += "+mouse_event"
        else:
            self.last_method += "+SendInput"
        time.sleep(self.delay)
        self.last_after = cursor_position()

    def scroll(self, x: int, y: int, notches: int) -> None:
        if self.dry_run:
            return
        if not self.move(x, y):
            raise MouseInputError(f"滚轮定位失败：{self.last_error}")
        time.sleep(0.02)
        delta = int(120 * notches)
        sent = _send_mouse_inputs([_mount_mouse_input(MOUSEEVENTF_WHEEL, 0, 0, delta)])
        if sent != 1:
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, delta, 0)
        time.sleep(0.08)

    def type_text(self, text: str) -> None:
        if self.dry_run or not text:
            return
        _set_clipboard_text(text)
        time.sleep(0.1)
        user32.keybd_event(0x11, 0, 0, 0)  # Ctrl down
        user32.keybd_event(0x56, 0, 0, 0)  # V down
        time.sleep(0.05)
        user32.keybd_event(0x56, 0, KEYEVENTF_KEYUP, 0)
        user32.keybd_event(0x11, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.25)


def _set_clipboard_text(text: str) -> None:
    if not user32.OpenClipboard(None):
        return
    try:
        user32.EmptyClipboard()
        data = text.encode("utf-16-le") + b"\x00\x00"
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
        if not handle:
            return
        ptr = kernel32.GlobalLock(handle)
        if not ptr:
            kernel32.GlobalFree(handle)
            return
        ctypes.memmove(ptr, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            kernel32.GlobalFree(handle)
    finally:
        user32.CloseClipboard()


class HotkeyListener(threading.Thread):
    """在独立线程里注册 F8 全局热键，用于紧急停止。"""

    def __init__(self, callback: Callable[[], None], vk: int = VK_F8):
        super().__init__(daemon=True, name="hotkey-listener")
        self.callback = callback
        self.vk = vk
        self.thread_id = 0
        self.ready = threading.Event()
        self.ok = False
        self._stop = threading.Event()

    def run(self) -> None:
        self.thread_id = kernel32.GetCurrentThreadId()
        self.ok = bool(user32.RegisterHotKey(None, 1, 0, self.vk))
        self.ready.set()
        if not self.ok:
            return
        msg = wintypes.MSG()
        try:
            while not self._stop.is_set():
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret <= 0:
                    break
                if msg.message == 0x0312:  # WM_HOTKEY
                    try:
                        self.callback()
                    except Exception:
                        pass
        finally:
            user32.UnregisterHotKey(None, 1)

    def stop(self) -> None:
        self._stop.set()
        if self.thread_id:
            try:
                user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)  # WM_QUIT
            except Exception:
                pass


APP_USER_MODEL_ID = "CourseBot.AutoStudy.1"


def set_app_user_model_id() -> None:
    try:
        shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass


def foreground_candidates(windows: Iterable[WindowInfo]) -> list[WindowInfo]:
    fg = user32.GetForegroundWindow()
    return [w for w in windows if int(w.hwnd) == int(fg)]
