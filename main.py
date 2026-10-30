from __future__ import annotations

import ctypes
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 某些机器把 SSLKEYLOGFILE 设成了 Wireshark/netdump 的日志路径。
# Python 3.13 的 ssl/urllib3 在创建 TLS 上下文时会尝试写这个文件；
# 如果文件正被占用，程序会在导入 requests 时直接 PermissionError。
# 本程序不需要导出 TLS 密钥，启动时清掉，避免无关的环境变量导致启动失败。
os.environ.pop("SSLKEYLOGFILE", None)

from coursebot.windows import enable_dpi_awareness, set_app_user_model_id


def show_error(title: str, message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
    except Exception:
        print(f"{title}: {message}")


def main() -> int:
    if sys.platform != "win32":
        show_error("不支持的系统", "本程序仅支持 Windows 10/11。")
        return 1
    enable_dpi_awareness()
    set_app_user_model_id()

    from coursebot.config import BASE_DIR, LOG_DIR, Settings, ensure_dirs

    ensure_dirs()
    try:
        settings = Settings.load()
    except Exception as exc:
        show_error("配置加载失败", f"无法读取配置：{exc}")
        settings = Settings()

    try:
        from coursebot.gui import run_app

        run_app(settings)
        return 0
    except Exception as exc:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"crash_{datetime.now():%Y%m%d_%H%M%S}.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        show_error(
            "程序启动失败",
            f"{type(exc).__name__}: {exc}\n\n详细日志：{log_path}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
