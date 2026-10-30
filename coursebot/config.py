from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

if getattr(sys, "frozen", False):
    # PyInstaller 打包后，配置和日志放到 exe 同级目录，而不是 _internal
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.json"
LOG_DIR = BASE_DIR / "logs"

# 不内置任何密钥；请通过 GUI、config.json 或 COURSEBOT_API_KEY / ARK_API_KEY 环境变量设置。
DEFAULT_API_KEY = ""
DEFAULT_API_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DEFAULT_MODEL = "glm-5-2-260617"


@dataclass
class Settings:
    # ---- 目标窗口 ----
    window_title: str = ""
    window_class: str = ""

    # ---- LLM ----
    llm_mode: str = "cloud"  # cloud | local
    api_url: str = DEFAULT_API_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    temperature: float = 0.1
    max_tokens: int = 2000
    llm_timeout: int = 90
    enable_thinking: bool = False

    # 本地模式（Ollama 原生 /api/chat，或 OpenAI 兼容 /v1/chat/completions）
    local_api_url: str = "http://127.0.0.1:11434/api/chat"
    local_model: str = "qwen3:4b"
    local_timeout: int = 180

    # ---- 自动化 ----
    poll_interval: float = 0.3
    full_scan_interval: float = 25.0
    completion_margin: float = 2.0
    completion_confirmations: int = 2
    stall_seconds: float = 8.0
    max_resume_attempts: int = 4
    resume_cooldown: float = 8.0
    auto_resume: bool = True
    auto_quiz: bool = True
    auto_next: bool = True
    max_sections: int = 50
    start_countdown: int = 3
    nav_wait_seconds: float = 1.5
    max_nav_attempts: int = 3
    quiz_retry: int = 1
    max_quiz_questions: int = 20

    # ---- 行为 / 界面 ----
    dry_run: bool = False
    minimize_on_start: bool = True
    always_on_top: bool = False
    save_debug_frames: bool = True
    preview_enabled: bool = True
    show_ocr_boxes: bool = True

    # ---- 内部 ----
    ocr_max_width: int = 2400

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        data: dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            except Exception:
                try:
                    path.replace(path.with_suffix(".broken.json"))
                except Exception:
                    pass

        # 环境变量优先，便于不把密钥写入文件。
        env_key = os.environ.get("COURSEBOT_API_KEY") or os.environ.get("ARK_API_KEY")
        if env_key:
            data["api_key"] = env_key

        obj = cls()
        valid = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key not in valid:
                continue
            current = getattr(obj, key)
            try:
                if isinstance(current, bool):
                    setattr(obj, key, bool(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(obj, key, int(value))
                elif isinstance(current, float):
                    setattr(obj, key, float(value))
                elif isinstance(current, str):
                    setattr(obj, key, str(value))
            except (TypeError, ValueError):
                continue
        return obj

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dirs() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def masked_key(key: str) -> str:
    key = key.strip()
    if not key:
        return "(未设置)"
    if len(key) <= 12:
        return key[:3] + "***"
    return f"{key[:8]}…{key[-4:]}"
