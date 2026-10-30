from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from .config import Settings


class LLMError(RuntimeError):
    pass


@dataclass
class LLMAnswer:
    letters: list[str]
    answer_text: str
    confidence: float
    reason: str
    raw: str
    latency: float = 0.0


SYSTEM_PROMPT = """你是在线课程答题助手。请根据题目和选项判断正确答案。
必须只输出一个严格 JSON 对象，不要输出 Markdown 代码块，不要输出 JSON 以外的解释。
JSON 格式：
{"answer":["A"],"answer_text":"","confidence":0.95,"reason":"不超过30字"}
规则：
1. 单选题、多选题、判断题使用 answer 数组填写选项字母；多选题按字母顺序填写多个。
2. 选项只有“正确/错误”或“对/错”时，正确选 A，错误选 B。
3. 填空题、简答题把答案写在 answer_text 字段，answer 留空数组。
4. 即使不确定，也必须给出最可能的答案，不要拒绝作答。
5. 你的最终回复必须能被 json.loads 直接解析。"""


class LLMClient:
    def __init__(self, settings: Settings):
        self.mode = str(getattr(settings, "llm_mode", "cloud") or "cloud").lower()
        self.local = self.mode == "local"
        if self.local:
            self.url = str(getattr(settings, "local_api_url", "") or "").strip()
            self.key = ""
            self.model = str(getattr(settings, "local_model", "") or "").strip()
            self.timeout = int(getattr(settings, "local_timeout", 180) or 180)
        else:
            self.url = settings.api_url.strip()
            self.key = settings.api_key.strip()
            self.model = settings.model.strip()
            self.timeout = int(settings.llm_timeout)
        self.temperature = float(settings.temperature)
        self.max_tokens = int(settings.max_tokens)
        self.enable_thinking = bool(getattr(settings, "enable_thinking", False))
        self.session = requests.Session()

    @property
    def label(self) -> str:
        if self.local:
            return f"本地 {self.model} @ {self.url}"
        return f"{self.model} / Key {self.key[:8] + '…' + self.key[-4:] if len(self.key) > 12 else '(未设置)'}"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if not self.enable_thinking:
            # 火山方舟的 GLM 模型支持关闭深度思考；答题场景下更快、更稳定。
            payload["thinking"] = {"type": "disabled"}
        return payload

    def _post(self, payload: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        if not self.url:
            raise LLMError("未配置 LLM API 地址")
        if not self.key:
            raise LLMError("未配置 LLM API Key")
        if not self.model:
            raise LLMError("未配置模型名称")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}",
        }
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.post(
                    self.url,
                    headers=headers,
                    json=payload,
                    timeout=(10, max(15, self.timeout)),
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise LLMError(f"LLM 网络请求失败：{type(exc).__name__}") from exc
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            if resp.status_code == 400 and "thinking" in payload:
                # 非方舟 / 非 GLM 模型可能不认识 thinking 参数，去掉后重试
                payload = {k: v for k, v in payload.items() if k != "thinking"}
                continue
            if resp.status_code != 200:
                detail = resp.text[:500].replace(self.key, "***")
                raise LLMError(f"LLM 接口返回 HTTP {resp.status_code}：{detail}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise LLMError(f"LLM 返回不是 JSON：{resp.text[:300]}") from exc
            choices = data.get("choices") or []
            if not choices:
                raise LLMError(f"LLM 返回缺少 choices：{str(data)[:300]}")
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            reasoning = str(message.get("reasoning_content") or "").strip()
            usage = data.get("usage") or {}
            return content, reasoning, usage
        raise LLMError(f"LLM 请求失败：{last_error}")

    def _native_ollama(self) -> bool:
        return "/api/chat" in self.url or (
            ":11434" in self.url and "/v1/" not in self.url
        )

    def _post_local(
        self, system: str, user: str, json_mode: bool = True
    ) -> tuple[str, str, dict[str, Any]]:
        if not self.url:
            raise LLMError("未配置本地 LLM 接口地址")
        if not self.model:
            raise LLMError("未配置本地模型名称")
        local_user = user
        if "qwen" in self.model.lower() and "/no_think" not in local_user:
            # Qwen3 默认会输出大段思考内容；/no_think 让它直接作答。
            local_user = "/no_think\n" + local_user
        local_user += "\n\n只输出一个 JSON 对象，不要重复内容，不要输出 Markdown。"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": local_user},
        ]
        native = self._native_ollama()
        if native:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": max(0.05, min(0.7, self.temperature)),
                    # 本地模型偶尔会陷入重复输出，限制长度并加重复惩罚。
                    "num_predict": min(self.max_tokens, 512),
                    "repeat_penalty": 1.12,
                    "repeat_last_n": 128,
                    "top_p": 0.9,
                },
            }
            if json_mode:
                payload["format"] = "json"
        else:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": max(0.05, min(0.7, self.temperature)),
                "max_tokens": min(self.max_tokens, 512),
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1,
                "stream": False,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.session.post(
                    self.url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=(5, max(15, self.timeout)),
                )
            except requests.RequestException as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise LLMError(
                    f"本地 LLM 连接失败：{type(exc).__name__}，请确认 Ollama 已启动"
                ) from exc
            if resp.status_code == 400 and native and (
                "think" in payload or "format" in payload
            ):
                # 老版本 Ollama 不认识 think/format 参数，去掉后重试一次。
                payload.pop("think", None)
                payload.pop("format", None)
                continue
            if (
                resp.status_code == 400
                and not native
                and "response_format" in payload
            ):
                payload.pop("response_format", None)
                continue
            if (
                resp.status_code == 500
                and "repeat" in resp.text.lower()
                and attempt < 2
            ):
                # “token repeat limit reached”：提高重复惩罚、缩短输出后重试。
                if native:
                    options = payload.setdefault("options", {})
                    options["repeat_penalty"] = 1.32 + attempt * 0.12
                    options["num_predict"] = min(320, int(options.get("num_predict", 320)))
                    options["temperature"] = 0.35
                else:
                    payload["frequency_penalty"] = 0.5
                    payload["max_tokens"] = min(320, int(payload.get("max_tokens", 320)))
                    payload["temperature"] = 0.35
                time.sleep(0.25)
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise LLMError(
                    f"本地 LLM 返回 HTTP {resp.status_code}：{resp.text[:500]}"
                )
            try:
                data = resp.json()
            except ValueError as exc:
                raise LLMError(f"本地 LLM 返回不是 JSON：{resp.text[:300]}") from exc
            if native:
                message = data.get("message") or {}
                content = str(message.get("content") or "").strip()
                reasoning = str(message.get("thinking") or "").strip()
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                }
                if not content and data.get("done_reason") == "length":
                    raise LLMError(
                        "本地模型输出被 max_tokens 截断，请增大最大输出或换更小模型"
                    )
            else:
                choices = data.get("choices") or []
                if not choices:
                    raise LLMError(f"本地 LLM 返回缺少 choices：{str(data)[:300]}")
                message = choices[0].get("message") or {}
                content = str(message.get("content") or "").strip()
                reasoning = str(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ).strip()
                usage = data.get("usage") or {}
            return content, reasoning, usage
        raise LLMError(f"本地 LLM 请求失败：{last_error}")

    def _chat(
        self, system: str, user: str, json_mode: bool = True
    ) -> tuple[str, str, dict[str, Any]]:
        if self.local:
            return self._post_local(system, user, json_mode=json_mode)
        return self._post(self._payload(system, user))

    def test_connection(self) -> tuple[str, float]:
        start = time.time()
        if self.local:
            content, reasoning, _ = self._post_local(
                "你只输出JSON，不要解释。",
                '只输出 {"ok": true} 这个JSON，不要输出其他内容。',
                json_mode=True,
            )
        else:
            payload = self._payload(
                "你是一个连通性测试助手，请严格按要求回复。",
                "只回复四个字：连接成功",
            )
            content, reasoning, _ = self._post(payload)
        latency = time.time() - start
        reply = (content or reasoning).strip()
        if not reply:
            raise LLMError("接口已响应，但回复内容为空（可能 max_tokens 太小）")
        return reply[:500], latency

    def solve(
        self,
        question: str,
        options: dict[str, str] | list[tuple[str, str]],
        kind: str = "选择题",
        feedback: str = "",
        previous_letters: list[str] | None = None,
    ) -> LLMAnswer:
        if isinstance(options, dict):
            pairs = list(options.items())
        else:
            pairs = list(options)
        option_text = "\n".join(f"{letter}. {text}" for letter, text in pairs)
        user = f"题型：{kind}\n题目：\n{question.strip()}\n\n选项：\n{option_text}"
        if not pairs:
            user = f"题型：{kind}\n题目：\n{question.strip()}\n\n（页面没有识别到选项）"
        if feedback:
            user += f"\n\n上一轮提交后的页面反馈：\n{feedback[:1200]}\n请据此重新判断。"
        if previous_letters:
            user += f"\n上一轮选择：{', '.join(previous_letters)}"
        user += "\n\n请严格按系统要求输出 JSON。"
        start = time.time()
        content, reasoning, _ = self._chat(SYSTEM_PROMPT, user, json_mode=True)
        latency = time.time() - start
        if not content and reasoning:
            content = reasoning
        if not content:
            raise LLMError("LLM 返回内容为空，可能输出被 max_tokens 截断")
        answer = parse_llm_answer(content, {p[0] for p in pairs})
        answer.latency = latency
        answer.raw = content
        return answer


def list_local_models(url: str, timeout: int = 8) -> list[str]:
    """查询 Ollama /api/tags 或 OpenAI 兼容 /v1/models 的本地模型列表。"""
    url = (url or "").strip()
    if not url:
        raise LLMError("未配置本地接口地址")
    candidates: list[str] = []
    if "/api/chat" in url:
        candidates.append(url.split("/api/chat", 1)[0] + "/api/tags")
    if "/v1/chat/completions" in url:
        candidates.append(url.split("/v1/chat/completions", 1)[0] + "/v1/models")
    base = url.rstrip("/")
    candidates.extend([base + "/api/tags", base + "/v1/models"])
    seen: set[str] = set()
    last_error = ""
    for endpoint in candidates:
        if endpoint in seen:
            continue
        seen.add(endpoint)
        try:
            resp = requests.get(endpoint, timeout=timeout)
            if resp.status_code != 200:
                last_error = f"{endpoint} HTTP {resp.status_code}"
                continue
            data = resp.json()
        except Exception as exc:
            last_error = f"{endpoint} {type(exc).__name__}"
            continue
        models: list[str] = []
        for item in data.get("models") or []:
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                models.append(name)
        for item in data.get("data") or []:
            name = str(item.get("id") or "").strip()
            if name:
                models.append(name)
        if models:
            return sorted(set(models))
    raise LLMError(
        "未发现本地模型。请确认 Ollama 已运行并已 pull 模型；"
        + (f"最后错误：{last_error}" if last_error else "")
    )


def _find_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def parse_llm_answer(text: str, valid_letters: set[str] | None = None) -> LLMAnswer:
    valid = {x.upper() for x in (valid_letters or set("ABCDEFGH"))}
    obj = _find_json_object(text)
    answer_field: Any = None
    answer_text = ""
    confidence = 0.0
    reason = ""
    if obj:
        answer_field = obj.get("answer", obj.get("answers", obj.get("选项", "")))
        answer_text = str(obj.get("answer_text") or obj.get("text") or "").strip()
        try:
            confidence = float(obj.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        reason = str(obj.get("reason") or obj.get("解析") or "").strip()

    letters: list[str] = []
    raw_answer = ""
    if isinstance(answer_field, (list, tuple)):
        raw_answer = "".join(str(x) for x in answer_field)
    elif answer_field is not None:
        raw_answer = str(answer_field)
    for token in re.findall(r"[A-H]", raw_answer.upper()):
        if token in valid and token not in letters:
            letters.append(token)

    if not letters and not answer_text:
        # 兜底：从“答案：A”或最终一两行中提取，避免模型带了 Markdown
        matches = re.findall(r"(?:答案|正确选项|answer)\s*[:：]?\s*[\[\"'（(]*([A-H](?:\s*[,，、/]\s*[A-H])*)", text, re.I)
        if matches:
            for token in re.findall(r"[A-H]", matches[-1].upper()):
                if token in valid and token not in letters:
                    letters.append(token)
        if not letters:
            tail = text.strip().splitlines()[-3:]
            tail_text = "\n".join(tail)
            if re.fullmatch(r"[\sA-H,，、/\[\]\"']+", tail_text):
                for token in re.findall(r"[A-H]", tail_text.upper()):
                    if token in valid and token not in letters:
                        letters.append(token)

    letters.sort()
    if not letters and not answer_text:
        raise LLMError(f"无法从模型回复中解析答案：{text[:300]}")
    return LLMAnswer(
        letters=letters,
        answer_text=answer_text,
        confidence=confidence,
        reason=reason,
        raw=text,
    )
