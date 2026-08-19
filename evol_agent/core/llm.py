from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from .config import (
    LLM_CALL_MAX_ATTEMPTS,
    LLM_CALL_RETRY_DELAYS_SECONDS,
    LLM_CALL_TIMEOUT_SECONDS,
)
from .run_recorder import append_run_event

logger = logging.getLogger(__name__)


_PROMPT_STRUCTURAL_PREFIXES = (
    "SCHEMA ",
    "MATCH ",
    "SELECTOR ",
    "R ",
    "#",
    "* ",
)


def _unwrap_prompt_block(block: str) -> str:
    lines = [line.rstrip() for line in block.splitlines()]
    if len(lines) <= 1:
        return block
    stripped = [line.strip() for line in lines]
    if re.match(r"^\d+\. [^.!?]{1,80}$", stripped[0]):
        return stripped[0] + "\n" + _unwrap_prompt_block("\n".join(lines[1:]))
    if any(
        line.startswith(_PROMPT_STRUCTURAL_PREFIXES)
        or line.startswith(("{", "}", "[", "]", '"'))
        for line in stripped
    ):
        return "\n".join(lines)

    output: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            output.append(current)
            current = ""

    for line in stripped:
        is_list_item = bool(re.match(r"^(?:- |\d+\. )", line))
        is_short_heading = bool(
            re.match(r"^\d+\. [^.!?]{1,80}$", line)
            or (line.endswith(":") and len(line) <= 80)
        )
        if is_list_item or is_short_heading:
            flush()
            current = line
        elif current:
            current = f"{current} {line}".strip()
        else:
            current = line
    flush()
    return "\n".join(output)


def normalize_prompt_layout(prompt: str) -> str:
    """Remove source-code soft wraps while preserving real prompt structure."""
    blocks = re.split(r"\n[ \t]*\n", str(prompt))
    return "\n\n".join(_unwrap_prompt_block(block) for block in blocks)


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    json_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _usage_payload(usage: Any) -> dict[str, Any]:
    """Keep provider usage, including cache fields, in a JSON-safe shape."""
    if isinstance(usage, dict):
        return usage
    for method_name in ("model_dump", "dict"):
        method = getattr(usage, method_name, None)
        if callable(method):
            try:
                value = method()
                return value if isinstance(value, dict) else {}
            except Exception:  # pragma: no cover - provider-specific object
                pass
    return {}


def call_json_llm(
    prompt: str | list[dict[str, str]],
    *,
    model: str,
    system: str = "You are an expert SC2 bot analyst and strategy designer. Output valid JSON only.",
    is_reasoning: bool = False,
) -> Optional[dict[str, Any]]:
    from llm.caller import call_openai_detailed

    if isinstance(prompt, str):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": normalize_prompt_layout(prompt)},
        ]
    else:
        messages = [
            {
                **message,
                "content": normalize_prompt_layout(str(message.get("content") or "")),
            }
            for message in prompt
        ]
    last_error = ""
    for attempt in range(1, LLM_CALL_MAX_ATTEMPTS + 1):
        response = ""
        try:
            response_data = call_openai_detailed(
                messages=messages,
                model_key=model,
                is_reasoning=is_reasoning,
                response_format={"type": "json_object"},
                timeout=LLM_CALL_TIMEOUT_SECONDS,
            )
            response = str(response_data.get("content") or "")
            if response_data.get("error"):
                raise RuntimeError(str(response_data["error"]))
            if not response.strip():
                raise RuntimeError("empty response from LLM provider")
            parsed = extract_json_object(response.strip())
            append_run_event(
                "llm_call",
                {
                    "model": model,
                    "is_reasoning": is_reasoning,
                    "attempt": attempt,
                    "messages": messages,
                    "raw_response": response,
                    "parsed_response": parsed,
                    "usage": _usage_payload(response_data.get("usage")),
                    "latency_seconds": response_data.get("latency_seconds"),
                },
            )
            return parsed
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = str(exc)
            append_run_event(
                "llm_call_attempt_failed",
                {
                    "model": model,
                    "is_reasoning": is_reasoning,
                    "attempt": attempt,
                    "max_attempts": LLM_CALL_MAX_ATTEMPTS,
                    "error": last_error,
                    "raw_response": response,
                    "response_chars": len(response),
                },
            )
            if attempt < LLM_CALL_MAX_ATTEMPTS:
                delay = LLM_CALL_RETRY_DELAYS_SECONDS[
                    min(attempt - 1, len(LLM_CALL_RETRY_DELAYS_SECONDS) - 1)
                ]
                logger.warning(
                    "LLM JSON call failed on attempt %s/%s; retrying in %.1fs (%s)",
                    attempt,
                    LLM_CALL_MAX_ATTEMPTS,
                    delay,
                    exc,
                )
                try:
                    time.sleep(delay)
                except KeyboardInterrupt:
                    raise

    append_run_event(
        "llm_call_failed",
        {
            "model": model,
            "is_reasoning": is_reasoning,
            "attempts": LLM_CALL_MAX_ATTEMPTS,
            "messages": messages,
            "error": last_error,
        },
    )
    logger.error("LLM JSON call failed after %s attempts: %s", LLM_CALL_MAX_ATTEMPTS, last_error)
    return None
