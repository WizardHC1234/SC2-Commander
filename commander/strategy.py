"""Parse strategy.md into summary/detail."""

from __future__ import annotations

import re
from typing import Dict

_SUMMARY_HEADER_RE = re.compile(
    r"^\s*#\s*(?:摘要|Summary|Abstract)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_DETAIL_HEADER_RE = re.compile(
    r"^\s*#\s*(?:详细内容|Detail|Details|Full|Content)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
def parse_strategy_document(text: str) -> Dict[str, str]:
    if not text:
        return {"summary": "", "detail": ""}
    raw = text.strip()
    summary_match = _SUMMARY_HEADER_RE.search(raw)
    detail_match = _DETAIL_HEADER_RE.search(raw)
    summary = ""
    detail = ""

    if summary_match and detail_match:
        if summary_match.start() < detail_match.start():
            summary = raw[summary_match.end() : detail_match.start()].strip()
            detail = raw[detail_match.end() :].strip()
        else:
            detail = raw[detail_match.end() : summary_match.start()].strip()
            summary = raw[summary_match.end() :].strip()
    elif summary_match:
        summary = raw[summary_match.end() :].strip()
        detail = raw
    elif detail_match:
        detail = raw[detail_match.end() :].strip()
        summary = _fallback_summary_from_detail(detail)
    else:
        detail = raw
        summary = _fallback_summary_from_detail(detail)
    return {"summary": summary, "detail": detail}


def _fallback_summary_from_detail(detail: str) -> str:
    if not detail:
        return ""
    for paragraph in detail.split("\n\n"):
        line = paragraph.strip()
        if not line:
            continue
        line = re.sub(r"^[#>*\-\+\s]+", "", line).strip()
        if line:
            return line[:500]
    return detail.strip()[:500]
