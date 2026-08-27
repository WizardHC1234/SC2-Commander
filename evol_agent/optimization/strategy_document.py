from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from evol_agent.core.config import REQUIRED_TOP_HEADINGS, STRUCTURED_TOP_HEADINGS


_FIELD_LINE = re.compile(r"^[*-]\s+([^:\n]+):\s+(\S.*)$")
_LEGACY_HEADINGS = ("# Summary", "# Details")


def paragraph_id(title: str) -> str:
    """Return the stable identifier used by candidate patch operations."""
    return re.sub(r"[^a-z0-9]+", "_", str(title).strip().lower()).strip("_")


def paragraph_hash(value: str) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class StrategyDetail:
    id: str
    title: str
    value: str
    section: str = "Details"


@dataclass
class StrategyDocument:
    """Parsed strategy document with legacy read compatibility.

    New strategies use Summary/Details. Goal/Macro/Combat remains readable only
    for checkpoints created during the retired three-section experiment.
    """

    summary: str
    details: list[StrategyDetail]
    format_name: str = "legacy"

    @classmethod
    def parse(cls, text: str) -> "StrategyDocument":
        source = str(text or "").strip()
        headings = [
            line.strip()
            for line in source.splitlines()
            if line.startswith("# ")
        ]
        if headings == REQUIRED_TOP_HEADINGS:
            return cls._parse_legacy(source)
        if headings == STRUCTURED_TOP_HEADINGS:
            return cls._parse_goal_macro_combat(source)
        raise ValueError(
            "strategy.md must use # Summary, # Details in that order"
        )

    @classmethod
    def _parse_goal_macro_combat(cls, source: str) -> "StrategyDocument":
        details: list[StrategyDetail] = []
        seen: set[str] = set()
        sections: dict[str, list[StrategyDetail]] = {}
        for index, heading in enumerate(STRUCTURED_TOP_HEADINGS):
            section_name = heading[2:]
            start = source.index(heading) + len(heading)
            end = (
                source.index(STRUCTURED_TOP_HEADINGS[index + 1])
                if index + 1 < len(STRUCTURED_TOP_HEADINGS)
                else len(source)
            )
            section_items: list[StrategyDetail] = []
            for line in source[start:end].splitlines():
                if not line.strip():
                    continue
                match = _FIELD_LINE.fullmatch(line.strip())
                if not match:
                    raise ValueError(
                        f"{heading} lines must use '- Field: instruction'"
                    )
                title, value = match.groups()
                field_id = paragraph_id(title)
                if not field_id:
                    raise ValueError(f"strategy field has no stable id: {title!r}")
                if field_id in seen:
                    raise ValueError(f"duplicate strategy field id: {field_id}")
                seen.add(field_id)
                item = StrategyDetail(
                    field_id,
                    title.strip(),
                    value.strip(),
                    section_name,
                )
                section_items.append(item)
                details.append(item)
            if not section_items:
                raise ValueError(f"{heading} section is empty")
            sections[section_name] = section_items

        goal_summary = "\n".join(
            f"- {item.title}: {item.value}" for item in sections["Goal"]
        )
        return cls(
            summary=goal_summary,
            details=details,
            format_name="goal_macro_combat",
        )

    @classmethod
    def _parse_legacy(cls, source: str) -> "StrategyDocument":
        summary_heading, details_heading = _LEGACY_HEADINGS
        details_at = source.find(f"\n{details_heading}")
        if details_at < 0:
            raise ValueError("strategy.md is missing # Details")
        summary = source[len(summary_heading):details_at].strip()
        detail_text = source[details_at + len(details_heading) + 1 :].strip()
        if not summary:
            raise ValueError("# Summary section is empty")

        details: list[StrategyDetail] = []
        seen: set[str] = set()
        for line in detail_text.splitlines():
            if not line.strip():
                continue
            match = _FIELD_LINE.fullmatch(line.strip())
            if not match:
                raise ValueError(
                    "# Details lines must use the form '* Title: instruction'"
                )
            title, value = match.groups()
            field_id = paragraph_id(title)
            if not field_id:
                raise ValueError(f"strategy detail has no stable id: {title!r}")
            if field_id in seen:
                raise ValueError(f"duplicate strategy detail id: {field_id}")
            seen.add(field_id)
            details.append(
                StrategyDetail(field_id, title.strip(), value.strip(), "Details")
            )
        if not details:
            raise ValueError("# Details section is empty")
        return cls(summary=summary, details=details, format_name="legacy")

    def render(self) -> str:
        if self.format_name == "legacy":
            lines = ["# Summary", "", self.summary.strip(), "", "# Details", ""]
            lines.extend(f"* {item.title}: {item.value}" for item in self.details)
            return "\n".join(lines).strip() + "\n"

        lines: list[str] = []
        for heading in STRUCTURED_TOP_HEADINGS:
            section_name = heading[2:]
            lines.extend([heading, ""])
            section_items = [
                item for item in self.details if item.section == section_name
            ]
            lines.extend(f"- {item.title}: {item.value}" for item in section_items)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def patch_context(self) -> dict[str, Any]:
        return {
            "format": self.format_name,
            "summary": {
                "id": "summary",
                "hash": paragraph_hash(self.summary),
                "value": self.summary,
            },
            "details": [
                {
                    "id": item.id,
                    "section": item.section,
                    "title": item.title,
                    "hash": paragraph_hash(item.value),
                    "value": item.value,
                }
                for item in self.details
            ],
        }

    def apply_patch(
        self,
        operations: Any,
    ) -> tuple[str, list[dict[str, str]]]:
        if not isinstance(operations, list) or not operations:
            raise ValueError("candidate operations must be a non-empty list")

        detail_by_id = {item.id: item for item in self.details}
        replacements: dict[str, str] = {}
        summary_replacement: str | None = None
        seen_targets: set[str] = set()
        changes: list[dict[str, str]] = []

        for raw in operations:
            if not isinstance(raw, dict):
                raise ValueError("each candidate operation must be an object")
            operation = str(raw.get("op") or "").strip()
            target = str(raw.get("target") or "").strip()
            value = str(raw.get("value") or "").strip()
            expected_hash = str(raw.get("expected_old_hash") or "").strip()
            if target in seen_targets:
                raise ValueError(f"candidate modifies paragraph {target!r} more than once")
            seen_targets.add(target)
            if not value or "\n" in value:
                raise ValueError(
                    f"candidate paragraph {target!r} must be one non-empty line"
                )

            if operation == "replace_summary":
                if self.format_name != "legacy":
                    raise ValueError("Goal fields must be replaced individually")
                if target not in ("", "summary"):
                    raise ValueError("replace_summary target must be 'summary'")
                if expected_hash and expected_hash != paragraph_hash(self.summary):
                    raise ValueError("summary precondition hash does not match parent")
                if value == self.summary:
                    raise ValueError("summary replacement does not change the parent")
                summary_replacement = value
                changes.append({"op": operation, "target": "summary"})
                continue

            if operation != "replace_detail":
                raise ValueError(f"unsupported candidate operation: {operation!r}")
            current = detail_by_id.get(target)
            if current is None:
                allowed = ", ".join(item.id for item in self.details)
                raise ValueError(
                    f"unknown strategy detail {target!r}; allowed targets: {allowed}"
                )
            if expected_hash and expected_hash != paragraph_hash(current.value):
                raise ValueError(
                    f"paragraph {target!r} precondition hash does not match parent"
                )
            if value == current.value:
                raise ValueError(f"paragraph {target!r} replacement is unchanged")
            replacements[target] = value
            changes.append({"op": operation, "target": target})

        patched_details = [
            StrategyDetail(
                item.id,
                item.title,
                replacements.get(item.id, item.value),
                item.section,
            )
            for item in self.details
        ]
        patched_summary = summary_replacement or self.summary
        if self.format_name == "goal_macro_combat":
            patched_summary = "\n".join(
                f"- {item.title}: {item.value}"
                for item in patched_details
                if item.section == "Goal"
            )
        patched = StrategyDocument(
            summary=patched_summary,
            details=patched_details,
            format_name=self.format_name,
        )
        return patched.render(), changes


__all__ = [
    "StrategyDetail",
    "StrategyDocument",
    "paragraph_hash",
    "paragraph_id",
]
