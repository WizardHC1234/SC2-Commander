"""Deterministic, compact SC2 knowledge retrieval for strategy evolution."""

from __future__ import annotations

import importlib
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .sc2_data_store import (
    ALL_SECTIONS,
    DEFAULT_DATABASE_PATH,
    get_dataset_store,
    normalize_key,
)


DEFAULT_DATA_PATH = DEFAULT_DATABASE_PATH


KNOWLEDGE_PACKET_SCHEMA = "strategy_knowledge.v2"
MAX_ENTITIES = 6
MAX_RELATIONS = 8
MAX_DESCRIPTIONS_PER_ENTITY = 1
GENERIC_COMMAND_ABILITIES = {
    "attackattack",
    "holdpositionhold",
    "movemove",
    "patrolpatrol",
    "scanmove",
    "smart",
    "stopstop",
}

NEED_ALIASES = {
    "effect": "effects",
    "effects": "effects",
    "ability": "effects",
    "abilities": "effects",
    "capability": "effects",
    "synergy": "synergy",
    "support": "synergy",
    "counter": "counters",
    "counters": "counters",
    "matchup": "counters",
    "requirement": "requirements",
    "requirements": "requirements",
    "prerequisite": "requirements",
    "prerequisites": "requirements",
    "production": "requirements",
}

NEED_PATTERNS = {
    "effects": (
        "effect", "ability", "capability", "heal", "damage", "range",
        "作用", "能力", "治疗", "伤害", "射程", "效果",
    ),
    "synergy": (
        "synergy", "synerg", "support", "combine", "together",
        "协同", "配合", "搭配", "支援", "组合",
    ),
    "counters": (
        "counter", "against", "versus", " vs ", "克制", "应对", "对抗",
    ),
    "requirements": (
        "cost", "time", "build", "produce", "research", "prerequisite",
        "producer", "addon", "资源", "成本", "时间", "建造", "生产",
        "研究", "前置", "科技", "建筑",
    ),
}

RELATIONS_BY_NEED = {
    "effects": ("unlocks_unit_ability", "grants_stat_bonus", "enables_morph"),
    "synergy": ("synergizes_with", "garrisons_in"),
    "counters": ("counters",),
}

ENTITY_ALIASES = {
    "兵营": "Barracks",
    "重工厂": "Factory",
    "工厂": "Factory",
    "星港": "Starport",
    "机场": "Starport",
    "指挥中心": "CommandCenter",
    "科技实验室": "TechLab",
    "反应堆": "Reactor",
    "机枪兵": "Marine",
    "劫掠者": "Marauder",
    "坦克": "SiegeTank",
    "雷神": "Thor",
    "医疗运输机": "Medivac",
    "维京": "Viking",
    "女妖": "Banshee",
    "解放者": "Liberator",
    "战列巡航舰": "Battlecruiser",
    "大和战舰": "Battlecruiser",
    "兴奋剂": "Stimpack",
}


def _clean_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        value = [value] if value else []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def infer_knowledge_needs(question: str, explicit: Any = None) -> list[str]:
    """Use explicit needs when supplied; otherwise infer them from the question."""
    needs: list[str] = []
    for raw in _clean_strings(explicit):
        normalized = NEED_ALIASES.get(raw.casefold())
        if normalized and normalized not in needs:
            needs.append(normalized)
    if needs:
        return needs
    folded = str(question or "").casefold()
    for need, patterns in NEED_PATTERNS.items():
        if need not in needs and any(pattern in folded for pattern in patterns):
            needs.append(need)
    return needs or ["effects"]


def _exact_entity(name: str, data_path: str | Path) -> dict[str, Any] | None:
    store = get_dataset_store(data_path)
    candidates: list[dict[str, Any]] = []
    for section in ALL_SECTIONS:
        entity = store.get_entity(section, name)
        if entity:
            candidates.append(
                {"section": section, "name": entity.get("name"), "race": entity.get("race")}
            )
    return candidates[0] if candidates else None


def _fuzzy_entity(name: str, data_path: str | Path) -> dict[str, Any] | None:
    store = get_dataset_store(data_path)
    mention = ENTITY_ALIASES.get(name.strip(), name)
    wanted = normalize_key(mention)
    ranked: list[tuple[float, dict[str, Any]]] = []
    for section in ALL_SECTIONS:
        for entity in store.data.get(section, []):
            candidate = normalize_key(entity.get("name"))
            score = SequenceMatcher(None, wanted, candidate).ratio()
            if wanted and wanted in candidate:
                score += 0.25
            if score >= 0.45:
                ranked.append(
                    (
                        score,
                        {
                            "section": section,
                            "name": entity.get("name"),
                            "race": entity.get("race"),
                        },
                    )
                )
    return max(ranked, key=lambda pair: pair[0])[1] if ranked else None


def resolve_knowledge_entities(
    question: str,
    explicit: Any = None,
    *,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> list[dict[str, Any]]:
    """Resolve explicit names first; otherwise find canonical Unit/Upgrade names."""
    mentions = _clean_strings(explicit)
    store = get_dataset_store(data_path)
    if not mentions:
        normalized_question = normalize_key(question)
        matches: list[tuple[int, int, str]] = []
        for section in ("Unit", "Upgrade"):
            for entity in store.data.get(section, []):
                name = str(entity.get("name") or "")
                key = normalize_key(name)
                if len(key) >= 5 and key in normalized_question:
                    matches.append((normalized_question.index(key), -len(key), name))
        mentions = [name for _, _, name in sorted(matches)]

    resolved: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for mention in mentions:
        item = _exact_entity(mention, data_path)
        if item is None:
            item = _fuzzy_entity(mention, data_path)
        if not isinstance(item, dict) or not item.get("name") or not item.get("section"):
            continue
        key = (str(item["section"]), normalize_key(item["name"]))
        if key in seen:
            continue
        seen.add(key)
        resolved.append(
            {
                "section": str(item["section"]),
                "name": str(item["name"]),
                "race": item.get("race"),
            }
        )
        if len(resolved) >= MAX_ENTITIES:
            break
    return resolved


def _seconds(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / 22.4, 1)


def _relevant_descriptions(
    descriptions: Any,
    question: str,
    entity_names: Iterable[str],
) -> list[str]:
    rows = _clean_strings(descriptions)
    if not rows:
        return []
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", question)
        if token.casefold() not in {"using", "bundled", "dataset", "tools", "source", "truth"}
    }
    tokens.update(normalize_key(name) for name in entity_names)
    ranked = []
    for index, row in enumerate(rows):
        folded = row.casefold()
        normalized = normalize_key(row)
        score = sum(1 for token in tokens if token and (token in folded or token in normalized))
        ranked.append((-score, index, row[:450]))
    return [row for _, _, row in sorted(ranked)[:MAX_DESCRIPTIONS_PER_ENTITY]]


def _entity_fact(
    entity_ref: dict[str, Any],
    question: str,
    entity_names: list[str],
    needs: list[str],
    data_path: str | Path,
) -> dict[str, Any]:
    store = get_dataset_store(data_path)
    section = entity_ref["section"]
    entity = store.get_entity(section, entity_ref["name"]) or {}
    fact: dict[str, Any] = {
        "section": section,
        "name": entity.get("name") or entity_ref["name"],
    }
    if entity.get("race"):
        fact["race"] = entity.get("race")
    if section == "Unit" and "requirements" in needs:
        fact["cost"] = {
            "minerals": entity.get("minerals", 0),
            "gas": entity.get("gas", 0),
            "supply": entity.get("supply", 0),
            "time_seconds": _seconds(entity.get("time")),
        }
    if section == "Unit" and "effects" in needs:
        fact["stats"] = {
            key: entity.get(key)
            for key in ("max_health", "armor", "speed", "attributes", "attack_type")
            if entity.get(key) is not None
        }
        weapons = []
        for weapon in (entity.get("weapons") or [])[:2]:
            weapons.append(
                {
                    key: weapon.get(key)
                    for key in ("target_type", "damage_per_hit", "attacks", "range", "cooldown", "bonuses")
                    if weapon.get(key) not in (None, [], {})
                }
            )
        if weapons:
            fact["weapons"] = weapons
    elif section == "Upgrade" and "requirements" in needs:
        cost = entity.get("cost") if isinstance(entity.get("cost"), dict) else {}
        fact["cost"] = {
            "minerals": cost.get("minerals", 0),
            "gas": cost.get("gas", 0),
            "time_seconds": _seconds(cost.get("time")),
        }
    elif section == "Ability" and "effects" in needs:
        fact["ability"] = {
            key: entity.get(key)
            for key in ("energy_cost", "cast_range", "cooldown", "target")
            if entity.get(key) is not None
        }
    descriptions = _relevant_descriptions(
        entity.get("description"), question, entity_names
    )
    if descriptions:
        fact["descriptions"] = descriptions
    chains = _clean_strings(entity.get("tech_chain"))
    if chains and "requirements" in needs:
        fact["tech_chain"] = chains[:2]
    return fact


def _load_action_specs(race: str) -> dict[str, Any]:
    normalized = str(race or "").strip().casefold()
    if normalized not in {"terran", "protoss", "zerg"}:
        return {}
    try:
        module = importlib.import_module(f"commander.races.{normalized}.actions")
    except Exception:
        return {}
    specs = getattr(module, "ACTION_SPECS", {})
    return specs if isinstance(specs, dict) else {}


def _action_facts(entity_names: list[str], race: str) -> list[dict[str, Any]]:
    wanted = {normalize_key(name) for name in entity_names}
    rows: list[dict[str, Any]] = []
    for action_name, spec in _load_action_specs(race).items():
        stem = re.sub(r"^(train|build|research|morph)_?", "", action_name)
        if normalize_key(stem) not in wanted:
            continue
        rows.append(
            {
                "action": action_name,
                "description": spec.description,
                "cost_kind": spec.cost_kind,
                "minerals": spec.minerals,
                "gas": spec.vespene,
                "supply": spec.supply,
                "base_time_seconds": spec.base_time_seconds,
                "production_location": spec.production_location,
                "prerequisites": list(spec.prerequisites),
                "dependencies": list(spec.dependencies),
            }
        )
    return rows


def _control_effect_facts(entity_names: list[str], race: str) -> list[dict[str, Any]]:
    """Return structured effects encoded in authoritative control-tool metadata."""
    wanted = {normalize_key(name) for name in entity_names}
    rows: list[dict[str, Any]] = []
    specs = _load_action_specs(race)
    if any(name.startswith("scannersweep") for name in wanted) and "scanner_sweep" in specs:
        rows.append(
            {
                "entity": "Scanner Sweep",
                "energy_cost": 50,
                "cooldown": None,
                "limit": "energy_limited",
                "source": "commander_action_metadata",
            }
        )
    return rows


def _production_facts(
    entities: list[dict[str, Any]],
    action_entity_names: set[str],
    race: str,
    data_path: str | Path,
) -> list[dict[str, Any]]:
    store = get_dataset_store(data_path)
    rows: list[dict[str, Any]] = []
    for entity in entities:
        if entity["section"] not in {"Unit", "Upgrade"}:
            continue
        if normalize_key(entity["name"]) in action_entity_names:
            continue
        for item in store.production_sources(
            entity["section"], entity["name"], race=race
        )[:3]:
            producer = item.get("producer") or {}
            requirements = item.get("requirements") or []
            required_addon = next(
                (
                    requirement.get("addon_name") or requirement.get("addon_to_name")
                    for requirement in requirements
                    if requirement.get("addon_name") or requirement.get("addon_to_name")
                ),
                None,
            )
            rows.append(
                {
                    "target": entity["name"],
                    "producer": producer.get("name"),
                    "required_addon": required_addon,
                    "requirements": requirements,
                }
            )
    return rows


def _ability_facts(
    entities: list[dict[str, Any]],
    data_path: str | Path,
) -> list[dict[str, Any]]:
    store = get_dataset_store(data_path)
    rows: list[dict[str, Any]] = []
    for entity in entities:
        if entity["section"] != "Unit":
            continue
        unit = store.get_entity("Unit", entity["name"]) or {}
        for ability_ref in unit.get("abilities") or []:
            ability = store.ability_for_ref(ability_ref) or {}
            if normalize_key(ability.get("name")) in GENERIC_COMMAND_ABILITIES:
                continue
            descriptions = _clean_strings(ability.get("description"))
            structured = {
                "energy_cost": ability.get("energy_cost"),
                "cast_range": ability.get("cast_range"),
                "cooldown": ability.get("cooldown"),
            }
            if not descriptions and all(value is None for value in structured.values()):
                continue
            rows.append(
                {
                    "unit": entity["name"],
                    "ability": ability.get("name"),
                    **structured,
                    "description": [descriptions[0][:350]] if descriptions else [],
                }
            )
    return rows[:12]


def _relation_facts(
    entities: list[dict[str, Any]],
    needs: list[str],
    data_path: str | Path,
) -> list[dict[str, Any]]:
    store = get_dataset_store(data_path)
    names = {normalize_key(entity["name"]) for entity in entities}
    relation_names = list(
        dict.fromkeys(
            relation
            for need in needs
            for relation in RELATIONS_BY_NEED.get(need, ())
        )
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entity in entities:
        for relation in store.relations_for(entity["name"], relation_names):
            relation_name = str(relation.get("relation") or "")
            other = (
                relation.get("object_name")
                if normalize_key(relation.get("subject_name")) == normalize_key(entity["name"])
                else relation.get("subject_name")
            )
            if len(names) > 1 and normalize_key(other) not in names:
                continue
            relation_id = str(relation.get("relation_id") or "")
            signature = relation_id or "|".join(
                str(relation.get(key) or "")
                for key in ("subject_name", "relation", "object_name")
            )
            if signature in seen:
                continue
            seen.add(signature)
            rows.append(
                {
                    "subject": relation.get("subject_name"),
                    "relation": relation_name,
                    "object": relation.get("object_name"),
                    "description": [
                        description[:450]
                        for description in _clean_strings(relation.get("description"))[:1]
                    ],
                    "relation_id": relation_id,
                }
            )
            if len(rows) >= MAX_RELATIONS:
                return rows
    return rows


def build_strategy_knowledge(
    item: dict[str, Any],
    *,
    race: str = "",
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> dict[str, Any]:
    """Build one bounded knowledge packet without an LLM planning loop."""
    question = str(item.get("question") or "").strip()
    needs = infer_knowledge_needs(question, item.get("needs"))
    entities = resolve_knowledge_entities(
        question,
        item.get("entities"),
        data_path=data_path,
    )
    entity_names = [entity["name"] for entity in entities]
    action_facts = (
        _action_facts(entity_names, race) if "requirements" in needs else []
    )
    action_entity_names = {
        normalize_key(re.sub(r"^(train|build|research|morph)_?", "", row["action"]))
        for row in action_facts
    }
    entity_facts = [
        _entity_fact(entity, question, entity_names, needs, data_path)
        for entity in entities
    ]
    production = (
        _production_facts(entities, action_entity_names, race, data_path)
        if "requirements" in needs
        else []
    )
    # A directly resolved Ability already carries its own structured fields.
    # In that case, do not also dump every command exposed by accompanying units.
    abilities = (
        _ability_facts(entities, data_path)
        if "effects" in needs
        and not any(entity["section"] == "Ability" for entity in entities)
        else []
    )
    control_effects = (
        _control_effect_facts(entity_names, race) if "effects" in needs else []
    )
    if control_effects:
        for fact in entity_facts:
            if normalize_key(fact.get("name")).startswith("scannersweep"):
                fact["ability"] = {
                    "energy_cost": 50,
                    "cooldown": None,
                    "target": (fact.get("ability") or {}).get("target"),
                    "source": "commander_action_metadata",
                }
    relations = _relation_facts(entities, needs, data_path)

    missing: list[str] = []
    if not entities:
        missing.append("No canonical Unit or Upgrade entity could be resolved.")
    for need in needs:
        if need in {"synergy", "counters"} and not any(
            row.get("relation") in RELATIONS_BY_NEED[need] for row in relations
        ):
            missing.append(f"No structured {need} relation was found for the resolved entities.")
        if need == "effects":
            has_effects = bool(abilities or control_effects or relations) or any(
                row.get("stats") or row.get("weapons") or row.get("ability")
                for row in entity_facts
            )
            if not has_effects:
                missing.append("No structured effects facts were found for the resolved entities.")

    return {
        "schema": KNOWLEDGE_PACKET_SCHEMA,
        "question": question,
        "entities": entities,
        "needs": needs,
        "action_facts": action_facts,
        "entity_facts": entity_facts,
        "production": production,
        "abilities": abilities,
        "control_effects": control_effects,
        "relations": relations,
        "missing": missing,
    }


def render_strategy_knowledge(packet: dict[str, Any]) -> str:
    """Render the packet once so downstream prompts do not duplicate raw evidence."""
    lines = [f"Question: {packet.get('question') or '(none)'}"]
    entities = packet.get("entities") or []
    lines.append(
        "Resolved entities: "
        + (", ".join(f"{row['name']} ({row['section']})" for row in entities) or "none")
    )
    lines.append("Verified facts:")
    for row in packet.get("action_facts") or []:
        cost = f"{row['minerals']}M/{row['gas']}G"
        if row.get("supply"):
            cost += f"/{row['supply']}S"
        requirements = list(row.get("prerequisites") or []) + list(row.get("dependencies") or [])
        lines.append(
            f"- {row['action']}: {cost}; {row['base_time_seconds']}s; "
            f"at {row['production_location']}; requires "
            f"{', '.join(requirements) if requirements else 'none listed'}"
        )
    action_names = {
        normalize_key(re.sub(r"^(train|build|research|morph)_?", "", row["action"]))
        for row in packet.get("action_facts") or []
    }
    for row in packet.get("entity_facts") or []:
        if normalize_key(row.get("name")) not in action_names and row.get("cost"):
            lines.append(f"- {row['name']} database card: {row['cost']}")
        if row.get("stats"):
            lines.append(f"- {row['name']} stats: {row['stats']}")
        for weapon in row.get("weapons") or []:
            lines.append(f"- {row['name']} weapon: {weapon}")
        if row.get("ability"):
            lines.append(f"- {row['name']} ability: {row['ability']}")
        for description in row.get("descriptions") or []:
            lines.append(f"- {row['name']}: {description}")
    for row in packet.get("production") or []:
        lines.append(
            f"- {row['target']} is produced/researched at {row.get('producer') or 'unknown'}"
            + (f" with {row['required_addon']}" if row.get("required_addon") else "")
        )
    for row in packet.get("abilities") or []:
        details = ", ".join(
            f"{key}={row[key]}"
            for key in ("energy_cost", "cast_range", "cooldown")
            if row.get(key) is not None
        )
        prefix = f"- {row['unit']} / {row['ability']}"
        if details:
            lines.append(f"{prefix}: {details}")
        for description in row.get("description") or []:
            lines.append(f"{prefix}: {description}")
    for row in packet.get("control_effects") or []:
        details = [f"energy_cost={row['energy_cost']}"]
        details.append(
            f"cooldown={row['cooldown']}" if row.get("cooldown") is not None
            else "cooldown=none listed"
        )
        details.append(f"limit={row['limit']}")
        lines.append(f"- {row['entity']}: {', '.join(details)}")
    for row in packet.get("relations") or []:
        description = "; ".join(row.get("description") or [])
        lines.append(
            f"- {row['subject']} {row['relation']} {row['object']}"
            + (f": {description}" if description else "")
        )
    for missing in packet.get("missing") or []:
        lines.append(f"- Evidence limit: {missing}")
    return "\n".join(lines)


__all__ = [
    "KNOWLEDGE_PACKET_SCHEMA",
    "build_strategy_knowledge",
    "infer_knowledge_needs",
    "render_strategy_knowledge",
    "resolve_knowledge_entities",
]
