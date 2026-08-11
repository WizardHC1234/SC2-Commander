"""Version-aware access to the SC2 2026-07-01 dataset and its evidence corpus."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT_DIR / "data_sc2_260701"
DEFAULT_DATABASE_PATH = DEFAULT_DATASET_DIR / "data_base_sc2_260701.json"
ENTITY_SECTIONS = ("Ability", "Unit", "Upgrade")
ALL_SECTIONS = (*ENTITY_SECTIONS, "SubOntology")
def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class DatasetStore:
    """Load one immutable dataset release and expose indexed entities and evidence."""

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        path = Path(database_path).expanduser().resolve()
        if path.is_dir():
            candidates = sorted(path.glob("data_base_sc2_*.json"))
            if not candidates:
                raise FileNotFoundError(f"No data_base_sc2_*.json found in {path}")
            path = candidates[-1]
        if not path.exists():
            raise FileNotFoundError(path)
        self.database_path = path
        self.dataset_dir = path.parent
        self.data: dict[str, list[dict[str, Any]]] = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = self.dataset_dir / "BUILD_MANIFEST.json"
        self.manifest: dict[str, Any] = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        self._entity_indexes = {
            section: {
                normalize_key(item.get("name")): item
                for item in self.data.get(section, [])
                if item.get("name")
            }
            for section in ALL_SECTIONS
        }
        self._abilities_by_id = {
            item.get("id"): item
            for item in self.data.get("Ability", [])
            if item.get("id") is not None
        }
        self._production_by_target = self._build_production_index()
        self._relations = self._collect_relations()
        self._relations_by_id = {
            relation["relation_id"]: relation
            for relation in self._relations
            if relation.get("relation_id")
        }
        self._relations_by_subject: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._relations_by_object: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for relation in self._relations:
            relation_name = str(relation.get("relation") or "")
            subject = normalize_key(relation.get("subject_name"))
            object_name = normalize_key(relation.get("object_name"))
            if relation_name and subject:
                self._relations_by_subject.setdefault(
                    (relation_name, subject), []
                ).append(relation)
            if relation_name and object_name:
                self._relations_by_object.setdefault(
                    (relation_name, object_name), []
                ).append(relation)
        self._facts_by_id: dict[str, dict[str, Any]] = {}
        for relation in self._relations:
            for fact in relation.get("fact") or []:
                if fact.get("fact_id"):
                    self._facts_by_id[fact["fact_id"]] = fact

    def _collect_relations(self) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for section in ALL_SECTIONS:
            for entity in self.data.get(section, []):
                for relation in entity.get("relations") or []:
                    relation = dict(relation)
                    relation.setdefault("subject_type", section)
                    relation.setdefault("object_type", "")
                    relation["description"] = list_value(relation.get("description"))
                    relation["source"] = list(relation.get("source") or [])
                    relation["fact"] = list(relation.get("fact") or [])
                    signature = relation.get("relation_id") or json.dumps(
                        [
                            relation.get("subject_type"),
                            relation.get("subject_name"),
                            relation.get("relation"),
                            relation.get("object_type"),
                            relation.get("object_name"),
                        ],
                        ensure_ascii=False,
                    )
                    if signature in seen:
                        continue
                    seen.add(signature)
                    collected.append(relation)
        return collected

    def _build_production_index(self) -> dict[tuple[str, str], list[dict[str, Any]]]:
        index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for producer in self.data.get("Unit", []):
            for ability_ref in producer.get("abilities") or []:
                ability = self.ability_for_ref(ability_ref)
                target = ability.get("target") if isinstance(ability, dict) else None
                if not isinstance(target, dict):
                    continue
                requirements = [
                    {
                        key: value
                        for key, value in requirement.items()
                        if key.endswith("_name")
                    }
                    for requirement in ability_ref.get("requirements") or []
                    if isinstance(requirement, dict)
                ]
                for target_type, payload in target.items():
                    if not isinstance(payload, dict):
                        continue
                    produced = (
                        ("Unit", payload.get("produces_name"))
                        if payload.get("produces_name")
                        else ("Upgrade", payload.get("upgrade_name"))
                    )
                    section, name = produced
                    if not name:
                        continue
                    index.setdefault((section, normalize_key(name)), []).append(
                        {
                            "producer": producer,
                            "ability": ability,
                            "target_type": target_type,
                            "requirements": requirements,
                        }
                    )
        return index

    def items(self, sections: Iterable[str] = ENTITY_SECTIONS) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for section in sections:
            for item in self.data.get(section, []):
                row = dict(item)
                row["_section"] = section
                rows.append(row)
        return rows

    def get_entity(self, section: str, name: str) -> dict[str, Any] | None:
        return self._entity_indexes.get(section, {}).get(normalize_key(name))

    def ability_for_ref(self, ability_ref: dict[str, Any]) -> dict[str, Any] | None:
        name = ability_ref.get("ability_name")
        if name:
            ability = self.get_entity("Ability", name)
            if ability:
                return ability
        return self._abilities_by_id.get(ability_ref.get("ability"))

    def production_sources(
        self,
        section: str,
        name: str,
        *,
        race: str = "",
    ) -> list[dict[str, Any]]:
        rows = self._production_by_target.get((section, normalize_key(name)), [])
        if not race:
            return rows
        wanted = normalize_key(race)
        return [
            row
            for row in rows
            if normalize_key((row.get("producer") or {}).get("race")) == wanted
        ]

    def relations(self) -> list[dict[str, Any]]:
        return self._relations

    def relations_for(
        self,
        entity_name: str,
        relation_names: Iterable[str],
        *,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        """Return endpoint-indexed relations without rescanning the dataset."""
        normalized = normalize_key(entity_name)
        rows: list[dict[str, Any]] = []
        for relation_name in relation_names:
            if direction in {"forward", "both"}:
                rows.extend(
                    self._relations_by_subject.get((relation_name, normalized), [])
                )
            if direction in {"reverse", "both"}:
                rows.extend(
                    self._relations_by_object.get((relation_name, normalized), [])
                )
        return rows

    def relation(self, relation_id: str) -> dict[str, Any] | None:
        return self._relations_by_id.get(relation_id)

    def fact(self, fact_id: str) -> dict[str, Any] | None:
        return self._facts_by_id.get(fact_id)

    def subontology(self, name: str) -> dict[str, Any] | None:
        return self.get_entity("SubOntology", name)

    def unit_classes(self, unit_name: str) -> list[str]:
        unit = self.get_entity("Unit", unit_name)
        if not unit:
            return []
        direct = list(unit.get("dimension_a_classes") or [])
        race = unit.get("race")
        result = direct + ([race] if race else [])
        result.extend(f"{race}_{value}" for value in direct if race and self.subontology(f"{race}_{value}"))
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "database_path": str(self.database_path),
            "schema_version": self.manifest.get("schema_version"),
            "generated_at": self.manifest.get("generated_at"),
            "counts": {section: len(self.data.get(section, [])) for section in ALL_SECTIONS},
            "relation_count": len(self._relations),
        }


def list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


@lru_cache(maxsize=8)
def get_dataset_store(database_path: str | Path = DEFAULT_DATABASE_PATH) -> DatasetStore:
    return DatasetStore(Path(database_path).resolve())


__all__ = [
    "ALL_SECTIONS",
    "DEFAULT_DATABASE_PATH",
    "DEFAULT_DATASET_DIR",
    "DatasetStore",
    "ENTITY_SECTIONS",
    "get_dataset_store",
    "normalize_key",
]
