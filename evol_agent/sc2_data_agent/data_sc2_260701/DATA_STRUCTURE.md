# StarCraft II Structured Data Graph — 2026-07-01 Data Structure

This document describes `data_base_sc2_260701.json`, a self-contained StarCraft II: Legacy of the Void entity graph with concrete semantic relations, SubOntology definitions, and file-addressable Markdown evidence.

## 1. Release Layout

```text
data_sc2_260701/
├── data_base_sc2_260701.json
├── DATA_STRUCTURE.md
├── SUB_ONTOLOGY.md
├── BUILD_MANIFEST.json
├── markdown/{Terran,Protoss,Zerg}/*.md
└── relations/
    ├── entity_expanded_relations.json
    └── README.md
```

## 2. Top-Level Collections

| Collection | Count | Purpose |
|---|---:|---|
| `Ability` | 683 | Commands, actions, casts, production, research, and morph abilities. |
| `Unit` | 204 | Units, structures, transformed forms, temporary units, and special forms. |
| `Upgrade` | 124 | Researched technologies and stat improvements. |
| `SubOntology` | 109 | Hierarchical Unit classes and their canonical membership lists. |

## 3. Unified Relation Object

Every relation uses the same schema. `description`, `source`, and `fact` are always lists.

```json
{
  "relation_id": "stable SHA-256",
  "subject_name": "Marine",
  "subject_type": "Unit",
  "relation": "counters",
  "object_name": "Zergling",
  "object_type": "Unit",
  "description": ["One or more retained descriptions."],
  "source": [{"kind": "markdown_semantic_extraction", "fact_ids": ["..."]}],
  "fact": [{"document": "markdown/Terran/marine.md", "line_start": 91, "line_end": 91, "evidence_text": "..."}]
}
```

### 3.1 `description`

A deduplicated list of every description retained for the triple. Multiple documents, direct extraction, SubOntology expansion, and deterministic post-processing may all contribute descriptions.

### 3.2 `source`

A list describing how the relation was created. Supported source kinds are:

- `structured_data_direct`
- `structured_data_inference`
- `markdown_semantic_extraction`
- `subontology_expansion`
- `pending_postprocess`

A source may include an original relation ID, SubOntology names, member bindings, split or upgrade-expansion group IDs, qualifiers, and references to facts through `fact_ids`.

### 3.3 `fact`

Semantic relations contain one or more Markdown facts. Each fact records the exact release-relative document path, document entity, race, heading path, starting and ending source lines, block ID, and exact evidence text. Structured relations use an empty fact list because their derivation is recorded in `source` rather than in Markdown.

## 4. Relation Inventory

The release stores **6503** relation objects: **2034** structured relations, **4463** concrete entity-level semantic relations, and **6** ontology-scope subject relations.

| Relation | Count |
|---|---:|
| `ability_requires_unit` | 126 |
| `ability_requires_upgrade` | 60 |
| `action_result` | 326 |
| `counters` | 2221 |
| `enables_morph` | 20 |
| `garrisons_in` | 344 |
| `grants_stat_bonus` | 1353 |
| `has_ability` | 1078 |
| `morphs_into` | 123 |
| `produces` | 108 |
| `researches` | 105 |
| `spawns` | 47 |
| `synergizes_with` | 531 |
| `unlocks_unit_ability` | 61 |

## 5. Semantic Relations

The concrete semantic graph contains five relation types:

| Relation | Concrete count | Direction |
|---|---:|---|
| `counters` | 2217 | Unit → Unit |
| `synergizes_with` | 531 | Unit → Unit |
| `garrisons_in` | 342 | Unit → Unit |
| `grants_stat_bonus` | 1353 | Upgrade → Unit |
| `enables_morph` | 20 | Upgrade or structure Unit → Unit |

`hard_counters` and `soft_counters` are deprecated and absent. They are replaced by `counters`. Generic-group rules and new composite classes proposed in the deferred Section 8 were not applied.

## 6. Structured Relations

Structured relations are preserved from the original graph and migrated without changing their triples. Direct relations use field-level sources; inferred relations identify their deterministic rule. Their fact lists are empty.

## 7. SubOntology Expansion

The concrete semantic relation file expands SubOntology endpoints to member Units. A triple may therefore have direct Markdown sources, one or more SubOntology expansion sources, and deterministic pending-normalization sources simultaneously. Duplicate triples are stored once and retain all descriptions, sources, and facts.

## 8. Evidence Addressing

All Markdown facts use paths under `markdown/`. Line numbers are one-based and refer to byte-identical copies of the extraction corpus. `line_start`, `line_end`, `block_id`, and `evidence_text` together provide exact verification of the fact.

## 9. Migration Summary

- Removed legacy semantic relations: **989**
- Preserved and migrated structured relations: **2034**
- Added concrete semantic relations: **4463**
- Added SubOntology records: **109**

## 10. Data Invariants

1. Every relation has list-valued `description`, `source`, and `fact` fields.
2. Every concrete semantic relation has at least one Markdown fact.
3. Every structured relation has an empty fact list.
4. Every fact document resolves inside this release and its evidence text matches the stored source line range.
5. Concrete semantic triples are unique.
6. All 109 SubOntology names and all members are canonical.
7. `GroundUnits` and `AirUnits` form a disjoint, complete partition of the 204 Units.
8. The 116 Markdown files are checksum-identical to the source corpus.
