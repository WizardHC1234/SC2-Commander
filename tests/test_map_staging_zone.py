from types import SimpleNamespace

from commander.combat_state import _zone_topology
from commander.observation import format_map_topology
from commander.prompts import _ARMY_ZONES


def _zone(*, own=False, enemy=False, island=False):
    return SimpleNamespace(
        is_ours=own,
        is_enemys=enemy,
        is_island=island,
        ramp=None,
        paths={},
        center_location=object(),
    )


def test_zone_topology_uses_own_natural_as_default_staging_zone():
    topology = _zone_topology(
        [
            _zone(own=True),
            _zone(own=True),
            _zone(),
            _zone(enemy=True),
        ],
        set(),
    )

    assert topology["default_pre_attack_staging_zone_id"] == "zone_1"


def test_zone_topology_falls_back_to_own_main_for_island_natural():
    topology = _zone_topology(
        [
            _zone(own=True),
            _zone(own=True, island=True),
            _zone(),
            _zone(enemy=True),
        ],
        set(),
    )

    assert topology["default_pre_attack_staging_zone_id"] == "zone_0"


def test_map_topology_exposes_default_staging_zone():
    text = format_map_topology(
        {
            "default_pre_attack_staging_zone_id": "zone_1",
            "zones": [],
        }
    )

    assert "default_pre_attack_staging_zone_id=zone_1" in text


def test_army_prompt_uses_staging_zone_with_safe_fallbacks():
    assert "default_pre_attack_staging_zone_id" in _ARMY_ZONES
    assert "current threat" in _ARMY_ZONES
    assert "unsafe or unreachable" in _ARMY_ZONES
