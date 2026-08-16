"""Terran Commander action catalog and Sharpy execution factories.

Every tool is defined once in ``ACTION_SPECS``. A macro specification contains
its model-facing mechanics and executable factory together; army/meta tools have
no factory and are handled directly by Commander.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Set, Tuple

from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sharpy.plans.acts import (
    ActUnit,
    BuildGas,
    DefensePosition,
    DefensiveBuilding,
    Expand,
    GridBuilding,
    Tech,
)
from sharpy.plans.acts.terran import BuildAddon, MorphOrbitals, MorphPlanetary

from commander.races.action_spec import (
    ActionSpec,
    control_action,
    expand_action_dependencies as _expand_action_dependencies,
    macro_action,
    render_action_space,
    research_action,
    validate_action_specs,
)


def _train(
    description: str,
    unit_type,
    producer_type,
    location: str,
    minerals: int,
    vespene: int,
    seconds: float,
    *,
    supply: float = 0,
    prerequisites: Tuple[str, ...] = (),
    dependencies: Tuple[str, ...] = (),
) -> ActionSpec:
    def action_func(*args, **kwargs):
        return ActUnit(unit_type, producer_type, *args, **kwargs)

    return macro_action(
        description,
        "unit",
        action_func,
        minerals,
        vespene,
        seconds,
        supply=supply,
        production_role="producer",
        production_location=location,
        prerequisites=prerequisites,
        dependencies=dependencies,
    )


def _grid_build(
    description: str,
    unit_type,
    minerals: int,
    vespene: int,
    seconds: float,
    *,
    prerequisites: Tuple[str, ...] = (),
    dependencies: Tuple[str, ...] = (),
) -> ActionSpec:
    def action_func(*args, **kwargs):
        return GridBuilding(unit_type, *args, **kwargs)

    return macro_action(
        description,
        "building",
        action_func,
        minerals,
        vespene,
        seconds,
        production_role="builder",
        production_location="SCV",
        prerequisites=prerequisites,
        dependencies=dependencies,
    )


def _addon(
    description: str,
    addon_type,
    parent_type,
    parent_name: str,
    minerals: int,
    vespene: int,
    seconds: float,
    *,
    prerequisites: Tuple[str, ...],
    dependencies: Tuple[str, ...],
) -> ActionSpec:
    def action_func(*args, **kwargs):
        return BuildAddon(addon_type, parent_type, *args, **kwargs)

    return macro_action(
        description,
        "building",
        action_func,
        minerals,
        vespene,
        seconds,
        production_role="producer",
        production_location=parent_name,
        prerequisites=prerequisites,
        dependencies=dependencies,
    )


def _research(
    description: str,
    upgrade_id,
    location: str,
    minerals: int,
    vespene: int,
    seconds: float,
    *,
    from_building=None,
    prerequisites: Tuple[str, ...] = (),
    dependencies: Tuple[str, ...] = (),
) -> ActionSpec:
    def action_func(*args, **kwargs):
        if from_building is None:
            return Tech(upgrade_id)
        return Tech(upgrade_id, from_building)

    return research_action(
        description,
        action_func,
        minerals,
        vespene,
        seconds,
        location,
        prerequisites=prerequisites,
        dependencies=dependencies,
    )


ACTION_SPECS: Dict[str, ActionSpec] = {
    # Units and payloads
    "train_scv": _train(
        "Train SCV from Command Center",
        UnitTypeId.SCV,
        UnitTypeId.COMMANDCENTER,
        "Command Center",
        50,
        0,
        12.1,
        supply=1,
        dependencies=("expand",),
    ),
    "train_marine": _train(
        "Train Marine from Barracks",
        UnitTypeId.MARINE,
        UnitTypeId.BARRACKS,
        "Barracks",
        50,
        0,
        17.9,
        supply=1,
        dependencies=("build_barracks",),
    ),
    "train_marauder": _train(
        "Absolute Marauder count from Barracks with Tech Lab "
        "(emit build_barracks_techlab; Reactor Barracks cannot train Marauders)",
        UnitTypeId.MARAUDER,
        UnitTypeId.BARRACKS,
        "Barracks",
        100,
        25,
        21.4,
        supply=2,
        prerequisites=("Barracks Tech Lab",),
        dependencies=("build_barracks", "build_barracks_techlab"),
    ),
    "train_reaper": _train(
        "Train Reaper from Barracks",
        UnitTypeId.REAPER,
        UnitTypeId.BARRACKS,
        "Barracks",
        50,
        50,
        34.0,
        supply=1,
        dependencies=("build_barracks",),
    ),
    "train_ghost": _train(
        "Train Ghost from Barracks (requires Tech Lab and Ghost Academy)",
        UnitTypeId.GHOST,
        UnitTypeId.BARRACKS,
        "Barracks",
        150,
        125,
        28.6,
        supply=2,
        prerequisites=("Barracks Tech Lab", "Ghost Academy"),
        dependencies=(
            "build_barracks",
            "build_barracks_techlab",
            "build_ghost_academy",
        ),
    ),
    "build_nuke": _train(
        "Build Nuke at Ghost Academy",
        UnitTypeId.NUKE,
        UnitTypeId.GHOSTACADEMY,
        "Ghost Academy",
        100,
        100,
        43.0,
        dependencies=("build_ghost_academy",),
    ),
    "train_hellion": _train(
        "Train Hellion from Factory",
        UnitTypeId.HELLION,
        UnitTypeId.FACTORY,
        "Factory",
        100,
        0,
        21.4,
        supply=2,
        dependencies=("build_factory",),
    ),
    "train_hellbat": _train(
        "Train Hellbat from Factory (requires Armory)",
        UnitTypeId.HELLIONTANK,
        UnitTypeId.FACTORY,
        "Factory",
        100,
        0,
        21.4,
        supply=2,
        prerequisites=("Armory",),
        dependencies=("build_factory", "build_armory"),
    ),
    "train_widow_mine": _train(
        "Train Widow Mine from Factory",
        UnitTypeId.WIDOWMINE,
        UnitTypeId.FACTORY,
        "Factory",
        75,
        25,
        21.4,
        supply=2,
        dependencies=("build_factory",),
    ),
    "train_cyclone": _train(
        "Train Cyclone from Factory",
        UnitTypeId.CYCLONE,
        UnitTypeId.FACTORY,
        "Factory",
        150,
        100,
        32.1,
        supply=3,
        prerequisites=("Factory Tech Lab",),
        dependencies=("build_factory", "build_factory_techlab"),
    ),
    "train_siege_tank": _train(
        "Absolute Siege Tank count from Factory with Tech Lab "
        "(emit build_factory_techlab; Reactor Factory cannot train Tanks)",
        UnitTypeId.SIEGETANK,
        UnitTypeId.FACTORY,
        "Factory",
        150,
        125,
        32.1,
        supply=3,
        prerequisites=("Factory Tech Lab",),
        dependencies=("build_factory", "build_factory_techlab"),
    ),
    "train_thor": _train(
        "Train Thor from Factory (requires Tech Lab and Armory)",
        UnitTypeId.THOR,
        UnitTypeId.FACTORY,
        "Factory",
        300,
        200,
        42.9,
        supply=6,
        prerequisites=("Factory Tech Lab", "Armory"),
        dependencies=("build_factory", "build_factory_techlab", "build_armory"),
    ),
    "train_viking": _train(
        "Train Viking from Starport",
        UnitTypeId.VIKINGFIGHTER,
        UnitTypeId.STARPORT,
        "Starport",
        125,
        75,
        30.0,
        supply=2,
        dependencies=("build_starport",),
    ),
    "train_medivac": _train(
        "Train Medivac from Starport",
        UnitTypeId.MEDIVAC,
        UnitTypeId.STARPORT,
        "Starport",
        100,
        100,
        30.0,
        supply=2,
        dependencies=("build_starport",),
    ),
    "train_liberator": _train(
        "Train Liberator from Starport",
        UnitTypeId.LIBERATOR,
        UnitTypeId.STARPORT,
        "Starport",
        150,
        125,
        42.9,
        supply=3,
        dependencies=("build_starport",),
    ),
    "train_raven": _train(
        "Train Raven from Starport (requires Tech Lab)",
        UnitTypeId.RAVEN,
        UnitTypeId.STARPORT,
        "Starport",
        100,
        150,
        34.3,
        supply=2,
        prerequisites=("Starport Tech Lab",),
        dependencies=("build_starport", "build_starport_techlab"),
    ),
    "train_banshee": _train(
        "Train Banshee from Starport (requires Tech Lab)",
        UnitTypeId.BANSHEE,
        UnitTypeId.STARPORT,
        "Starport",
        150,
        100,
        42.9,
        supply=3,
        prerequisites=("Starport Tech Lab",),
        dependencies=("build_starport", "build_starport_techlab"),
    ),
    "train_battlecruiser": _train(
        "Absolute Battlecruiser count from Starport with Tech Lab and Fusion Core "
        "(emit build_starport_techlab and build_fusion_core)",
        UnitTypeId.BATTLECRUISER,
        UnitTypeId.STARPORT,
        "Starport",
        400,
        300,
        64.3,
        supply=6,
        prerequisites=("Starport Tech Lab", "Fusion Core"),
        dependencies=(
            "build_starport",
            "build_starport_techlab",
            "build_fusion_core",
        ),
    ),

    # Structures
    "build_supply_depot": _grid_build(
        "Build Supply Depot",
        UnitTypeId.SUPPLYDEPOT,
        100,
        0,
        21.4,
        dependencies=("train_scv",),
    ),
    "build_barracks": _grid_build(
        "Absolute Barracks count. Hosts at most one addon each; raise this when "
        "Tech Lab + Reactor targets need more free Barracks",
        UnitTypeId.BARRACKS,
        150,
        0,
        46.4,
        prerequisites=("Supply Depot",),
        dependencies=("train_scv", "build_supply_depot"),
    ),
    "build_factory": _grid_build(
        "Absolute Factory count. Hosts at most one addon each; raise this when "
        "Factory Tech Lab / Reactor targets need more free Factories",
        UnitTypeId.FACTORY,
        150,
        100,
        42.9,
        prerequisites=("Barracks",),
        dependencies=("train_scv", "build_barracks"),
    ),
    "build_starport": _grid_build(
        "Absolute Starport count. Hosts at most one addon each; raise this when "
        "Starport Tech Lab / Reactor targets need more free Starports",
        UnitTypeId.STARPORT,
        150,
        100,
        35.7,
        prerequisites=("Factory",),
        dependencies=("train_scv", "build_factory"),
    ),
    "build_gas": macro_action(
        "Build Refinery for gas",
        "building",
        lambda *args, **kwargs: BuildGas(*args, **kwargs),
        75,
        0,
        21.4,
        production_role="builder",
        production_location="SCV",
        prerequisites=("free Vespene geyser",),
        dependencies=("train_scv",),
    ),
    "expand": macro_action(
        "Expand to new base (Command Center)",
        "building",
        lambda *args, **kwargs: Expand(*args, **kwargs),
        300,
        0,
        71.4,
        production_role="builder",
        production_location="SCV",
        prerequisites=("available expansion site",),
        dependencies=("train_scv",),
    ),
    "build_engineering_bay": _grid_build(
        "Build Engineering Bay for infantry upgrades and turrets",
        UnitTypeId.ENGINEERINGBAY,
        125,
        0,
        25.0,
        prerequisites=("Command Center",),
        dependencies=("train_scv", "expand"),
    ),
    "build_armory": _grid_build(
        "Build Armory for vehicle/ship upgrades and Thors",
        UnitTypeId.ARMORY,
        150,
        50,
        46.4,
        prerequisites=("Factory",),
        dependencies=("train_scv", "build_factory"),
    ),
    "build_ghost_academy": _grid_build(
        "Build Ghost Academy for Ghosts and nukes",
        UnitTypeId.GHOSTACADEMY,
        150,
        50,
        28.6,
        prerequisites=("Barracks",),
        dependencies=("train_scv", "build_barracks"),
    ),
    "build_fusion_core": _grid_build(
        "Build Fusion Core for Battlecruisers and advanced upgrades",
        UnitTypeId.FUSIONCORE,
        150,
        150,
        46.4,
        prerequisites=("Starport",),
        dependencies=("train_scv", "build_starport"),
    ),
    "build_bunker": macro_action(
        "Build Defensive Bunker",
        "building",
        lambda *args, **kwargs: DefensiveBuilding(
            UnitTypeId.BUNKER,
            DefensePosition.Entrance,
            None,
            args[0] if args else 1,
        ),
        100,
        0,
        28.6,
        production_role="builder",
        production_location="SCV",
        prerequisites=("Barracks",),
        dependencies=("train_scv", "build_barracks"),
    ),
    "build_missile_turret": macro_action(
        "Build Missile Turret at mineral lines for anti-air and detection",
        "building",
        lambda *args, **kwargs: DefensiveBuilding(
            UnitTypeId.MISSILETURRET,
            DefensePosition.CenterMineralLine,
            None,
            args[0] if args else 1,
        ),
        100,
        0,
        17.9,
        production_role="builder",
        production_location="SCV",
        prerequisites=("Engineering Bay",),
        dependencies=("train_scv", "build_engineering_bay"),
    ),
    "build_entrance_missile_turret": macro_action(
        "Build Missile Turret near base entrances for anti-air and detection",
        "building",
        lambda *args, **kwargs: DefensiveBuilding(
            UnitTypeId.MISSILETURRET,
            DefensePosition.Entrance,
            None,
            args[0] if args else 1,
        ),
        100,
        0,
        17.9,
        production_role="builder",
        production_location="SCV",
        prerequisites=("Engineering Bay",),
        dependencies=("train_scv", "build_engineering_bay"),
    ),
    "build_sensor_tower": _grid_build(
        "Build Sensor Tower",
        UnitTypeId.SENSORTOWER,
        100,
        50,
        17.9,
        prerequisites=("Engineering Bay",),
        dependencies=("train_scv", "build_engineering_bay"),
    ),

    # Add-ons
    "build_barracks_techlab": _addon(
        "Absolute Barracks Tech Lab count. Needs one completed Barracks with no "
        "addon per Tech Lab; does not build Barracks—raise build_barracks if no free slot",
        UnitTypeId.BARRACKSTECHLAB,
        UnitTypeId.BARRACKS,
        "Barracks",
        50,
        25,
        17.9,
        prerequisites=("completed Barracks with free addon slot",),
        dependencies=("build_barracks",),
    ),
    "build_barracks_reactor": _addon(
        "Absolute Barracks Reactor count. Needs one completed Barracks with no addon "
        "per Reactor; Tech Lab and Reactor cannot share a Barracks—raise build_barracks "
        "when slots are full (e.g. 1 Tech Lab + 2 Reactors needs 3 Barracks)",
        UnitTypeId.BARRACKSREACTOR,
        UnitTypeId.BARRACKS,
        "Barracks",
        50,
        50,
        35.7,
        prerequisites=("completed Barracks with free addon slot",),
        dependencies=("build_barracks",),
    ),
    "build_factory_techlab": _addon(
        "Absolute Factory Tech Lab count. Needs one completed Factory with no addon "
        "per Tech Lab; does not build Factories—raise build_factory if no free slot",
        UnitTypeId.FACTORYTECHLAB,
        UnitTypeId.FACTORY,
        "Factory",
        50,
        25,
        17.9,
        prerequisites=("completed Factory with free addon slot",),
        dependencies=("build_factory",),
    ),
    "build_factory_reactor": _addon(
        "Absolute Factory Reactor count. Needs one completed Factory with no addon "
        "per Reactor; Tech Lab and Reactor cannot share a Factory—raise build_factory "
        "when slots are full",
        UnitTypeId.FACTORYREACTOR,
        UnitTypeId.FACTORY,
        "Factory",
        50,
        50,
        35.7,
        prerequisites=("completed Factory with free addon slot",),
        dependencies=("build_factory",),
    ),
    "build_starport_techlab": _addon(
        "Absolute Starport Tech Lab count. Needs one completed Starport with no addon "
        "per Tech Lab; does not build Starports—raise build_starport if no free slot",
        UnitTypeId.STARPORTTECHLAB,
        UnitTypeId.STARPORT,
        "Starport",
        50,
        25,
        17.9,
        prerequisites=("completed Starport with free addon slot",),
        dependencies=("build_starport",),
    ),
    "build_starport_reactor": _addon(
        "Absolute Starport Reactor count. Needs one completed Starport with no addon "
        "per Reactor; Tech Lab and Reactor cannot share a Starport—raise build_starport "
        "when slots are full",
        UnitTypeId.STARPORTREACTOR,
        UnitTypeId.STARPORT,
        "Starport",
        50,
        50,
        35.7,
        prerequisites=("completed Starport with free addon slot",),
        dependencies=("build_starport",),
    ),

    # Research
    "research_shieldwall": _research(
        "Research Combat Shield (Shield Wall) for Marines",
        UpgradeId.SHIELDWALL,
        "Barracks Tech Lab",
        100,
        100,
        78.6,
        dependencies=("build_barracks_techlab",),
    ),
    "research_stimpack": _research(
        "Research Stimpack for Marines and Marauders",
        UpgradeId.STIMPACK,
        "Barracks Tech Lab",
        100,
        100,
        100.0,
        dependencies=("build_barracks_techlab",),
    ),
    "research_concussive_shells": _research(
        "Research Concussive Shells for Marauders",
        UpgradeId.PUNISHERGRENADES,
        "Barracks Tech Lab",
        50,
        50,
        42.9,
        dependencies=("build_barracks_techlab",),
    ),
    "research_personal_cloaking": _research(
        "Research Personal Cloaking for Ghosts",
        UpgradeId.PERSONALCLOAKING,
        "Ghost Academy",
        150,
        150,
        85.7,
        dependencies=("build_ghost_academy",),
    ),
    "research_infernal_preigniter": _research(
        "Research Infernal Pre-igniter for Hellions/Hellbats",
        UpgradeId.HIGHCAPACITYBARRELS,
        "Factory Tech Lab",
        100,
        100,
        78.6,
        dependencies=("build_factory_techlab",),
    ),
    "research_drilling_claws": _research(
        "Research Drilling Claws for Widow Mines",
        UpgradeId.DRILLCLAWS,
        "Factory Tech Lab",
        75,
        75,
        78.6,
        prerequisites=("Armory",),
        dependencies=("build_factory_techlab", "build_armory"),
    ),
    "research_magfield_accelerator": _research(
        "Research Mag-Field Accelerator for Cyclones",
        UpgradeId.MAGFIELDLAUNCHERS,
        "Factory Tech Lab",
        100,
        100,
        100.0,
        from_building=UnitTypeId.FACTORYTECHLAB,
        dependencies=("build_factory_techlab",),
    ),
    "research_smart_servos": _research(
        "Research Smart Servos for transforming units (Thor, Hellbat, Viking)",
        UpgradeId.SMARTSERVOS,
        "Factory Tech Lab",
        100,
        100,
        78.6,
        prerequisites=("Armory",),
        dependencies=("build_factory_techlab", "build_armory"),
    ),
    "research_banshee_cloak": _research(
        "Research Cloaking Field for Banshees",
        UpgradeId.BANSHEECLOAK,
        "Starport Tech Lab",
        100,
        100,
        78.6,
        dependencies=("build_starport_techlab",),
    ),
    "research_banshee_speed": _research(
        "Research Hyperflight Rotors for Banshees",
        UpgradeId.BANSHEESPEED,
        "Starport Tech Lab",
        125,
        125,
        79.0,
        dependencies=("build_starport_techlab",),
    ),
    "research_raven_corvid_reactor": _research(
        "Research Corvid Reactor for Ravens",
        UpgradeId.RAVENCORVIDREACTOR,
        "Starport Tech Lab",
        150,
        150,
        78.6,
        dependencies=("build_starport_techlab",),
    ),
    "research_liberator_range": _research(
        "Research Advanced Ballistics for Liberators",
        UpgradeId.LIBERATORAGRANGEUPGRADE,
        "Fusion Core",
        150,
        150,
        78.6,
        dependencies=("build_fusion_core",),
    ),
    "research_yamato_cannon": _research(
        "Research Yamato Cannon for Battlecruisers",
        UpgradeId.YAMATOCANNON,
        "Fusion Core",
        150,
        150,
        100.0,
        from_building=UnitTypeId.FUSIONCORE,
        dependencies=("build_fusion_core",),
    ),
    "research_hisec_auto_tracking": _research(
        "Research Hi-Sec Auto Tracking (Turret Range)",
        UpgradeId.HISECAUTOTRACKING,
        "Engineering Bay",
        100,
        100,
        57.1,
        dependencies=("build_engineering_bay",),
    ),
    "research_neosteel_armor": _research(
        "Research Neosteel Armor (Building Armor)",
        UpgradeId.TERRANBUILDINGARMOR,
        "Engineering Bay",
        150,
        150,
        100.0,
        dependencies=("build_engineering_bay",),
    ),
    "research_infantry_weapons_1": _research(
        "Upgrade Infantry Weapons Level 1",
        UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
        "Engineering Bay",
        100,
        100,
        114.3,
        dependencies=("build_engineering_bay",),
    ),
    "research_infantry_weapons_2": _research(
        "Upgrade Infantry Weapons Level 2",
        UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
        "Engineering Bay",
        150,
        150,
        135.7,
        prerequisites=("Armory", "Terran Infantry Weapons Level 1"),
        dependencies=(
            "build_engineering_bay",
            "build_armory",
            "research_infantry_weapons_1",
        ),
    ),
    "research_infantry_weapons_3": _research(
        "Upgrade Infantry Weapons Level 3",
        UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
        "Engineering Bay",
        200,
        200,
        157.1,
        prerequisites=("Armory", "Terran Infantry Weapons Level 2"),
        dependencies=(
            "build_engineering_bay",
            "build_armory",
            "research_infantry_weapons_2",
        ),
    ),
    "research_infantry_armor_1": _research(
        "Upgrade Infantry Armor Level 1",
        UpgradeId.TERRANINFANTRYARMORSLEVEL1,
        "Engineering Bay",
        100,
        100,
        114.3,
        dependencies=("build_engineering_bay",),
    ),
    "research_infantry_armor_2": _research(
        "Upgrade Infantry Armor Level 2",
        UpgradeId.TERRANINFANTRYARMORSLEVEL2,
        "Engineering Bay",
        150,
        150,
        135.7,
        prerequisites=("Armory", "Terran Infantry Armors Level 1"),
        dependencies=(
            "build_engineering_bay",
            "build_armory",
            "research_infantry_armor_1",
        ),
    ),
    "research_infantry_armor_3": _research(
        "Upgrade Infantry Armor Level 3",
        UpgradeId.TERRANINFANTRYARMORSLEVEL3,
        "Engineering Bay",
        200,
        200,
        157.1,
        prerequisites=("Armory", "Terran Infantry Armors Level 2"),
        dependencies=(
            "build_engineering_bay",
            "build_armory",
            "research_infantry_armor_2",
        ),
    ),
    "research_vehicle_weapons_1": _research(
        "Upgrade Vehicle Weapons Level 1",
        UpgradeId.TERRANVEHICLEWEAPONSLEVEL1,
        "Armory",
        100,
        100,
        114.3,
        dependencies=("build_armory",),
    ),
    "research_vehicle_weapons_2": _research(
        "Upgrade Vehicle Weapons Level 2",
        UpgradeId.TERRANVEHICLEWEAPONSLEVEL2,
        "Armory",
        175,
        175,
        135.7,
        prerequisites=("Terran Vehicle Weapons Level 1",),
        dependencies=("build_armory", "research_vehicle_weapons_1"),
    ),
    "research_vehicle_weapons_3": _research(
        "Upgrade Vehicle Weapons Level 3",
        UpgradeId.TERRANVEHICLEWEAPONSLEVEL3,
        "Armory",
        250,
        250,
        157.1,
        prerequisites=("Terran Vehicle Weapons Level 2",),
        dependencies=("build_armory", "research_vehicle_weapons_2"),
    ),
    "research_ship_weapons_1": _research(
        "Upgrade Ship Weapons Level 1",
        UpgradeId.TERRANSHIPWEAPONSLEVEL1,
        "Armory",
        100,
        100,
        114.3,
        dependencies=("build_armory",),
    ),
    "research_ship_weapons_2": _research(
        "Upgrade Ship Weapons Level 2",
        UpgradeId.TERRANSHIPWEAPONSLEVEL2,
        "Armory",
        175,
        175,
        135.7,
        prerequisites=("Terran Ship Weapons Level 1",),
        dependencies=("build_armory", "research_ship_weapons_1"),
    ),
    "research_ship_weapons_3": _research(
        "Upgrade Ship Weapons Level 3",
        UpgradeId.TERRANSHIPWEAPONSLEVEL3,
        "Armory",
        250,
        250,
        157.1,
        prerequisites=("Terran Ship Weapons Level 2",),
        dependencies=("build_armory", "research_ship_weapons_2"),
    ),
    "research_vehicle_and_ship_armor_1": _research(
        "Upgrade Vehicle and Ship Armor Level 1",
        UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1,
        "Armory",
        100,
        100,
        114.3,
        dependencies=("build_armory",),
    ),
    "research_vehicle_and_ship_armor_2": _research(
        "Upgrade Vehicle and Ship Armor Level 2",
        UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL2,
        "Armory",
        175,
        175,
        135.7,
        prerequisites=("Terran Vehicle And Ship Armors Level 1",),
        dependencies=(
            "build_armory",
            "research_vehicle_and_ship_armor_1",
        ),
    ),
    "research_vehicle_and_ship_armor_3": _research(
        "Upgrade Vehicle and Ship Armor Level 3",
        UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL3,
        "Armory",
        250,
        250,
        157.1,
        prerequisites=("Terran Vehicle And Ship Armors Level 2",),
        dependencies=(
            "build_armory",
            "research_vehicle_and_ship_armor_2",
        ),
    ),

    # Morphs
    "morph_orbital_command": macro_action(
        "Morph Command Center into Orbital Command",
        "building",
        lambda *args, **kwargs: MorphOrbitals(*args),
        150,
        0,
        25.0,
        target_semantics="absolute_count",
        cost_kind="incremental_cost",
        production_role="producer",
        production_location="Command Center",
        prerequisites=("Barracks",),
        dependencies=("expand", "build_barracks"),
    ),
    "morph_planetary_fortress": macro_action(
        "Morph Command Center into Planetary Fortress (requires Engineering Bay)",
        "building",
        lambda *args, **kwargs: MorphPlanetary(*args),
        250,
        150,
        35.7,
        target_semantics="absolute_count",
        cost_kind="incremental_cost",
        production_role="producer",
        production_location="Command Center",
        prerequisites=("Engineering Bay",),
        dependencies=("expand", "build_engineering_bay"),
    ),

    # Army control
    "army_intent": control_action(
        "Required every decision cycle, including before combat units exist: set the "
        "persistent stance for the whole army. Use exactly one of "
        "mode=hold|attack|regroup|cleanup and copy zone_id from the observation. "
        "Use cleanup only when the Runtime Cleanup Hint appears. The runtime "
        "moves the main force, sends reinforcements to the live main-force position, "
        "and handles local defense, retreat, and re-gathering.",
        "army",
    ),
    "scanner_sweep": control_action(
        "Request one Scanner Sweep on a zone (costs 50 Orbital energy). zone_id must "
        "exist in the observation. Omit this tool to request no scan this cycle.",
        "army",
    ),
    "scout": control_action(
        "Send or keep one SCV zone scout. Temporarily reserves one existing SCV; no "
        "mineral or gas cost. zone_id must exist in the observation. While an SCV scout "
        "is already active, repeat the same zone_id to preserve it. Omit this tool to "
        "cancel any active scout.",
        "army",
    ),

    # Decision scheduling
    "set_wake_event": control_action(
        "Required each decision cycle: declare exactly one composite wake condition for "
        "the next Commander decision. Use logic all|any over whitelist predicates "
        "(unit_count_at_least / unit_count_less_than, objective_status_became, "
        "destination_reached, scan_ready, cleanup_hint_present, game_time_at_least, "
        "supply_left_at_most). Do not use scout_result_is, scout_just_finished, "
        "movement_mode_in, movement_mode_not_in, army_group_count_at_least, "
        "army_group_count_less_than, or objective_status_is. unit_count wakes require "
        "matching train tools in the same cycle. Align the condition with the next "
        "strategy reassessment; omitting this tool triggers a weak "
        "game_time_at_least=now+60 fallback.",
        "meta",
    ),
}

validate_action_specs(ACTION_SPECS)


def get_action_space() -> Dict[str, str]:
    """Return the complete model-facing Terran tool catalog."""
    return render_action_space(ACTION_SPECS)


def expand_action_dependencies(
    action_names: Iterable[str],
    *,
    known_action_names: Optional[Iterable[str]] = None,
) -> Set[str]:
    """Expand Terran structural dependencies and positive-gas requirements."""
    return _expand_action_dependencies(
        ACTION_SPECS,
        action_names,
        known_action_names=known_action_names,
        gas_action_name="build_gas",
    )


def get_action(action_key: str, *args, **kwargs):
    """Instantiate the Sharpy Act for one macro action key."""
    spec = ACTION_SPECS.get(action_key)
    if spec is None:
        raise ValueError(f"Action key '{action_key}' not found in action space.")
    if not spec.is_macro or spec.action_func is None:
        raise ValueError(
            f"Action key '{action_key}' is a {spec.action_type} tool; "
            "it has no Sharpy Act."
        )
    return spec.action_func(*args, **kwargs)


__all__ = [
    "ACTION_SPECS",
    "expand_action_dependencies",
    "get_action",
    "get_action_space",
]
