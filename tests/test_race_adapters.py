import importlib

from sharpy.plans import BuildOrder


def test_terran_runtime_adapters_match_commander_loader_contract():
    actions = importlib.import_module("commander.races.terran.actions")
    tactics = importlib.import_module("commander.races.terran.tactics")

    assert callable(actions.get_action)
    assert callable(actions.get_action_space)
    assert callable(actions.expand_action_dependencies)
    assert len(actions.get_action_space()) == 74
    assert isinstance(tactics.create_tactics(), BuildOrder)


def test_every_terran_macro_action_factory_can_be_instantiated():
    actions = importlib.import_module("commander.races.terran.actions")
    for action_name, spec in actions.ACTION_SPECS.items():
        if spec.is_macro:
            assert actions.get_action(action_name, 1) is not None, action_name
