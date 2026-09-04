from evol_agent.core import config
from llm.caller import _apply_reasoning_disable, _apply_reasoning_enable


def test_reasoning_is_disabled_for_all_evol_agent_stages() -> None:
    assert config.MATCH_SUBAGENT_ENABLE_REASONING is False
    assert config.CROSS_MATCH_DISCOVERY_ENABLE_REASONING is False
    assert config.OPTIMIZATION_PACKAGE_GENERATION_ENABLE_REASONING is False
    assert config.OPTIMIZATION_PACKAGE_SELECTION_ENABLE_REASONING is False
    assert config.CANDIDATE_GENERATION_ENABLE_REASONING is False
    assert config.PARENT_TIMING_PACKAGE_EXTRACTION_ENABLE_REASONING is False
    assert config.CONTACT_TIMING_EXTRACTION_ENABLE_REASONING is False
    assert config.STRATEGY_SEMANTIC_VALIDATION_ENABLE_REASONING is False
    assert config.MECHANISM_HISTORY_ENABLE_REASONING is False
    assert config.EXPERIMENT_AUDIT_ENABLE_REASONING is False


def test_deepseek_reasoning_uses_chat_template_kwargs() -> None:
    enabled: dict = {}
    _apply_reasoning_enable("deepseek-v4-flash", enabled)
    assert enabled["extra_body"]["chat_template_kwargs"] == {
        "thinking": True,
        "enable_thinking": True,
    }
    assert "thinking" not in enabled["extra_body"]

    disabled: dict = {}
    _apply_reasoning_disable("deepseek-v4-flash", disabled)
    assert disabled["extra_body"]["chat_template_kwargs"] == {
        "thinking": False,
        "enable_thinking": False,
    }
