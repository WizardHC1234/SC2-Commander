from evol_agent.core import config


def test_reasoning_is_enabled_only_for_strategic_judgment_stages() -> None:
    assert config.MATCH_SUBAGENT_ENABLE_REASONING is False
    assert config.CROSS_MATCH_DISCOVERY_ENABLE_REASONING is True
    assert config.CROSS_MATCH_DECISION_ENABLE_REASONING is True
    assert config.CANDIDATE_GENERATION_ENABLE_REASONING is True
    assert config.PARENT_TIMING_PACKAGE_EXTRACTION_ENABLE_REASONING is True
    assert config.CONTACT_TIMING_EXTRACTION_ENABLE_REASONING is False
    assert config.STRATEGY_SEMANTIC_VALIDATION_ENABLE_REASONING is False
    assert config.EXPERIMENT_AUDIT_ENABLE_REASONING is True
