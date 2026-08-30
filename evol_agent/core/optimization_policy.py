from __future__ import annotations


OPTIMIZATION_POLICY = """
Use these rules as strategic guidance, not as a form-filling checklist:

1. Identify the Champion's combat style, win mechanism, useful timing window, and repeated winning behavior before proposing any change. Preserve that identity and those realized strengths unless direct evidence at the current difficulty contradicts them.
2. Compare wins and losses to find the earliest strategy-controlled cause of the outcome. Distinguish a missing or harmful strategy instruction from a runtime failure to execute an instruction that was already sufficient.
3. Optimize for winning the decisive engagement and then the match. Evaluate attack timing together with both armies at actual contact and the ability to reinforce or continue pressure; no unit, upgrade, producer, scouting step, or threshold is beneficial by itself.
4. Treat earliest_feasible_time as a production lower bound. Use observed commitment and contact timing to judge whether a proposed change still reaches a useful opponent window.
5. Preserve the Champion's core production unless match evidence supports changing it. Add support, technology, economy, or extra production only when it addresses the diagnosed loss and its timing and resource cost do not erase the winning sequence.
6. Change one causal direction per candidate, with only the dependencies needed to realize it. Do not redesign unrelated sections or turn strategy prose into a detailed state machine.
7. Use experiment history semantically. Preserve implemented score-improving changes, do not replay a failed direction under new wording, and allow at most one concrete repair of a failed direction before switching to a different causal lever.
8. Write concise, observable, map-agnostic rules within the runtime boundary. Do not require hidden state, exact timestamps or zone ids, unit-level micro, fixed custom detachments, or scripted group splitting and merging.
""".strip()


HARD_VALIDATION_POLICY = """
Reject only a hard strategy error: an unsupported runtime action, a missing mandatory
technology or production dependency, a direct internal contradiction that makes the
plan non-executable, or an explicit final composition above 200 supply. Missing audit
fields, concise explanations, uncertain strategic quality, or similarity to a prior
experiment are not hard errors and must be resolved by match evaluation.
""".strip()


__all__ = ["OPTIMIZATION_POLICY", "HARD_VALIDATION_POLICY"]
