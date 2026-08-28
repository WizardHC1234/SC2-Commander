from __future__ import annotations


OPTIMIZATION_POLICY = """
Use these rules as strategic guidance, not as a form-filling checklist:

1. Identify the current strategy's combat style, intended power window, core army,
   and win mechanism before proposing changes. Preserve that identity unless
   repeated match evidence shows that the identity itself is failing.
2. Learn from both outcomes. Preserve behavior repeatedly associated with wins and
   repair the earliest strategy-fixable shortfall repeatedly associated with losses.
3. Separate strategy defects from execution defects. If the strategy already states
   an executable rule but the runtime fails to apply it, report the execution issue;
   do not keep changing thresholds or adding duplicate prose to compensate.
4. Optimize for winning the decisive army engagement and then the match. Economy,
   production, upgrades, scouting, attack gates, and retreat are supporting means.
5. Compare contact timing together with both armies' composition and growth. An
   earlier attack may exploit a weaker enemy; a later attack is justified only when
   the added combat value outweighs the opponent's growth and the survival risk.
   Judge retreat and continuation from the effective fighting cluster, nearby reinforcements, losses before or after the retreat trigger, and whether the enemy force is collapsing. Do not infer a strategy defect from global inventory or one isolated reinforcement group's automatic retreat.
6. Keep the production plan coherent across opening, first commitment, continued
   reinforcement, and late-game completion. Account for workers, bases, gas,
   production capacity, shared queues, upgrades, supply, and resource banking.
7. Make every change needed by one coherent intervention, but do not rewrite
   unrelated behavior. A complete strategy may coordinate several related sections.
8. History is evidence, not a ban list. A previously weak direction may be repaired
   when the new candidate explains the missing dependency or implementation change.
9. Write concise, observable, reusable rules. Do not fit exact match timestamps,
   opponent builds, map zone ids, hidden state, or unit-level micro.
""".strip()


HARD_VALIDATION_POLICY = """
Reject only a hard strategy error: an unsupported runtime action, a missing mandatory
technology or production dependency, a direct internal contradiction that makes the
plan non-executable, or an explicit final composition above 200 supply. Missing audit
fields, concise explanations, uncertain strategic quality, or similarity to a prior
experiment are not hard errors and must be resolved by match evaluation.
""".strip()


__all__ = ["OPTIMIZATION_POLICY", "HARD_VALIDATION_POLICY"]
