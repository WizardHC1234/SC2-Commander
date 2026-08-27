"""Compact replay-grounded demonstrations shared by EvolAgent prompts.

The examples intentionally teach causal comparisons rather than build orders.
Historical replay timings are provenance, not current-patch strategy targets.

Public sources checked 2026-08-27:
- https://lotv.spawningtool.com/build/142471/
- https://lotv.spawningtool.com/build/139133/
- https://lotv.spawningtool.com/build/166290/
- https://lotv.spawningtool.com/build/41705/
- https://doi.org/10.1038/s41597-023-02510-7
"""


REPLAY_GROUNDED_REASONING_EXAMPLES = """## Replay-grounded reasoning demonstrations

These historical public replay/build-order observations demonstrate how to reason;
they are not balance facts, target counts, or timing rules for the current match.
Never copy their numbers into a candidate unless current records and verified game
data independently support them.

1. Coordinated timing package, not an isolated unit count. A replay-backed TvT
   one-base push published on Spawning Tool (build 142471) describes contact around
   5:15 with Marines and Siege Tanks plus Viking/Liberator air control. Its build
   order coordinates the first Tank, continued Tank production, air production,
   and the move-out window. Reasoning lesson: if a candidate adds economy, an
   upgrade, or support as a hard gate, compare the resulting contact time and the
   opponent's growth. Do not infer that "more units later" is automatically better,
   and do not optimize one unit count while starving another part of the package.

2. Different styles justify different windows. The replay-backed one-base example
   above aims for a smaller early package, while the public TvT Special three-Raven
   mech timing (Spawning Tool build 139133) waits for roughly three Ravens and three
   Tanks before its stated move-out. Reasoning lesson: earlier and later are both
   viable only relative to the strategy's core mechanism. Preserve an early
   pressure window when it creates the advantage; accept a later window only when
   the added package improves matchup-adjusted power enough to offset enemy growth.

3. Scaling requires a bridge and a post-spike transition. A public TvT
   Battlecruiser build (Spawning Tool build 166290) sequences early Marines and
   Cyclones while reaching Battlecruisers, then adds Tanks and more Factories after
   the air-tech stage. A separate replay-backed direct Battlecruiser rush (build
   41705) explicitly reports vulnerability to early aggression. Reasoning lesson:
   when losses occur before the intended power spike, repair the minimum mobile
   survival/production dependency instead of increasing the final high-tech target.
   When the spike is reached but pressure stalls, analyze support, reinforcement,
   and transition after contact instead of redesigning the opening without evidence.

4. Engagement and retreat are temporal evidence. Public SC2 esports replay data
   such as SC2EGSet records ordered PlayerStats, UnitBorn, UnitDied, UnitPositions,
   and UpgradeComplete events. Apply the same temporal logic to Commander records:
   compare the main force immediately before contact, at losses or auto-retreat,
   after withdrawal, and at any regroup/re-engagement. Repeated losses before the
   retreat trigger support an earlier-retreat hypothesis; retreat before material
   losses followed by unused retained power supports a later-retreat or faster
   re-engagement hypothesis. Either conclusion still requires repeated own-match
   evidence and must not be inferred from the configured ratio alone.
"""


__all__ = ["REPLAY_GROUNDED_REASONING_EXAMPLES"]
