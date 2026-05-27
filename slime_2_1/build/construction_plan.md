# C-001: Construction Sequence

Source of truth: `docs/blueprint_2_1.md` (Sheet C-001).

Phases listed in the order the blueprint specifies. Each phase has a clear
exit condition; do not start the next phase until the previous one passes its
acceptance check.

| Phase | Months | Scope                                                              | Acceptance |
| :---- | :----- | :----------------------------------------------------------------- | :--------- |
| 1     | 1–2    | Foundation (as 2.0, no changes)                                    | 2.0 acceptance carries forward |
| 2     | 3      | AD and optimizer (as 2.0, SOT library + identity-loss branch)      | CAME path passes 2.0 acceptance with SOT library wired in |
| 3     | 4      | QD core: final-bmap descriptor archive, RFF KDE, sparsity bonus, gate fitness, validate single-role (classifier-only); confirm SOT corrigibility gate | Q-001 Class C (corrigibility) baseline passes; 100-gen classifier-only run reproduces 2.0 metrics |
| 4     | 5      | Captured-graph execution (as 2.0)                                  | Behavioral equivalence between captured and sequential modes on 1k-gen run |
| 5     | 6      | Placeholder regressor + probe set: regressor as sole surprise source, probe set generation + signing + host evaluation, CUSUM on placeholder surprise | Placeholder surprise tracks fitness drift on synthetic injections |
| 6     | 7      | BTRAJ capture + role machinery: sample bmap at intermediate CA steps; store BTRAJ; implement role tag in genome and role-switched input pathway; verify role mutations occur at the intended low rate; implement bootstrap-trigger condition and predictor-founder spawning | Unit tests in Q-001 (BTRAJ correctness, role-switched input) pass; role mutation rate measured at 1e-4 |
| 7     | 8      | Predictor role activation: run through bootstrap trigger in a controlled experiment; verify founders reproduce; prediction loss decreases over generations; ensemble surprise correlates with placeholder; implement hybrid blending; validate role-balance fitness scaling | Bootstrap-trigger integration test passes; r climbs above 0.5 within 1000 post-bootstrap generations |
| 8     | 9      | Structural pressures + PT ladders: role-aware audit + probe panel (incl. L_role); sentinel training on role-blind data; per-role lineage tracking; mutation-rate ladder with Metropolis swaps + adaptive beta; SOT-density stress ladder; Q-001 red-team exercises | All Q-001 attack classes A-F have detection within their T_detect_max windows |
| 9     | 10     | Full integration + long-run hardening: 10^5-generation stability runs with all subsystems active; optional MPK backend behavioral-equivalence validation | 10^4-generation post-bootstrap run sustains predictor sub-population, r > 0.5, no spontaneous Class A-F conditions |

Notes:

- Phases 1-4 align with the 2.0 construction sequence; reuse 2.0 acceptance
  artifacts wherever possible.
- Phase 6 is the smallest substantive change introducing the role concept;
  hold off on predictor *activation* until phase 7 so the bootstrap trigger can
  be exercised in isolation.
- Hybrid blending (phase 7) and PT ladders (phase 8) are independent in the
  source tree; they can be developed in parallel if staffing allows, but the
  acceptance runs in phase 9 should happen against the merged build.
