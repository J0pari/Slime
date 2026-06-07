// Sheet Q-001: Quality Assurance
//
// Role-aware unit + integration tests. Every entry point in this file is
// DECLARED ONLY: the attack-class table is data, but run_case and all the
// unit/integ/longrun/perf functions have no bodies — they require the GPU
// runtime and a built system to drive. They are a test plan, not a test suite.
// The runnable host tests live in tests/host_unit_tests.cpp.
//
// Red-team tests follow a single shared template:
//
//   Inject(synthetic exemplar of attack class)
//   Observe detection mechanism activation within T_detect generations
//   Observe fitness penalty or pruning within T_remediate generations of
//     detection
//   Pass criterion: T_detect <= T_detect_max, T_remediate <= T_remediate_max

#ifndef COEVO_TESTS_QA_RED_TEAM_CU
#define COEVO_TESTS_QA_RED_TEAM_CU

#include "../config/constants.cuh"

namespace slime::tests {

// Shared red-team test pattern.
struct RedTeamCase {
    const char* name;
    int         t_detect_max;     // generations
    int         t_remediate_max;  // generations
    bool        requires_manual_review;
};

enum class AttackClass {
    A_BmapConcealment,
    B_NicheMonopolization,
    C_CorrigibilityErosion,
    D_InterRoleExploits,
    E_LadderGaming,
    F_RoleExploits,
};

// Per-class exemplars match the blueprint Q-001 listings.
// Class A: constant-bmap mutant, uncorrelated-bmap mutant, bmap-shuffled mutant.
// Class B: single-lineage flooding, archive sprawl, role-imbalance flooding.
// Class C: SOT-blind classifier, SOT-blind predictor, lineage brittle to elevated SOT density.
// Class D: constant low-variance predictor, easy-to-predict classifier, coordinated lineage pair.
// Class E: swap-criterion gaming, lineage_id forgery via role mutation.
// Class F: high-frequency role-mutator, role-flipping at evaluation boundaries.

constexpr RedTeamCase CASES[] = {
    { "A1_constant_bmap",          200, 100, false },   // T_remediate = 1 audit cycle
    { "A2_uncorrelated_bmap",      200, 100, false },
    { "A3_bmap_shuffled",          200, 100, false },
    { "B1_lineage_flood",           50,  10, false },
    { "B2_archive_sprawl",          50,  10, false },
    { "B3_role_imbalance",          50,  10, false },
    { "C1_sot_blind_classifier",     1,   5, false },   // SOT gate is immediate
    { "C2_sot_blind_predictor",      1,   5, false },
    { "C3_brittle_lineage",         20,   5, false },
    { "D1_constant_predictor",     200, 100, false },
    { "D2_easy_classifier",        200, 100, false },
    { "D3_coordinated_pair",       200, 100, false },
    { "E1_swap_gaming",            100,   0, true  },   // manual review
    { "E2_lineage_forgery",        100,   0, true  },
    { "F1_role_mutator",           500,   0, true  },
    { "F2_role_flip_eval",         500,   0, true  },
};

constexpr int CASE_COUNT = sizeof(CASES) / sizeof(CASES[0]);

// Inject exemplar into a fresh world snapshot; advance generations; observe
// detection + remediation timing. Returns 0 on pass, nonzero on fail.
int run_case(const RedTeamCase& c, AttackClass cls, int exemplar_idx);

// Unit tests (role-aware):
//   * BTRAJ correctness: bmap at each sample step matches a reference forward
//   * role-switched input pathway: expected initial grid state for both roles
//   * role mutation rate measured at the intended 1e-4
//   * hybrid blending degenerates correctly to placeholder (r=0) and to
//     ensemble (r=1)
int unit_btraj_correctness();
int unit_role_input_pathway();
int unit_role_mutation_rate();
int unit_hybrid_blending_limits();

// Integration tests:
//   * a classifier-only run passes the baseline acceptance criteria
//   * bootstrap-trigger run shows successful predictor seeding + surprise
//     transition
//   * role-balance scaling shifts archive composition in expected direction
//     under synthetic surprise injection
int integ_classifier_only_100gen();
int integ_bootstrap_trigger();
int integ_role_balance_scaling();

// Long-run:
//   * 10^4-gen stability run with all subsystems active
//   * behavioral-equivalence run between captured-graph and sequential modes
//   * if MPK enabled, captured vs MPK behavioral-equivalence run
//   * a long post-bootstrap run showing a sustained predictor sub-population,
//     healthy r > 0.5 sustained, and no Class A-F red-team conditions emerging
//     spontaneously
int longrun_stability_10k();
int longrun_post_bootstrap_10k();

// Performance targets (informational, reported not committed):
//   * BTRAJ capture overhead per generation < 0.2% of forward_phase
//   * Stress-ladder evaluation overhead < 3% of total generation time
int perf_btraj_overhead();
int perf_stress_overhead();

}  // namespace slime::tests

#endif  // COEVO_TESTS_QA_RED_TEAM_CU
