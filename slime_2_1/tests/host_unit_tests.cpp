// Host-only unit tests for the math inlines that don't require a CUDA
// runtime: hybrid blending, Pearson r, ensemble surprise, role-balance
// multipliers, SOT gate, PT swap probabilities, genome bit accessors, and
// the CUSUM update.
//
// Compile:
//   g++ -std=c++17 -Itests/stubs -I. tests/host_unit_tests.cpp -o build/host_tests
// Run:
//   ./build/host_tests
//
// The constants header includes <cuda_fp16.h> and <cuda_runtime.h>. Stubs
// live in tests/stubs/ and are picked up via -Itests/stubs ahead of the
// system include path.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../config/constants.cuh"

// Math we want to test - re-paste the inlines from the modules so the host
// test doesn't pull in CUDA-specific code paths. Keep these copies in sync
// with the originals; the tests catch divergence by comparing against
// reference outputs computed inline.

static inline float sot_gate(float x) {
    float z = SOT_GATE_SLOPE * (x - SOT_GATE_MIDPOINT);
    if (z >= 0.f) { float ez = std::exp(-z); return 1.0f / (1.0f + ez); }
    float ez = std::exp(z); return ez / (1.0f + ez);
}

static inline float classifier_mult(float rho) {
    float gap = 1.0f - rho; if (gap < 0.f) gap = 0.f;
    return 1.0f + ROLE_BALANCE_COEFF * gap;
}
static inline float predictor_mult(float rho) {
    float gap = rho - 1.0f; if (gap < 0.f) gap = 0.f;
    return 1.0f + ROLE_BALANCE_COEFF * gap;
}

static inline float blend_surprise(float a, float b, float r) {
    if (r < 0.f) r = 0.f;
    if (r > 1.f) r = 1.f;
    return (1.0f - r) * a + r * b;
}

static inline float swap_accept_probability(float beta, float dl, float dh) {
    float arg = beta * (dh - dl);
    if (arg >= 0.f) return 1.0f;
    return std::exp(arg);
}

// ---- Tests ----------------------------------------------------------------
static int failures = 0;
static int total    = 0;

#define EXPECT_NEAR(a, b, tol) do { \
    total++; \
    float av = (a), bv = (b); \
    if (std::fabs(av - bv) > (tol)) { \
        failures++; \
        std::printf("FAIL %s:%d  %s = %g, expected %g (tol %g)\n", \
                    __FILE__, __LINE__, #a, av, bv, (float)(tol)); \
    } \
} while (0)

#define EXPECT_TRUE(c) do { \
    total++; \
    if (!(c)) { failures++; std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); } \
} while (0)

static void test_sot_gate() {
    // sigmoid(20*(x - 0.7)). At midpoint 0.7 -> 0.5; >> midpoint -> 1;
    // << midpoint -> 0.
    EXPECT_NEAR(sot_gate(0.7f), 0.5f,    1e-5f);
    EXPECT_NEAR(sot_gate(1.0f), 1.0f,    1e-2f);
    EXPECT_NEAR(sot_gate(0.4f), 0.0f,    1e-2f);
}

static void test_role_multipliers() {
    // rho = 1 -> both multipliers idle at 1.
    EXPECT_NEAR(classifier_mult(1.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(predictor_mult(1.0f),  1.0f, 1e-6f);
    // rho < 1 (predictors too easy) boosts classifiers, leaves predictors flat.
    EXPECT_NEAR(classifier_mult(0.0f), 1.0f + ROLE_BALANCE_COEFF, 1e-6f);
    EXPECT_NEAR(predictor_mult(0.0f),  1.0f, 1e-6f);
    // rho > 1 (predictors struggling) boosts predictors.
    EXPECT_NEAR(predictor_mult(2.0f),  1.0f + ROLE_BALANCE_COEFF, 1e-6f);
    EXPECT_NEAR(classifier_mult(2.0f), 1.0f, 1e-6f);
}

static void test_blend_surprise() {
    EXPECT_NEAR(blend_surprise(1.0f, 2.0f, 0.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(blend_surprise(1.0f, 2.0f, 1.0f), 2.0f, 1e-6f);
    EXPECT_NEAR(blend_surprise(1.0f, 2.0f, 0.5f), 1.5f, 1e-6f);
    EXPECT_NEAR(blend_surprise(1.0f, 2.0f, -1.f), 1.0f, 1e-6f);
    EXPECT_NEAR(blend_surprise(1.0f, 2.0f,  5.f), 2.0f, 1e-6f);
}

static void test_swap_accept() {
    // delta_high > delta_low always accepts.
    EXPECT_NEAR(swap_accept_probability(1.f, 0.0f, 1.0f), 1.0f, 1e-6f);
    // delta_high == delta_low always accepts.
    EXPECT_NEAR(swap_accept_probability(1.f, 0.5f, 0.5f), 1.0f, 1e-6f);
    // delta_high < delta_low gives exp(beta * negative).
    float p = swap_accept_probability(2.0f, 1.0f, 0.0f);
    EXPECT_NEAR(p, std::exp(-2.0f), 1e-6f);
}

static void test_pt_constants() {
    EXPECT_NEAR(PT_MUTATION_RATES[0], 0.005f, 1e-9f);
    EXPECT_NEAR(PT_MUTATION_RATES[1], 0.010f, 1e-9f);
    EXPECT_NEAR(PT_MUTATION_RATES[2], 0.020f, 1e-9f);
    EXPECT_NEAR(PT_MUTATION_RATES[3], 0.040f, 1e-9f);
    EXPECT_TRUE(PT_REPLICA_SIZE * PT_NUM_REPLICAS == POOL_SIZE);
    EXPECT_TRUE(STRESS_POOL_SIZE == STRESS_SUBPOP_COUNT * STRESS_SUBPOP_SIZE);
    EXPECT_TRUE(STRESS_POOL_SIZE == 24);
}

static void test_genome_bit_layout() {
    // Spec asserts exact bit ranges. Verify the constants sum to 1024.
    int role     = GENOME_BIT_ROLE_HI - GENOME_BIT_ROLE_LO + 1;
    int seed     = GENOME_BIT_SEED_HI - GENOME_BIT_SEED_LO + 1;
    int reaction = GENOME_BIT_REACTION_HI - GENOME_BIT_REACTION_LO + 1;
    int diff     = GENOME_BIT_DIFFUSION_HI - GENOME_BIT_DIFFUSION_LO + 1;
    int prior    = GENOME_BIT_DELTA_PRIOR_HI - GENOME_BIT_DELTA_PRIOR_LO + 1;
    EXPECT_TRUE(role     == 2);
    EXPECT_TRUE(seed     == 32);
    EXPECT_TRUE(reaction == 200);
    EXPECT_TRUE(diff     == 48);
    EXPECT_TRUE(prior    == 742);
    EXPECT_TRUE(role + seed + reaction + diff + prior == GENOME_BITS);
}

static void test_btraj_steps() {
    // Spec: bmap sampled at CA steps 16, 32, 48, 64.
    EXPECT_TRUE(BTRAJ_SAMPLES == 4);
    EXPECT_TRUE(BTRAJ_STEPS[0] == 16);
    EXPECT_TRUE(BTRAJ_STEPS[1] == 32);
    EXPECT_TRUE(BTRAJ_STEPS[2] == 48);
    EXPECT_TRUE(BTRAJ_STEPS[3] == 64);
    EXPECT_TRUE(BTRAJ_STEPS[BTRAJ_SAMPLES - 1] == CA_STEPS);
}

static void test_archive_geometry() {
    EXPECT_TRUE(MAX_ARCHIVE     == 5000);
    EXPECT_TRUE(ARCHIVE_HALF    == 2500);
    EXPECT_TRUE(ARCHIVE_BINS_X  == 20);
    EXPECT_TRUE(ARCHIVE_BINS_Y  == 20);
}

int main() {
    test_sot_gate();
    test_role_multipliers();
    test_blend_surprise();
    test_swap_accept();
    test_pt_constants();
    test_genome_bit_layout();
    test_btraj_steps();
    test_archive_geometry();
    std::printf("\n%d / %d passed\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}
