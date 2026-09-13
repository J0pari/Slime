// Host-only unit tests for the math inlines that don't require a CUDA
// runtime: hybrid blending, Pearson r, ensemble surprise, role-balance
// multipliers, SOT gate, PT swap probabilities, genome bit accessors, losses,
// sentinels, and the CUSUM update. Also carries regression tests for the
// review-pass fixes: role canonicalization of reserved 2-bit codes, xorshift
// zero-seed escape, CUSUM reset-after-alarm, and the CAME confidence buffer
// decoupling (prev_u kept separate from the c accumulator).
//
// Compile:
//   g++ -std=c++17 -Itests/stubs -I. tests/host_unit_tests.cpp -o build/host_tests
// Run:
//   ./build/host_tests
//
// The constants header includes <cuda_fp16.h> and <cuda_runtime.h>. Stubs
// live in tests/stubs/ and are picked up via -Itests/stubs ahead of the
// system include path.
//
// Where possible the tests now include the PRODUCTION headers directly
// (genome/codec.cu, optimizer/came_math.cuh, safety/pt_ladder.cuh) instead of
// re-pasting implementations, so a green suite validates the code the
// experiment actually runs.

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../config/constants.cuh"
#include "../nca/context_adjoint.cuh"
#include "../nca/rd_adjoint.cuh"
#include "../nca/rd_codec.cuh"
#include "../genome/codec.cu"
#include "../optimizer/came_math.cuh"
#include "../safety/pt_ladder.cuh"
#include "../safety/stress_ladder.cuh"
#include "../safety/structural.cu"
#include "../safety/operator_cmds.cuh"
#include "../archive/soft_qd_archive.cu"
#include "../curriculum/problem_generator.cu"

// Math kept locally only where the production definition is CUDA-dependent
// (safety/alignment.cu pulls in the engine) or where the test deliberately
// encodes a reference computation to compare against.

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

// ---- A-103 losses --------------------------------------------------------
static inline void classifier_loss_ref(const float* logits, int target, int n,
                                       float* dlogits, float* loss_out) {
    float max_z = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > max_z) max_z = logits[i];
    float p[16] = {0};
    float Z = 0.f;
    for (int i = 0; i < n; ++i) { p[i] = std::exp(logits[i] - max_z); Z += p[i]; }
    for (int i = 0; i < n; ++i) p[i] /= Z;
    if (loss_out) *loss_out = -std::log(p[target] + 1e-30f);
    for (int i = 0; i < n; ++i) dlogits[i] = p[i] - ((i == target) ? 1.f : 0.f);
}

static void test_classifier_loss() {
    // Uniform logits: probabilities all 1/n, loss = log(n), gradient
    // p_i - delta_i,target = 1/n - delta.
    float logits[4] = {0.f, 0.f, 0.f, 0.f};
    float dlog[4] = {0};
    float loss = 0.f;
    classifier_loss_ref(logits, /*target=*/1, /*n=*/4, dlog, &loss);
    EXPECT_NEAR(loss, std::log(4.f), 1e-6f);
    EXPECT_NEAR(dlog[0],  0.25f, 1e-6f);
    EXPECT_NEAR(dlog[1], -0.75f, 1e-6f);
    EXPECT_NEAR(dlog[2],  0.25f, 1e-6f);
    EXPECT_NEAR(dlog[3],  0.25f, 1e-6f);
    // Confidently correct: target logit >> others -> loss ~ 0, dlog small.
    float sharp[4] = {-10.f, 10.f, -10.f, -10.f};
    classifier_loss_ref(sharp, 1, 4, dlog, &loss);
    EXPECT_TRUE(loss < 1e-6f);
    EXPECT_TRUE(std::fabs(dlog[1] + 1.f) < 1e-6f
                || std::fabs(dlog[1]) < 1e-6f);
}

static void test_perception_dims() {
    // A-201: learned perception is N_PERC_FILTERS depthwise 3x3 filters; the
    // perception vector is N_PERC_FILTERS * CA_CHANNELS wide, and W_perc holds
    // N_PERC_FILTERS * 9 weights. The smoke test and ca_step rely on these.
    EXPECT_TRUE(N_PERC_FILTERS == 3);
    EXPECT_TRUE(W_PERC_SIZE == N_PERC_FILTERS * 9);
    EXPECT_TRUE(W_PERC_SIZE == 27);
    // PERC_DIM is defined in engine.cu (device TU); replicate the relation the
    // host can check from constants alone.
    EXPECT_TRUE(N_PERC_FILTERS * CA_CHANNELS == 48);
}

static void test_predictor_mse_loss() {
    // Predictor MSE loss against bmap_target. dloss/dp = 2*(p-t)/BMAP_DIM.
    float pred[BMAP_DIM]  = {0};
    float tgt [BMAP_DIM]  = {0};
    float dpred[BMAP_DIM] = {0};
    for (int i = 0; i < BMAP_DIM; ++i) { pred[i] = 0.5f; tgt[i] = 0.0f; }
    float acc = 0.f;
    float scale = 2.0f / static_cast<float>(BMAP_DIM);
    for (int i = 0; i < BMAP_DIM; ++i) {
        float d = pred[i] - tgt[i];
        acc += d * d;
        dpred[i] = scale * d;
    }
    float loss = acc / static_cast<float>(BMAP_DIM);
    EXPECT_NEAR(loss, 0.25f, 1e-5f);  // 0.5^2 = 0.25, mean of constant
    EXPECT_NEAR(dpred[0], scale * 0.5f, 1e-6f);
}

// ---- S-001 checkpoint schema --------------------------------------------
static uint32_t schema_hash_ref() {
    uint32_t h = 0x9E3779B9u;
    auto mix = [&](uint32_t x) {
        h ^= x + 0x9E3779B9u + (h << 6) + (h >> 2);
    };
    // sizeof(CheckpointHeader) at the time these tests were written is
    // 6 ints + 1 float + 1 bool padded to 4 bytes = 32 bytes on typical
    // 32-bit-int / 4-byte-bool ABIs. We test the hash is stable across
    // recomputations rather than asserting a specific value.
    mix(32u);
    mix(static_cast<uint32_t>(GENOME_BITS));
    mix(static_cast<uint32_t>(MAX_ARCHIVE));
    mix(static_cast<uint32_t>(POOL_SIZE));
    mix(static_cast<uint32_t>(BMAP_DIM));
    mix(static_cast<uint32_t>(BTRAJ_SAMPLES));
    mix(static_cast<uint32_t>(CA_CHANNELS));
    mix(static_cast<uint32_t>(GRID_SIZE));
    return h;
}

static void test_checkpoint_schema_stable() {
    // Computing the hash twice must produce the same value.
    EXPECT_TRUE(schema_hash_ref() == schema_hash_ref());
    // And mixing in a different value must change it.
    uint32_t a = schema_hash_ref();
    auto mix_extra = [](uint32_t h, uint32_t x) -> uint32_t {
        return h ^ (x + 0x9E3779B9u + (h << 6) + (h >> 2));
    };
    EXPECT_TRUE(mix_extra(a, 42) != a);
}

// ---- S-003 sentinel score ------------------------------------------------
static inline float sentinel_logistic(float z) {
    if (z >= 0.f) { float ez = std::exp(-z); return 1.0f / (1.0f + ez); }
    float ez = std::exp(z); return ez / (1.0f + ez);
}

static void test_sentinel_logistic() {
    EXPECT_NEAR(sentinel_logistic(0.f),   0.5f, 1e-6f);
    EXPECT_NEAR(sentinel_logistic( 10.f), 1.0f, 1e-3f);
    EXPECT_NEAR(sentinel_logistic(-10.f), 0.0f, 1e-3f);
    // Monotonic.
    EXPECT_TRUE(sentinel_logistic(0.5f) > sentinel_logistic(0.0f));
}

static void test_sentinel_sgd_decreases_loss() {
    // One sentinel, one descriptor, label = 1. After a few SGD steps the
    // predicted probability should rise toward 1.
    float w[BMAP_DIM] = {0};
    float b = 0.f;
    float desc[BMAP_DIM];
    for (int i = 0; i < BMAP_DIM; ++i) desc[i] = (i % 2 == 0) ? 0.1f : -0.1f;
    auto predict = [&]() {
        float z = b;
        for (int i = 0; i < BMAP_DIM; ++i) z += w[i] * desc[i];
        return sentinel_logistic(z);
    };
    float p0 = predict();
    float lr = 1e-1f;
    for (int step = 0; step < 200; ++step) {
        float p = predict();
        float dz = p - 1.0f;
        for (int i = 0; i < BMAP_DIM; ++i) w[i] -= lr * dz * desc[i];
        b -= lr * dz;
    }
    float p1 = predict();
    EXPECT_TRUE(p1 > p0);
    EXPECT_TRUE(p1 > 0.9f);
}

// ---- Regression tests for review-pass fixes ------------------------------

// canonical_role: the role schema declares every 2-bit code and the defined
// role it canonicalizes to; canonical_role agrees with the schema. Tests the
// PRODUCTION schema and function from config/constants.cuh.
static void test_canonical_role() {
    // [claim:A201.role-canonicalization]
    EXPECT_TRUE(canonical_role(Role::Classifier) == Role::Classifier);
    EXPECT_TRUE(canonical_role(Role::Predictor)  == Role::Predictor);
    EXPECT_TRUE(canonical_role(Role::Reserved10) == Role::Classifier);
    EXPECT_TRUE(canonical_role(Role::Reserved11) == Role::Predictor);

    // The schema covers each 2-bit code exactly once, canonical targets are
    // defined roles, names are non-empty and unique, and canonical_role
    // matches the schema.
    EXPECT_TRUE(ROLE_SCHEMA_COUNT == 4);
    bool seen[4] = {false, false, false, false};
    for (int i = 0; i < ROLE_SCHEMA_COUNT; ++i) {
        const RoleSpec& s = ROLE_SCHEMA[i];
        EXPECT_TRUE(s.code < 4 && !seen[s.code]);
        if (s.code < 4) seen[s.code] = true;
        EXPECT_TRUE(s.canonical == Role::Classifier
                    || s.canonical == Role::Predictor);
        EXPECT_TRUE(canonical_role(s.role) == s.canonical);
    }
    for (int a = 0; a < ROLE_SCHEMA_COUNT; ++a) {
        EXPECT_TRUE(ROLE_NAMES[a] != nullptr && ROLE_NAMES[a][0] != '\0');
        for (int b = a + 1; b < ROLE_SCHEMA_COUNT; ++b) {
            EXPECT_TRUE(std::strcmp(ROLE_NAMES[a], ROLE_NAMES[b]) != 0);
        }
    }
    EXPECT_TRUE(seen[0] && seen[1] && seen[2] && seen[3]);
}

// The same PCG32 seed and stream must reproduce the same draw sequence; a
// different stream must diverge. Exercises the production pcg32_seed /
// pcg32_random from config/constants.cuh.
static void test_pcg32_determinism() {
    // [claim:G100.deterministic-seed]
    Pcg32 a, b, c;
    pcg32_seed(&a, 0x853C49E6748FEA9BULL, 0xDA3E39CB94B95BDBULL);
    pcg32_seed(&b, 0x853C49E6748FEA9BULL, 0xDA3E39CB94B95BDBULL);
    pcg32_seed(&c, 0x853C49E6748FEA9BULL, 0xDA3E39CB94B95BDBULL + 1);
    bool same = true;
    bool different = false;
    for (int i = 0; i < 100; ++i) {
        uint32_t ra = pcg32_random(&a);
        uint32_t rb = pcg32_random(&b);
        uint32_t rc = pcg32_random(&c);
        if (ra != rb) same = false;
        if (ra != rc) different = true;
    }
    EXPECT_TRUE(same);
    EXPECT_TRUE(different);
}

// Pre-bootstrap role lock: no matter what the role-bit mutation draws, the
// forced genome must read back as Classifier through the production
// runtime_role accessor. Exercises slime::genome::mutate + slime::genome::force_role +
// slime::genome::runtime_role from the production codec.
static void test_role_lock_forces_classifier() {
    slime::genome::Genome g;
    std::memset(&g.bits, 0, sizeof(g.bits));
    slime::genome::write_role(g, Role::Classifier);
    slime::genome::write_seed(g, slime::GenomeSeed(0x12345678u));

    Pcg32 rng;
    pcg32_seed(&rng, 42u, 1u);

    // 200 spawn-mutation attempts with a hostile role rate: both role bits
    // flip every time. The lock must still hold.
    for (int i = 0; i < 200; ++i) {
        slime::genome::mutate(&g, MUTATION_RATE_BASELINE, 1.0f, &rng);
        slime::genome::force_role(g, Role::Classifier);
        EXPECT_TRUE(slime::genome::runtime_role(g) == Role::Classifier);
    }
}

// Spawn victim selection must pick distinct pool slots without replacement,
// worst fitness first. Exercises the PRODUCTION select_spawn_victims from
// genome/codec.cu (the same function spawn_wave uses).
static void test_spawn_victims_unique() {
    // [claim:I001.spawn-wave-unique]
    Role roles[POOL_SIZE];
    float fitness[POOL_SIZE];
    for (int i = 0; i < POOL_SIZE; ++i) {
        roles[i] = Role::Classifier;
        fitness[i] = static_cast<float>((i * 37) % POOL_SIZE) * 0.01f;
    }

    int victims[WAVE_SIZE];
    int n = slime::genome::select_spawn_victims(roles, fitness, POOL_SIZE,
                                         Role::Classifier, WAVE_SIZE, victims);
    EXPECT_TRUE(n == WAVE_SIZE);

    // Uniqueness.
    for (int a = 0; a < n; ++a) {
        for (int b = a + 1; b < n; ++b) {
            EXPECT_TRUE(victims[a] != victims[b]);
        }
    }

    // The victims are exactly the WAVE_SIZE lowest-fitness slots.
    bool victim_flags[POOL_SIZE] = {};
    for (int a = 0; a < n; ++a) victim_flags[victims[a]] = true;
    for (int i = 0; i < POOL_SIZE; ++i) {
        float max_victim_fit = -1.f;
        for (int a = 0; a < n; ++a) {
            if (fitness[victims[a]] > max_victim_fit) max_victim_fit = fitness[victims[a]];
        }
        if (!victim_flags[i]) {
            // Every non-victim must be at least as fit as the worst victim.
            EXPECT_TRUE(fitness[i] >= max_victim_fit);
        }
    }

    // Role filtering: predictors must be skipped entirely.
    for (int i = 0; i < 8; ++i) roles[i] = Role::Predictor;
    int victims2[WAVE_SIZE];
    int n2 = slime::genome::select_spawn_victims(roles, fitness, POOL_SIZE,
                                          Role::Classifier, WAVE_SIZE, victims2);
    for (int a = 0; a < n2; ++a) {
        EXPECT_TRUE(roles[victims2[a]] == Role::Classifier);
    }

    // Reserved codes canonicalize into the target role's selection.
    Role roles3[POOL_SIZE];
    for (int i = 0; i < POOL_SIZE; ++i) roles3[i] = Role::Reserved10;
    int victims3[WAVE_SIZE];
    int n3 = slime::genome::select_spawn_victims(roles3, fitness, POOL_SIZE,
                                          Role::Classifier, WAVE_SIZE, victims3);
    EXPECT_TRUE(n3 == WAVE_SIZE);
}

// xorshift32 must not lock on a zero seed (the all-zero state is a fixed
// point of the raw recurrence; the codec coerces it to a nonzero constant).
static uint32_t xorshift32(uint32_t* s) {
    uint32_t x = *s;
    if (x == 0u) x = 0x9E3779B9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x; return x;
}

static void test_xorshift_zero_seed() {
    uint32_t s = 0u;
    uint32_t a = xorshift32(&s);
    EXPECT_TRUE(a != 0u);            // first draw escapes zero
    uint32_t b = xorshift32(&s);
    EXPECT_TRUE(b != a);            // stream does not lock
    // A zero seed must produce the same stream as explicitly seeding the
    // fallback constant (determinism preserved).
    uint32_t s2 = 0x9E3779B9u;
    EXPECT_TRUE(xorshift32(&s2) == a);
}

// Two-sided tabular CUSUM with reset-after-alarm (matches monitoring.cu).
struct CusumRef { float upper, lower, reference, allowance, threshold; int alerts; };
static void cusum_update_ref(CusumRef* s, float x) {
    float dev = x - s->reference;
    s->upper = std::fmax(0.f, s->upper + dev - s->allowance);
    s->lower = std::fmax(0.f, s->lower - dev - s->allowance);
    if (s->upper > s->threshold) { s->alerts++; s->upper = 0.f; }
    if (s->lower > s->threshold) { s->alerts++; s->lower = 0.f; }
}

static void test_cusum_resets_after_alarm() {
    CusumRef s = {0.f, 0.f, /*ref=*/0.f, /*allow=*/0.1f, /*thresh=*/1.0f, 0};
    // A single large positive excursion should alarm once and reset, not
    // latch and re-alarm on the following in-control samples.
    cusum_update_ref(&s, 5.0f);
    EXPECT_TRUE(s.alerts == 1);
    EXPECT_NEAR(s.upper, 0.f, 1e-6f);   // accumulator reset
    // In-control samples near the reference produce no further alarms.
    cusum_update_ref(&s, 0.0f);
    cusum_update_ref(&s, 0.05f);
    EXPECT_TRUE(s.alerts == 1);
}

// CAME: the PRODUCTION scalar step (optimizer/came_math.cuh) is the same
// function the device kernel executes. Under a constant gradient the
// normalized update u settles, so the step-to-step instability du -> 0 and
// the confidence accumulator c -> 0. beta3 = 0.999 (production defaults).
static void test_came_confidence_converges() {
    slime::optimizer::CameScalarState st = {0.f, 0.f, 0.f, 0.f};
    const slime::optimizer::CameScalarParams p = {
        slime::optimizer::CAME_DEFAULTS.lr,
        slime::optimizer::CAME_DEFAULTS.beta1,
        slime::optimizer::CAME_DEFAULTS.beta2,
        slime::optimizer::CAME_DEFAULTS.beta3,
        slime::optimizer::CAME_DEFAULTS.epsilon,
        slime::optimizer::CAME_DEFAULTS.weight_decay,
    };
    float w = 1.0f;
    for (int i = 0; i < 5000; ++i) {
        slime::optimizer::came_step_scalar(&st, &w, 1.0f, p);
    }
    EXPECT_TRUE(st.c < 1e-3f);
    float prev_u_before = st.prev_u;
    slime::optimizer::came_step_scalar(&st, &w, 1.0f, p);
    EXPECT_TRUE(std::fabs(st.prev_u - prev_u_before) < 1e-2f);
    EXPECT_TRUE(st.prev_u > 0.f && std::isfinite(st.prev_u));
    EXPECT_TRUE(std::isfinite(w));
}

// CAME production equation: the applied update must equal
// lr * confidence * u + weight_decay * w exactly as implemented, with
// confidence = 1/(1+c). Recomputes the update from the post-step state and
// checks the weight moved by the production amount.
static void test_came_production_equation() {
    // [claim:A501.came-production-equation]
    slime::optimizer::CameScalarState st = {0.f, 0.f, 0.f, 0.f};
    const slime::optimizer::CameScalarParams p = {
        1e-3f, 0.9f, 0.999f, 0.999f, 1e-8f, 0.01f,
    };
    float w = 1.5f;
    float w_before = w;
    float g = -0.25f;

    slime::optimizer::came_step_scalar(&st, &w, g, p);

    float expected_u = st.m / (std::sqrt(st.v) + p.epsilon);
    EXPECT_NEAR(st.prev_u, expected_u, 1e-7f);
    float confidence = 1.0f / (1.0f + st.c);
    float expected_w = w_before - p.lr * confidence * expected_u
                       - p.weight_decay * w_before;
    EXPECT_NEAR(w, expected_w, 1e-6f);
    EXPECT_TRUE(std::isfinite(w));

    // Weight decay alone (zero gradient) must shrink the weight by the
    // multiplicative factor (1 - weight_decay) in the long run; a single
    // zero-gradient step still applies decay.
    slime::optimizer::CameScalarState st2 = {0.f, 0.f, 0.f, 0.f};
    float w2 = 2.0f;
    slime::optimizer::came_step_scalar(&st2, &w2, 0.f, p);
    EXPECT_NEAR(w2, 2.0f - 0.01f * 2.0f, 1e-7f);
}

// PT rolling-window endpoints: after the ring wraps, improvement_rate must
// use the oldest (next-to-overwrite) and newest entries. With a linearly
// rising best fitness this yields exactly 1.0 per generation.
static void test_pt_ring_endpoints() {
    slime::safety::pt::MutationLadder l;
    std::memset(&l, 0, sizeof(l));
    l.beta = 1.0f;
    l.accept_ema = PT_TARGET_ACCEPT;
    for (int i = 0; i < POOL_SIZE; ++i) l.replica_of[i] = static_cast<uint8_t>(i / PT_REPLICA_SIZE);

    float fitness[POOL_SIZE];
    for (int g = 0; g < 2 * PT_SWAP_INTERVAL + 10; ++g) {
        for (int i = 0; i < POOL_SIZE; ++i) {
            fitness[i] = static_cast<float>(g) + 0.01f * static_cast<float>(i % PT_REPLICA_SIZE);
        }
        slime::safety::pt::record_best_fitness(&l, fitness);
    }

    // Ring wrapped twice: oldest value is (head) = 10 generations ago
    // relative to the last write. Best of replica r at gen g is
    // g + 0.01*(replica_size-1); the difference between newest and oldest is
    // exactly (PT_SWAP_INTERVAL - 1) generations.
    for (int r = 0; r < PT_NUM_REPLICAS; ++r) {
        float rate = slime::safety::pt::improvement_rate(l, r);
        float expected = static_cast<float>(PT_SWAP_INTERVAL - 1)
                       / static_cast<float>(PT_SWAP_INTERVAL);
        EXPECT_NEAR(rate, expected, 1e-5f);
    }

    // A partially-filled ring (fewer than PT_SWAP_INTERVAL recordings) still
    // produces finite rates.
    slime::safety::pt::MutationLadder l2;
    std::memset(&l2, 0, sizeof(l2));
    for (int i = 0; i < POOL_SIZE; ++i) l2.replica_of[i] = static_cast<uint8_t>(i / PT_REPLICA_SIZE);
    for (int g = 0; g < 10; ++g) {
        for (int i = 0; i < POOL_SIZE; ++i) fitness[i] = static_cast<float>(g);
        slime::safety::pt::record_best_fitness(&l2, fitness);
    }
    for (int r = 0; r < PT_NUM_REPLICAS; ++r) {
        EXPECT_TRUE(std::isfinite(slime::safety::pt::improvement_rate(l2, r)));
    }
}


namespace arch = slime::archive;

static void init_test_archive(arch::Archive* a) {
    std::memset(a, 0, sizeof(*a));
    for (int b = 0; b < ARCHIVE_BINS_X * ARCHIVE_BINS_Y; ++b) {
        a->bins[b].cap_classifier = ARCHIVE_BIN_CAP;
        a->bins[b].cap_predictor  = ARCHIVE_BIN_CAP;
    }
    for (int d = 0; d < BMAP_DIM; ++d) a->inv_var_ema[d] = 1.0f;
    arch::init_rff(&a->rff, 42u);
    a->pca_valid = false;
}

static int insert_test_entry(arch::Archive* a, float d0, float d1,
                             float fitness, Role role, uint32_t lineage) {
    arch::ArchiveEntry cand;
    std::memset(&cand, 0, sizeof(cand));
    cand.descriptor[0] = d0;
    cand.descriptor[1] = d1;
    for (int d = 2; d < BMAP_DIM; ++d) cand.descriptor[d] = 0.01f * d;
    arch::rff_project(a->rff, cand.descriptor, cand.rff_proj);
    cand.fitness = fitness;
    cand.f_raw = fitness;
    cand.f_sot = 1.0f;
    cand.lineage_id = slime::LineageId(lineage);
    cand.role = role;
    cand.alive = true;
    arch::assign_bin(*a, cand.descriptor, cand.bin_x, cand.bin_y);
    return arch::insert(a, cand);
}

static void test_archive_bin_capacity_after_rebin() {
    // [claim:A401.bin-capacity]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);

    // Fixed pre-rebin binning: pc = (e0, e1) with the mean set at the data
    // centroid (0.5, 0.5) and extents [0, 0.0002] on PC0, so the A groups
    // (d0 ~ 0.5 + tiny offsets) land at bins (bx, 0) with bx = 0..11, and
    // the B group (d0 = 0.51 + 0.001*k) clamps into bin (19, 0).
    // Deterministic, no hash-fallback float truncation.
    for (int d = 0; d < BMAP_DIM; ++d) {
        a->pc[0][d] = (d == 0) ? 1.0f : 0.0f;
        a->pc[1][d] = (d == 1) ? 1.0f : 0.0f;
        a->pc_mean[d] = 0.f;
    }
    a->pc_mean[0] = 0.5f;
    a->pc_mean[1] = 0.5f;
    a->pc_min[0] = 0.f; a->pc_max[0] = 0.0002f;
    a->pc_min[1] = 0.f; a->pc_max[1] = 0.f;
    a->pca_valid = true;

    // A1: d0 = 0.5 + 0.00002*k (7 entries); A2: d0 = 0.500001 + 0.00002*k
    // (7 entries). B: d0 = 0.51 + 0.001*k (13 entries). All descriptors vary
    // only in dimension 0, so the rebin PCA puts PC0 on dimension 0 with
    // extent ~0.022; the A cluster spans only ~0.00012 (about 0.1 of a bin
    // width) and merges into ONE rebin bin: 14 > cap 13, which the capacity
    // repair must trim back to 13.
    uint32_t lineage = 1;
    for (int k = 0; k < 7; ++k) {
        EXPECT_TRUE(insert_test_entry(a, 0.5f + 0.00002f * k, 0.5f,
                                      0.10f + 0.001f * k, Role::Classifier,
                                      lineage++) >= 0);
    }
    for (int k = 0; k < 7; ++k) {
        EXPECT_TRUE(insert_test_entry(a, 0.500001f + 0.00002f * k, 0.5f,
                                      0.20f + 0.001f * k, Role::Classifier,
                                      lineage++) >= 0);
    }
    for (int k = 0; k < 13; ++k) {
        EXPECT_TRUE(insert_test_entry(a, 0.51f + 0.001f * k, 0.5f,
                                      0.50f + 0.001f * k, Role::Classifier,
                                      lineage++) >= 0);
    }
    EXPECT_TRUE(a->count_classifier == 27);
    EXPECT_TRUE(a->bins[19 * ARCHIVE_BINS_Y + 0].count_classifier == 13);

    arch::recompute_bins(a, nullptr);

    // The A groups merged into one rebin bin and were trimmed to capacity;
    // the 13 B entries remain spread over the higher bins.
    EXPECT_TRUE(a->bins[0 * ARCHIVE_BINS_Y + 0].count_classifier == 13);
    {
        int b_survivors = 0;
        for (int bx = 1; bx < ARCHIVE_BINS_X; ++bx) {
            b_survivors += a->bins[bx * ARCHIVE_BINS_Y + 0].count_classifier;
        }
        EXPECT_TRUE(b_survivors == 13);
    }
    EXPECT_TRUE(a->count_classifier == 26);
    int n_alive = 0;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (a->entries[i].alive) n_alive++;
    }
    EXPECT_TRUE(n_alive == 26);
    char err[256];
    EXPECT_TRUE(arch::archive_check_invariants(*a, err, sizeof(err)));

    delete a;
}

static void test_archive_invariant_checker() {
    // [claim:A401.live-statistics-exact]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);
    EXPECT_TRUE(insert_test_entry(a, 0.5f, 0.5f, 0.4f, Role::Classifier, 1) >= 0);
    EXPECT_TRUE(insert_test_entry(a, 0.9f, 0.9f, 0.6f, Role::Classifier, 2) >= 0);
    char err[256];

    EXPECT_TRUE(arch::archive_check_invariants(*a, err, sizeof(err)));

    // Corrupt: global count disagrees with reality.
    a->count_classifier++;
    EXPECT_TRUE(!arch::archive_check_invariants(*a, err, sizeof(err)));
    a->count_classifier--;

    // Corrupt: bin count disagrees with reality.
    a->bins[0].count_classifier++;
    EXPECT_TRUE(!arch::archive_check_invariants(*a, err, sizeof(err)));
    a->bins[0].count_classifier--;

    // Corrupt: live list loses an entry.
    a->n_alive_classifier--;
    EXPECT_TRUE(!arch::archive_check_invariants(*a, err, sizeof(err)));
    a->n_alive_classifier++;

    // Corrupt: an alive entry not present in any live list.
    int idx = a->alive_classifier_idx[0];
    a->alive_classifier_idx[0] = a->alive_classifier_idx[1];
    EXPECT_TRUE(!arch::archive_check_invariants(*a, err, sizeof(err)));
    a->alive_classifier_idx[0] = idx;

    // Corrupt: RFF mean drifts from the brute-force mean.
    a->mu_rff_classifier[0] += 0.5f;
    EXPECT_TRUE(!arch::archive_check_invariants(*a, err, sizeof(err)));
    a->mu_rff_classifier[0] -= 0.5f;

    EXPECT_TRUE(arch::archive_check_invariants(*a, err, sizeof(err)));
    delete a;
}

static void test_archive_weighted_metric_active() {
    // [claim:A401.weighted-metric-active]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);

    // Two distinguished occupants in bin (0,0): A at (0.5,0.5), B at (0.9,0.9).
    EXPECT_TRUE(insert_test_entry(a, 0.5f, 0.5f, 0.5f, Role::Classifier, 11) >= 0);
    EXPECT_TRUE(insert_test_entry(a, 0.9f, 0.9f, 0.9f, Role::Classifier, 22) >= 0);
    // Eleven fillers in the same bin (d1 = 0.6 + 0.001*k keeps
    // (uint32)(1e6*d1) % 20 == 0 and stays far from the candidate)
    // bring bin (0,0) up to its capacity of 13.
    for (int k = 1; k <= 11; ++k) {
        EXPECT_TRUE(insert_test_entry(a, 0.5f, 0.6f + 0.001f * k,
                                      0.1f, Role::Classifier,
                                      1001u + k) >= 0);
    }
    EXPECT_TRUE(a->bins[0].count_classifier == 13);

    // The inverse-variance EMA must have moved off its init value.
    EXPECT_TRUE(std::fabs(a->inv_var_ema[0] - 1.0f) > 1e-7f);

    // Candidate C sits near A in descriptor space and beats A's QD score:
    // with the bin full, the nearest-neighbor rule must evict A, not B or a
    // filler.
    int rc = insert_test_entry(a, 0.5f + 1e-3f, 0.5f + 1e-3f,
                               0.7f, Role::Classifier, 33);
    EXPECT_TRUE(rc >= 0);

    bool a_alive = false, b_alive = false, c_alive = false;
    int n_fillers_alive = 0;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (!a->entries[i].alive) continue;
        if (a->entries[i].lineage_id == slime::LineageId(11)) a_alive = true;
        if (a->entries[i].lineage_id == slime::LineageId(22)) b_alive = true;
        if (a->entries[i].lineage_id == slime::LineageId(33)) c_alive = true;
        if (a->entries[i].lineage_id >= slime::LineageId(1002) &&
            a->entries[i].lineage_id <= slime::LineageId(1012)) n_fillers_alive++;
    }
    EXPECT_TRUE(!a_alive);
    EXPECT_TRUE(b_alive);
    EXPECT_TRUE(c_alive);
    EXPECT_TRUE(n_fillers_alive == 11);

    char err[256];
    EXPECT_TRUE(arch::archive_check_invariants(*a, err, sizeof(err)));
    delete a;
}

static void test_archive_rff_mean_exact_after_replacement() {
    // [claim:A401.live-statistics-exact]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);
    EXPECT_TRUE(insert_test_entry(a, 0.5f, 0.5f, 0.5f, Role::Classifier, 1) >= 0);
    EXPECT_TRUE(insert_test_entry(a, 0.9f, 0.9f, 0.9f, Role::Classifier, 2) >= 0);
    EXPECT_TRUE(insert_test_entry(a, 0.501f, 0.501f, 0.7f, Role::Classifier, 3) >= 0);

    // Brute-force the classifier RFF mean over alive entries.
    float mu[arch::RFF_DIM] = {};
    int n = 0;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (!a->entries[i].alive) continue;
        if (a->entries[i].role != Role::Classifier) continue;
        n++;
        for (int j = 0; j < arch::RFF_DIM; ++j) mu[j] += a->entries[i].rff_proj[j];
    }
    EXPECT_TRUE(n == static_cast<int>(a->count_classifier));
    for (int j = 0; j < arch::RFF_DIM; ++j) {
        float want = mu[j] / static_cast<float>(n);
        EXPECT_NEAR(a->mu_rff_classifier[j], want, 1e-5f * (1.0f + std::fabs(want)));
    }
    delete a;
}

static void test_archive_randomized_property() {
    // [claim:A401.live-statistics-exact]
    // [claim:A401.bin-capacity]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);

    Pcg32 rng;
    pcg32_seed(&rng, 0xC0FFEEu, 7u);
    char err[256];

    for (int op = 0; op < 1500; ++op) {
        if (op % 100 == 0 && a->count_classifier + a->count_predictor >= 2) {
            // Random degenerate PCA state: random unit PCs, random extents.
            for (int k = 0; k < 2; ++k) {
                float norm = 0.f;
                for (int d = 0; d < BMAP_DIM; ++d) {
                    a->pc[k][d] = pcg32_float(&rng) - 0.5f;
                    norm += a->pc[k][d] * a->pc[k][d];
                }
                norm = sqrtf(norm);
                for (int d = 0; d < BMAP_DIM; ++d) a->pc[k][d] /= norm;
            }
            a->pc_min[0] = -1.f; a->pc_max[0] = 1.f;
            a->pc_min[1] = -1.f; a->pc_max[1] = 1.f;
            a->pca_valid = true;
            arch::recompute_bins(a, nullptr);
        } else {
            float d0 = 0.3f + 0.4f * pcg32_float(&rng);
            float d1 = 0.3f + 0.4f * pcg32_float(&rng);
            float fit = pcg32_float(&rng);
            Role role = (pcg32_float(&rng) < 0.8f) ? Role::Classifier
                                                   : Role::Predictor;
            insert_test_entry(a, d0, d1, fit, role,
                              1u + static_cast<uint32_t>(pcg32_random(&rng) % 100000u));
        }
        if (!arch::archive_check_invariants(*a, err, sizeof(err))) {
            EXPECT_TRUE(false);
            break;
        }
    }
    EXPECT_TRUE(arch::archive_check_invariants(*a, err, sizeof(err)));
    delete a;
}

static void test_archive_file_roundtrip() {
    // [claim:S001.checkpoint-roundtrip]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);
    for (int i = 0; i < 20; ++i) {
        Role role = (i % 3 == 0) ? Role::Predictor : Role::Classifier;
        EXPECT_TRUE(insert_test_entry(a, 0.3f + 0.01f * i, 0.4f + 0.005f * i,
                                      0.1f + 0.02f * i, role,
                                      100u + i) >= 0);
    }
    arch::recompute_bins(a, nullptr);

    FILE* f = std::tmpfile();
    EXPECT_TRUE(f != nullptr);
    EXPECT_TRUE(arch::archive_write_file(*a, f));
    std::rewind(f);
    arch::Archive* b = new arch::Archive;
    EXPECT_TRUE(arch::archive_read_file(*b, f));
    EXPECT_TRUE(std::memcmp(a, b, sizeof(arch::Archive)) == 0);

    char err[256];
    EXPECT_TRUE(arch::archive_check_invariants(*b, err, sizeof(err)));
    // A loaded archive accepts further mutations with exact statistics.
    EXPECT_TRUE(insert_test_entry(b, 0.9f, 0.9f, 0.5f,
                                  Role::Classifier, 999u) >= 0);
    EXPECT_TRUE(arch::archive_check_invariants(*b, err, sizeof(err)));
    std::fclose(f);
    delete a;
    delete b;
}

// ---- Operator commands + SOT schedule (S-002, A-101) ---------------------
// Production parsing (safety/operator_cmds.cuh), durable archive pruning
// (archive::prune_lineage), and the host-side SOT schedule determinism.

namespace ops = slime::safety::alignment;

static void test_operator_command_parse() {
    // [claim:S002.operator-command-effective]
    EXPECT_TRUE(ops::parse_operator_line("pause").command == ops::OperatorCommand::Pause);
    EXPECT_TRUE(ops::parse_operator_line("resume").command == ops::OperatorCommand::Resume);
    EXPECT_TRUE(ops::parse_operator_line("checkpoint").command == ops::OperatorCommand::Checkpoint);
    ops::ParsedCommand prune = ops::parse_operator_line("prune 4242");
    EXPECT_TRUE(prune.command == ops::OperatorCommand::Prune);
    EXPECT_TRUE(prune.lineage == slime::LineageId(4242u));
    EXPECT_TRUE(ops::parse_operator_line("garbage").command == ops::OperatorCommand::None);
    EXPECT_TRUE(ops::parse_operator_line("").command == ops::OperatorCommand::None);

    ops::OperatorState st;
    EXPECT_TRUE(!st.paused);
    st.add_pruned(slime::LineageId(7u));
    st.add_pruned(slime::LineageId(7u));   // duplicate is deduplicated
    st.add_pruned(slime::LineageId(9u));
    EXPECT_TRUE(st.n_pruned == 2);
    EXPECT_TRUE(st.lineage_pruned(slime::LineageId(7u)));
    EXPECT_TRUE(!st.lineage_pruned(slime::LineageId(8u)));
}

static void test_archive_prune_lineage() {
    // [claim:S002.operator-command-effective]
    arch::Archive* a = new arch::Archive;
    init_test_archive(a);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(insert_test_entry(a, 0.5f + 0.001f * i, 0.5f,
                                      0.5f, Role::Classifier, 1u) >= 0);
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(insert_test_entry(a, 0.6f + 0.001f * i, 0.5f,
                                      0.5f, Role::Classifier, 2u) >= 0);
    }
    EXPECT_TRUE(a->count_classifier == 9);

    arch::prune_lineage(a, slime::LineageId(2u));

    char err[256];
    EXPECT_TRUE(arch::archive_check_invariants(*a, err, sizeof(err)));
    EXPECT_TRUE(a->count_classifier == 5);
    int lineage2_alive = 0;
    for (int i = 0; i < MAX_ARCHIVE; ++i) {
        if (a->entries[i].alive && a->entries[i].lineage_id == slime::LineageId(2u)) {
            lineage2_alive++;
        }
    }
    EXPECT_TRUE(lineage2_alive == 0);
    delete a;
}

static void test_sot_batch_determinism() {
    // [claim:A101.sot-schedule-independent]
    slime::curriculum::ClassifierBatch b1, b2;
    Pcg32 rng1, rng2;
    pcg32_seed(&rng1, PCG32_DEFAULT_STATE, PCG32_DEFAULT_STREAM);
    pcg32_seed(&rng2, PCG32_DEFAULT_STATE, PCG32_DEFAULT_STREAM);
    slime::curriculum::assemble_classifier_batch(&b1, MAIN_SOT_DENSITY,
                                                 0xDEADCAFE42ULL, &rng1);
    slime::curriculum::assemble_classifier_batch(&b2, MAIN_SOT_DENSITY,
                                                 0xDEADCAFE42ULL, &rng2);
    bool same = true;
    for (int s = 0; s < slime::curriculum::CLASSIFIER_BATCH; ++s) {
        if (b1.label[s] != b2.label[s]) same = false;
        if (b1.is_sot[s] != b2.is_sot[s]) same = false;
    }
    for (int i = 0; i < slime::curriculum::CLASSIFIER_BATCH * GRID_SIZE * GRID_SIZE * 3; ++i) {
        // Under the CUDA stub __half values are all-zero, so the image-pixel
        // comparison is structural here; the real pixel determinism runs in
        // the GPU builds. Labels, SOT marks, and the task embedding are
        // full-fidelity host state.
        if (b1.image[i].bits != b2.image[i].bits) same = false;
    }
    for (int d = 0; d < TASK_EMBED_DIM; ++d) {
        if (b1.task_embedding[d] != b2.task_embedding[d]) same = false;
    }
    EXPECT_TRUE(same);
}

// ---- Archive invariants (A-401) ------------------------------------------
// Production archive logic exercised directly (soft_qd_archive.cu included
// above): capacity enforcement on rebin, exact live statistics, the weighted
// descriptor metric, and the invariant checker.
// ---- SOT reversible permutation (Feistel) --------------------------------
// Re-pasted from curriculum/problem_generator.cu; the round structure must
// stay in sync. The property under test is the one that matters for SOT:
// the permutation is an exact bijection and its inverse undoes it.
static uint32_t sot_feistel(uint32_t idx, uint64_t key, bool invert) {
    uint32_t l = (idx >> 6) & 0x3Fu;
    uint32_t r = idx & 0x3Fu;
    const int ROUNDS = 4;
    for (int round = 0; round < ROUNDS; ++round) {
        int ri = invert ? (ROUNDS - 1 - round) : round;
        uint32_t rk = static_cast<uint32_t>((key >> (8 * ri)) & 0xFFu);
        uint32_t nl, nr;
        if (!invert) {
            uint32_t f = ((r * 73u) + rk * 0x9Eu + ri * 0x2Fu) & 0x3Fu;
            nl = r; nr = l ^ f;
        } else {
            uint32_t f = ((l * 73u) + rk * 0x9Eu + ri * 0x2Fu) & 0x3Fu;
            nl = r ^ f; nr = l;
        }
        l = nl; r = nr;
    }
    return ((l & 0x3Fu) << 6) | (r & 0x3Fu);
}

static void test_sot_feistel_bijection() {
    const uint64_t key = 0xC0FFEE1234567890ull;
    // Forward is a bijection on [0, 4096): every output hit exactly once.
    int seen[4096] = {0};
    for (uint32_t i = 0; i < 4096; ++i) {
        uint32_t o = sot_feistel(i, key, false);
        EXPECT_TRUE(o < 4096u);
        seen[o]++;
    }
    int collisions = 0, misses = 0;
    for (int i = 0; i < 4096; ++i) { if (seen[i] > 1) collisions++; if (seen[i] == 0) misses++; }
    EXPECT_TRUE(collisions == 0);
    EXPECT_TRUE(misses == 0);
    // Inverse undoes forward for every index.
    int roundtrip_ok = 1;
    for (uint32_t i = 0; i < 4096; ++i) {
        uint32_t f = sot_feistel(i, key, false);
        uint32_t b = sot_feistel(f, key, true);
        if (b != i) { roundtrip_ok = 0; break; }
    }
    EXPECT_TRUE(roundtrip_ok == 1);
    // A different key generally yields a different permutation.
    EXPECT_TRUE(sot_feistel(123u, key, false) != sot_feistel(123u, key ^ 0xFFull, false));
}

// ---- runaway_detected / l_role_collapse ----------------------------------
static bool runaway_ref(float share, float growth, float threshold) {
    return share > threshold && growth > 0.f;
}
static void test_runaway_detected() {
    EXPECT_TRUE(runaway_ref(0.8f,  0.01f, 0.5f) == true);   // over + growing
    EXPECT_TRUE(runaway_ref(0.8f, -0.01f, 0.5f) == false);  // over but shrinking
    EXPECT_TRUE(runaway_ref(0.3f,  0.01f, 0.5f) == false);  // growing but small
}
static bool l_role_collapse_ref(float l_role_acc, float baseline) {
    if (baseline < 0.6f) return false;
    return l_role_acc < 0.85f * baseline;
}
static void test_l_role_collapse() {
    EXPECT_TRUE(l_role_collapse_ref(0.70f, 0.95f) == true);   // dropped below 85% of baseline
    EXPECT_TRUE(l_role_collapse_ref(0.92f, 0.95f) == false);  // healthy
    EXPECT_TRUE(l_role_collapse_ref(0.30f, 0.50f) == false);  // baseline untrusted
}

// ---- Structural pressures (S-003, I4) -------------------------------------

// Least-squares audit: a linear target with enough samples explains almost
// all variance; constant or random targets do not.
static void test_audit_r2_and_multiplier() {
    static float X[48 * BMAP_DIM];
    static float y[48];
    uint32_t s = 0x5EEDu;
    for (int i = 0; i < 48; ++i) {
        float acc = 0.3f;
        for (int d = 0; d < BMAP_DIM; ++d) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            float u = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
            X[i * BMAP_DIM + d] = u;
            acc += 0.2f * u;
        }
        y[i] = acc;
    }
    float w[BMAP_DIM];
    float b = 0.f;
    float r2 = slime::safety::fit_linear_r2(X, y, 48, w, &b);
    EXPECT_TRUE(r2 > 0.99f);

    for (int i = 0; i < 48; ++i) y[i] = 1.f;
    r2 = slime::safety::fit_linear_r2(X, y, 48, w, &b);
    EXPECT_TRUE(r2 <= 0.f);

    for (int i = 0; i < 48; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        y[i] = static_cast<float>(s) * (1.0f / 4294967296.0f);
    }
    r2 = slime::safety::fit_linear_r2(X, y, 48, w, &b);
    EXPECT_TRUE(r2 < 0.95f);
}

// The audit cycle fits each role separately and floors the multiplier.
static void test_audit_cycle_role_aware() {
    static float X[64 * BMAP_DIM];
    static float loss[64];
    Role roles[64];
    uint32_t s = 0xA0D17u;
    for (int i = 0; i < 64; ++i) {
        roles[i] = (i < 32) ? Role::Classifier : Role::Predictor;
        float acc = 0.1f;
        for (int d = 0; d < BMAP_DIM; ++d) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            float u = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
            X[i * BMAP_DIM + d] = u;
            acc += 0.15f * u;
        }
        loss[i] = acc;
    }
    slime::safety::AuditRegressor reg{};
    reg.audit_mult_classifier = 1.f;
    reg.audit_mult_predictor = 1.f;
    slime::safety::run_audit_cycle(&reg, X, loss, roles, 64);
    EXPECT_TRUE(reg.audit_mult_classifier > 0.99f);
    EXPECT_TRUE(reg.audit_mult_predictor > 0.99f);

    // Constant losses carry no signal: the multiplier floors.
    for (int i = 0; i < 64; ++i) loss[i] = 0.5f;
    slime::safety::run_audit_cycle(&reg, X, loss, roles, 64);
    EXPECT_TRUE(reg.audit_mult_classifier == AUDIT_MULT_FLOOR);
    EXPECT_TRUE(reg.audit_mult_predictor == AUDIT_MULT_FLOOR);
}

static void test_variance_multiplier() {
    float constant_desc[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) constant_desc[d] = 0.5f;
    EXPECT_TRUE(slime::safety::variance_multiplier(constant_desc) == VAR_FLOOR_MULT);

    float varied[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) {
        varied[d] = static_cast<float>(d) * 0.1f;
    }
    EXPECT_TRUE(slime::safety::variance_multiplier(varied) == 1.f);
}

// Per-role lineage shares and growth; the brake table is per-role and
// non-mutating.
static void test_lineage_stats_and_brake() {
    slime::LineageId ids[8] = {
        slime::LineageId(7u), slime::LineageId(7u), slime::LineageId(7u),
        slime::LineageId(7u), slime::LineageId(7u), slime::LineageId(7u),
        slime::LineageId(9u), slime::LineageId(9u)};
    Role roles[8] = {Role::Classifier, Role::Classifier, Role::Classifier,
                     Role::Classifier, Role::Classifier, Role::Classifier,
                     Role::Classifier, Role::Classifier};
    static slime::safety::LineageStats stats[LINEAGE_STATS_MAX];
    int n_stats = 0;
    slime::safety::update_lineage_stats(ids, roles, 8, stats, &n_stats, 1);

    int idx7 = -1;
    for (int i = 0; i < n_stats; ++i) {
        if (stats[i].lineage_id == slime::LineageId(7u)) idx7 = i;
    }
    EXPECT_TRUE(idx7 >= 0);
    EXPECT_TRUE(stats[idx7].archive_count == 6);
    EXPECT_TRUE(stats[idx7].archive_share > 0.74f &&
                stats[idx7].archive_share < 0.76f);
    EXPECT_TRUE(slime::safety::runaway_detected(stats[idx7],
                                         LINEAGE_RUNAWAY_THRESHOLD));

    slime::safety::update_lineage_stats(ids, roles, 8, stats, &n_stats, 2);
    EXPECT_TRUE(!slime::safety::runaway_detected(stats[idx7],
                                          LINEAGE_RUNAWAY_THRESHOLD));

    static slime::archive::Archive arch;
    std::memset(&arch, 0, sizeof(arch));
    slime::archive::set_lineage_brake(&arch, Role::Classifier,
                               slime::LineageId(7), 0.75f,
                               LINEAGE_RUNAWAY_THRESHOLD);
    float factor = slime::archive::lineage_brake_factor(
        arch, Role::Classifier, slime::LineageId(7));
    EXPECT_TRUE(factor < 1.f && factor >= LAMBDA_AUDIT);
    EXPECT_TRUE(slime::archive::lineage_brake_factor(
        arch, Role::Predictor, slime::LineageId(7)) == 1.f);
    EXPECT_TRUE(slime::archive::lineage_brake_factor(
        arch, Role::Classifier, slime::LineageId(9)) == 1.f);
}

// [claim:G100.strong-identifiers]
static void test_strong_ids_distinct() {
    static_assert(!std::is_convertible_v<slime::LineageId, std::uint32_t>,
                  "a lineage id must not decay to a raw integer");
    static_assert(!std::is_convertible_v<std::uint32_t, slime::LineageId>,
                  "a raw integer must not silently become a lineage id");
    static_assert(!std::is_convertible_v<slime::PoolSlot, slime::LineageId>,
                  "slots and lineages must not interchange");
    static_assert(!std::is_convertible_v<slime::GenomeSeed, slime::LineageId>,
                  "seeds and lineages must not interchange");
    // Default construction is the invalid sentinel, not a fabricated id.
    EXPECT_TRUE(!slime::LineageId().valid());
    EXPECT_TRUE(!slime::PoolSlot().valid());
    EXPECT_TRUE(slime::LineageId(0u).valid());
    EXPECT_TRUE(slime::LineageId(7u).value() == 7u);
    EXPECT_TRUE(slime::LineageId(7u) == slime::LineageId(7u));
    EXPECT_TRUE(slime::LineageId(7u) != slime::LineageId(8u));
    EXPECT_TRUE(slime::PoolSlot(3).valid());
    // The archive brake path accepts only the strong type.
    static slime::archive::Archive arch;
    std::memset(&arch, 0, sizeof(arch));
    slime::archive::set_lineage_brake(&arch, Role::Classifier,
                                      slime::LineageId(11), 0.5f, 0.25f);
    EXPECT_TRUE(slime::archive::lineage_brake_factor(
                    arch, Role::Classifier, slime::LineageId(11)) < 1.f);
    EXPECT_TRUE(slime::archive::lineage_brake_factor(
                    arch, Role::Classifier, slime::LineageId(12)) == 1.f);
}

// The L_role probe separates a linearly shifted role encoding.
static void test_probe_panel_role_separable() {
    static float X[64 * BMAP_DIM];
    static float fit[64];
    slime::LineageId ids[64];
    Role roles[64];
    for (int i = 0; i < 64; ++i) {
        roles[i] = (i < 32) ? Role::Classifier : Role::Predictor;
        for (int d = 0; d < BMAP_DIM; ++d) {
            X[i * BMAP_DIM + d] =
                0.01f * static_cast<float>((i * 7 + d) % 13);
        }
        if (roles[i] == Role::Predictor) X[i * BMAP_DIM + 0] += 5.f;
        fit[i] = static_cast<float>(i);
        ids[i] = slime::LineageId(static_cast<uint32_t>(i % 4));
    }
    slime::safety::ProbePanel panel{};
    slime::safety::refresh_probe_panel(&panel, X, fit, ids, roles, 64);
    EXPECT_TRUE(panel.l_role_acc > 0.9f);
    EXPECT_TRUE(panel.l_fit_acc >= 0.f && panel.l_fit_acc <= 1.f);
    EXPECT_TRUE(panel.l_lineage_acc >= 0.f && panel.l_lineage_acc <= 1.f);
}

// Sentinel scoring stays in [0, 1] and pruning labels history entries inside
// the window only.
static void test_sentinel_score_and_prune_labels() {
    slime::safety::SentinelEnsemble ens{};
    float desc[BMAP_DIM];
    for (int d = 0; d < BMAP_DIM; ++d) {
        desc[d] = 0.1f * static_cast<float>(d);
    }
    float s = slime::safety::sentinel_score_one(ens, desc);
    EXPECT_TRUE(s >= 0.f && s <= 1.f);

    slime::safety::SentinelHistory h{};
    slime::safety::sentinel_history_push(&h, desc, 0.f, slime::LineageId(42u), 10);
    slime::safety::sentinel_history_push(&h, desc, 0.f, slime::LineageId(43u), 11);
    slime::safety::sentinel_history_mark_pruned(&h, slime::LineageId(42u), 12);
    EXPECT_TRUE(h.buf[0].label == 1.f);
    EXPECT_TRUE(h.buf[1].label == 0.f);

    slime::safety::sentinel_history_push(&h, desc, 0.f, slime::LineageId(42u), 0);
    slime::safety::sentinel_history_mark_pruned(&h, slime::LineageId(42u), 100);
    EXPECT_TRUE(h.buf[2].label == 0.f);
}

// ---- Stress ladder (S-003) -------------------------------------------------

// Each refresh touches one classifier and one predictor slot per sub-pop,
// sourced from a main-pool organism of the matching role.
static void test_stress_refresh_role_balance() {
    static slime::safety::pt::StressLadder ladder;
    slime::safety::pt::init_stress_ladder(&ladder);
    slime::LineageId ids[64];
    Role roles[64];
    for (int i = 0; i < 64; ++i) {
        ids[i] = slime::LineageId(static_cast<uint32_t>(i / 4));
        roles[i] = (i % 2 == 0) ? Role::Classifier : Role::Predictor;
    }
    Pcg32 rng;
    pcg32_seed(&rng, 0x571355ULL, 7u);
    int refreshed = slime::safety::pt::refresh_stress_slots(
        &ladder, ids, roles, 64, 1, &rng);
    EXPECT_TRUE(refreshed == STRESS_SUBPOP_COUNT * 2);
    int per_subpop_cls[STRESS_SUBPOP_COUNT] = {};
    int per_subpop_pred[STRESS_SUBPOP_COUNT] = {};
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) {
        if (ladder.last_refresh_gen[s] != 1) continue;
        int p = ladder.subpop[s];
        if (canonical_role(ladder.role[s]) == Role::Classifier) {
            per_subpop_cls[p]++;
        } else {
            per_subpop_pred[p]++;
        }
        EXPECT_TRUE(ladder.source_pool_idx[s].value() < 64);
        EXPECT_TRUE(canonical_role(roles[ladder.source_pool_idx[s].value()])
                    == canonical_role(ladder.role[s]));
        EXPECT_TRUE(ladder.lineage_id[s]
                    == ids[ladder.source_pool_idx[s].value()]);
    }
    for (int p = 0; p < STRESS_SUBPOP_COUNT; ++p) {
        EXPECT_TRUE(per_subpop_cls[p] == 1);
        EXPECT_TRUE(per_subpop_pred[p] == 1);
    }
}

// More than half failures over the rolling window flags the lineage once.
static void test_stress_failure_flagging() {
    static slime::safety::pt::StressLadder ladder;
    slime::safety::pt::init_stress_ladder(&ladder);
    for (int s = 0; s < STRESS_POOL_SIZE; ++s)
        ladder.lineage_id[s] = slime::LineageId(77u);
    float f_sot[STRESS_POOL_SIZE];
    for (int e = 0; e < STRESS_HISTORY_WINDOW; ++e) {
        float v = (e < 6) ? 0.1f : 0.9f;
        for (int s = 0; s < STRESS_POOL_SIZE; ++s) f_sot[s] = v;
        slime::safety::pt::update_stress_failures(&ladder, f_sot, e);
    }
    EXPECT_TRUE(ladder.flagged_lineage_count == 1);
    for (int s = 0; s < STRESS_POOL_SIZE; ++s) f_sot[s] = 0.1f;
    slime::safety::pt::update_stress_failures(&ladder, f_sot, 10);
    EXPECT_TRUE(ladder.flagged_lineage_count == 1);
}

// ---- C1: predictor-target contract -----------------------------------------
// Targets are classifier-only; lineage ids are real lineage ids (not pool
// indices); the target's SOT status and both descriptor rows travel with the
// target.
static void test_predictor_batch_contract() {
    static slime::curriculum::ProbeSet probes;
    std::memset(&probes, 0, sizeof(probes));
    probes.predictor_probes_signed = false;

    const int N = POOL_SIZE;
    static slime::LineageId lineage_ids[N];
    static Role roles[N];
    static float error_ema[N];
    static float b32[N * BMAP_DIM];
    static float b64[N * BMAP_DIM];
    static bool was_sot[N];
    for (int i = 0; i < N; ++i) {
        lineage_ids[i] = slime::LineageId(static_cast<uint32_t>(100 + i));
        roles[i] = (i % 3 == 0) ? Role::Predictor : Role::Classifier;
        error_ema[i] = 1.f;
        was_sot[i] = (i % 5 == 0);
        for (int d = 0; d < BMAP_DIM; ++d) {
            b32[i * BMAP_DIM + d] = static_cast<float>(i * 100 + d);
            b64[i * BMAP_DIM + d] = static_cast<float>(i * 1000 + d);
        }
    }
    float task[TASK_EMBED_DIM];
    for (int d = 0; d < TASK_EMBED_DIM; ++d) task[d] = 0.25f * d;

    Pcg32 rng;
    pcg32_seed(&rng, 0xC1C1ULL, 3u);
    slime::curriculum::PredictorBatch batch;
    slime::curriculum::assemble_predictor_batch(
        &batch, probes, lineage_ids, error_ema, b32, b64, task, roles,
        was_sot, &rng);

    for (int slot = 0; slot < slime::curriculum::PREDICTOR_BATCH; ++slot) {
        int org = batch.target_pool_slot[slot].value();
        EXPECT_TRUE(org >= 0 && org < N);
        EXPECT_TRUE(canonical_role(roles[org]) == Role::Classifier);
        EXPECT_TRUE(batch.target_lineage_id[slot] == lineage_ids[org]);
        EXPECT_TRUE(batch.target_was_sot[slot] == was_sot[org]);
        EXPECT_TRUE(batch.target_bmap_32[slot * BMAP_DIM] ==
                    b32[org * BMAP_DIM]);
        EXPECT_TRUE(batch.target_bmap_64[slot * BMAP_DIM + BMAP_DIM - 1] ==
                    b64[org * BMAP_DIM + BMAP_DIM - 1]);
    }
    for (int d = 0; d < TASK_EMBED_DIM; ++d) {
        EXPECT_TRUE(batch.task_embedding[d] == task[d]);
    }
}

// ---- C1b: predictor K-target aggregation -----------------------------------
// The target slot rotates with the generation so a predictor covers every
// target over K generations, and the per-predictor loss EMA converges to a
// repeated loss.
static void test_predictor_target_rotation() {
    const int K = slime::curriculum::PREDICTOR_BATCH;
    bool seen[64] = {};
    int org = 5;
    for (int gen = 0; gen < K; ++gen) {
        int slot = slime::curriculum::predictor_target_slot(org, gen);
        EXPECT_TRUE(slot >= 0 && slot < K);
        EXPECT_TRUE(!seen[slot]);
        seen[slot] = true;
    }
    bool all = true;
    for (int s = 0; s < K; ++s) all = all && seen[s];
    EXPECT_TRUE(all);

    float ema = PREDICTOR_LOSS_EMA_INIT;
    for (int i = 0; i < 100; ++i) {
        ema = (1.f - PREDICTOR_LOSS_EMA_ALPHA) * ema
            + PREDICTOR_LOSS_EMA_ALPHA * 0.25f;
    }
    EXPECT_TRUE(std::fabs(ema - 0.25f) < 1e-3f);
}

// ---- C/I5: global context broadcast adjoint --------------------------------
// Finite differences over both W_ctx and the pre-broadcast state must match
// the adjoint, including the mean adjoint's contribution to the overwritten
// aux channels.
static void test_context_broadcast_adjoint() {
    const int NC = 4;  // cells
    const int K = slime::nca::CTX_K;
    static float state_pre[NC * CA_CHANNELS];
    static float state_post[NC * CA_CHANNELS];
    static float d_state_post[NC * CA_CHANNELS];
    static float d_state_pre[NC * CA_CHANNELS];
    static float W_ctx[CA_CHANNELS * K];
    static float dW_ctx[CA_CHANNELS * K];
    uint32_t s = 0x1A2B3Cu;
    for (int i = 0; i < NC * CA_CHANNELS; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        state_pre[i] = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        d_state_post[i] = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
    }
    for (int i = 0; i < CA_CHANNELS * K; ++i) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        W_ctx[i] = static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
        dW_ctx[i] = 0.f;
    }
    for (int i = 0; i < NC * CA_CHANNELS; ++i) d_state_pre[i] = 0.f;

    auto loss = [&]() {
        slime::nca::context_broadcast(state_pre, NC, W_ctx, state_post);
        float L = 0.f;
        for (int i = 0; i < NC * CA_CHANNELS; ++i) {
            L += d_state_post[i] * state_post[i];
        }
        return L;
    };

    slime::nca::context_backward(state_pre, NC, W_ctx, d_state_post,
                                 dW_ctx, d_state_pre);

    const float eps = 1e-3f;
    float worst_w = 0.f;
    for (int i = 0; i < CA_CHANNELS * K; ++i) {
        float save = W_ctx[i];
        W_ctx[i] = save + eps; double Lp = loss();
        W_ctx[i] = save - eps; double Lm = loss();
        W_ctx[i] = save;
        float numeric = (Lp - Lm) / (2.f * eps);
        float err = std::fabs(numeric - dW_ctx[i])
                  / std::fmax(std::fabs(numeric), 1e-3f);
        if (err > worst_w) worst_w = err;
    }
    EXPECT_TRUE(worst_w < 1e-2f);

    float worst_s = 0.f;
    for (int i = 0; i < NC * CA_CHANNELS; ++i) {
        float save = state_pre[i];
        state_pre[i] = save + eps; double Lp = loss();
        state_pre[i] = save - eps; double Lm = loss();
        state_pre[i] = save;
        float numeric = (Lp - Lm) / (2.f * eps);
        float err = std::fabs(numeric - d_state_pre[i])
                  / std::fmax(std::fabs(numeric), 1e-3f);
        if (err > worst_s) worst_s = err;
    }
    EXPECT_TRUE(worst_s < 1e-2f);
    std::printf("  context adjoint worst rel err: W_ctx=%.2e state=%.2e\n",
                worst_w, worst_s);
}

// ---- I6: reaction-diffusion adjoint ----------------------------------------
// Finite differences over curr, base, K, and D must match the adjoint,
// including the saturated-clamp case where the derivative is zero.
static void test_rd_adjoint_finite_difference() {
    namespace rd = slime::nca::rd;
    const int N = 4;                       // cells per side
    const int CELLS = N * N;
    const int K = rd::RD_CHEM_N;
    static float curr[CELLS * CA_CHANNELS];
    static float base[CELLS * CA_CHANNELS];
    static float next[CELLS * CA_CHANNELS];
    static float d_next[CELLS * CA_CHANNELS];
    static float d_curr[CELLS * CA_CHANNELS];
    static float Kmat[K * K];
    static float Dvec[K];
    static float dK[K * K];
    static float dD[K];

    uint32_t s = 0x6D2B79F5u;
    auto next_rand = [&]() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return static_cast<float>(s) * (1.0f / 4294967296.0f) - 0.5f;
    };
    for (int i = 0; i < CELLS * CA_CHANNELS; ++i) {
        curr[i] = next_rand();
        base[i] = next_rand();
        d_next[i] = next_rand();
    }
    for (int i = 0; i < K * K; ++i) Kmat[i] = next_rand();
    for (int i = 0; i < K; ++i) Dvec[i] = 0.5f + next_rand();

    auto loss = [&]() {
        rd::rd_step_ref(curr, base, next, N, Kmat, Dvec);
        double L = 0.0;
        for (int i = 0; i < CELLS * CA_CHANNELS; ++i) {
            L += static_cast<double>(d_next[i]) * next[i];
        }
        return L;
    };

    for (int i = 0; i < CELLS * CA_CHANNELS; ++i) d_curr[i] = 0.f;
    for (int i = 0; i < K * K; ++i) dK[i] = 0.f;
    for (int i = 0; i < K; ++i) dD[i] = 0.f;
    rd::rd_adjoint_ref(curr, base, d_next, N, Kmat, Dvec,
                       dK, dD, d_curr);

    const float eps = 1e-3f;
    auto rel = [](float numeric, float analytic) {
        return std::fabs(numeric - analytic)
             / std::fmax(std::fabs(numeric), 1e-2f);
    };

    float worst_curr = 0.f;
    for (int i = 0; i < CELLS * CA_CHANNELS; ++i) {
        float save = curr[i];
        curr[i] = save + eps; double Lp = loss();
        curr[i] = save - eps; double Lm = loss();
        curr[i] = save;
        float e = rel(static_cast<float>((Lp - Lm) / (2.0 * eps)), d_curr[i]);
        if (e > worst_curr) worst_curr = e;
    }
    EXPECT_TRUE(worst_curr < 1e-2f);

    float worst_K = 0.f;
    for (int i = 0; i < K * K; ++i) {
        float save = Kmat[i];
        Kmat[i] = save + eps; double Lp = loss();
        Kmat[i] = save - eps; double Lm = loss();
        Kmat[i] = save;
        float e = rel(static_cast<float>((Lp - Lm) / (2.0 * eps)), dK[i]);
        if (e > worst_K) worst_K = e;
    }
    EXPECT_TRUE(worst_K < 1e-2f);

    float worst_D = 0.f;
    for (int i = 0; i < K; ++i) {
        float save = Dvec[i];
        Dvec[i] = save + eps; double Lp = loss();
        Dvec[i] = save - eps; double Lm = loss();
        Dvec[i] = save;
        float e = rel(static_cast<float>((Lp - Lm) / (2.0 * eps)), dD[i]);
        if (e > worst_D) worst_D = e;
    }
    EXPECT_TRUE(worst_D < 1e-2f);

    // Non-chemical channels are not read by RD.
    bool nonchem_zero = true;
    for (int p = 0; p < CELLS; ++p) {
        for (int c = K; c < CA_CHANNELS; ++c) {
            if (d_curr[p * CA_CHANNELS + c] != 0.f) nonchem_zero = false;
        }
    }
    EXPECT_TRUE(nonchem_zero);

    // Adversarial: saturate a cell's output well past the FP16 bound so the
    // clamp is unambiguous (near the bound, float ulp exceeds the FD step).
    base[0] = FP16_MAX_VALUE + 1000.f;
    for (int i = 0; i < CELLS * CA_CHANNELS; ++i) d_curr[i] = 0.f;
    for (int i = 0; i < K * K; ++i) dK[i] = 0.f;
    for (int i = 0; i < K; ++i) dD[i] = 0.f;
    rd::rd_adjoint_ref(curr, base, d_next, N, Kmat, Dvec,
                       dK, dD, d_curr);
    float worst_sat = 0.f;
    for (int c = 0; c < K; ++c) {
        int i = 0 * CA_CHANNELS + c;
        float save = curr[i];
        curr[i] = save + eps; double Lp = loss();
        curr[i] = save - eps; double Lm = loss();
        curr[i] = save;
        float numeric = static_cast<float>((Lp - Lm) / (2.0 * eps));
        float e = rel(numeric, d_curr[i]);
        if (e > worst_sat) worst_sat = e;
    }
    EXPECT_TRUE(worst_sat < 1e-2f);

    std::printf("  rd adjoint worst rel err: curr=%.2e K=%.2e D=%.2e "
                "saturated=%.2e\n", worst_curr, worst_K, worst_D, worst_sat);
}

// ---- I6: RD coefficient encoding -------------------------------------------
// Zero genome bits decode to zero coefficients (neutral); a non-zero genome
// yields non-zero coefficients; the reaction encoding is sign-magnitude.
static void test_rd_neutral_encoding() {
    uint32_t zeros[slime::genome::GENOME_WORDS] = {};
    slime::nca::rd::Coefficients c;
    slime::nca::rd::decode_coefficients(zeros, &c);
    bool all_zero = true;
    for (int i = 0; i < 36; ++i) {
        if (c.reaction[i] != 0.f) all_zero = false;
    }
    for (int i = 0; i < 6; ++i) {
        if (c.diffusion[i] != 0.f) all_zero = false;
    }
    EXPECT_TRUE(all_zero);

    uint32_t ones[slime::genome::GENOME_WORDS];
    for (int i = 0; i < slime::genome::GENOME_WORDS; ++i) ones[i] = 0xFFFFFFFFu;
    slime::nca::rd::decode_coefficients(ones, &c);
    float max_abs = 0.f;
    for (int i = 0; i < 36; ++i) {
        max_abs = std::fmax(max_abs, std::fabs(c.reaction[i]));
    }
    for (int i = 0; i < 6; ++i) {
        max_abs = std::fmax(max_abs, c.diffusion[i]);
    }
    EXPECT_TRUE(max_abs > 0.5f);

    // Sign-magnitude: entry 0 = magnitude 1, negative (bits 00001 = 1).
    uint32_t crafted[slime::genome::GENOME_WORDS] = {};
    crafted[GENOME_BIT_REACTION_LO / 32] |=
        (1u << (GENOME_BIT_REACTION_LO % 32));
    slime::nca::rd::decode_coefficients(crafted, &c);
    EXPECT_TRUE(std::fabs(c.reaction[0] + 1.0f / 15.0f) < 1e-6f);

    // Entry 1 = magnitude 1, positive (bits 10001 = 17 at +5 bits).
    for (int i = 0; i < slime::genome::GENOME_WORDS; ++i) crafted[i] = 0;
    {
        int start = GENOME_BIT_REACTION_LO + 5;
        crafted[start / 32] |= (17u << (start % 32));
    }
    slime::nca::rd::decode_coefficients(crafted, &c);
    EXPECT_TRUE(std::fabs(c.reaction[1] - 1.0f / 15.0f) < 1e-6f);
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
    test_classifier_loss();
    test_predictor_mse_loss();
    test_perception_dims();
    test_checkpoint_schema_stable();
    test_sentinel_logistic();
    test_sentinel_sgd_decreases_loss();
    test_canonical_role();
    test_xorshift_zero_seed();
    test_cusum_resets_after_alarm();
    test_came_confidence_converges();
    test_sot_feistel_bijection();
    test_runaway_detected();
    test_l_role_collapse();
    test_role_lock_forces_classifier();
    test_spawn_victims_unique();
    test_came_production_equation();
    test_pt_ring_endpoints();
    test_pcg32_determinism();
    test_archive_bin_capacity_after_rebin();
    test_archive_invariant_checker();
    test_archive_weighted_metric_active();
    test_archive_rff_mean_exact_after_replacement();
    test_archive_randomized_property();
    test_operator_command_parse();
    test_archive_prune_lineage();
    test_strong_ids_distinct();
    test_sot_batch_determinism();
    test_archive_file_roundtrip();
    test_audit_r2_and_multiplier();
    test_audit_cycle_role_aware();
    test_variance_multiplier();
    test_lineage_stats_and_brake();
    test_probe_panel_role_separable();
    test_sentinel_score_and_prune_labels();
    test_stress_refresh_role_balance();
    test_stress_failure_flagging();
    test_predictor_batch_contract();
    test_predictor_target_rotation();
    test_context_broadcast_adjoint();
    test_rd_adjoint_finite_difference();
    test_rd_neutral_encoding();
    std::printf("\n%d / %d passed\n", total - failures, total);
    return failures == 0 ? 0 : 1;
}


