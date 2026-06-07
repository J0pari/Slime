// I-001 host entry. Minimal driver: initialise the world, run generations
// until termination, write a final checkpoint, exit.
//
// At this stage the phase graphs are stubbed; this main is a structural
// placeholder that proves the layout links and exits cleanly. Each phase
// gets filled in over C-001 phases 3-8.

#include "main_loop.cu"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace slime::integration {

void initialize_world(World* world) {
    std::memset(world, 0, sizeof(World));
    world->generation = 0;
    world->s_target_calibrated = false;
    world->bootstrap_fired = false;
    world->mut_ladder.beta = 1.0f;
    world->mut_ladder.accept_ema = PT_TARGET_ACCEPT;
    // Spread initial active-pool organisms across the four mutation-rate
    // replicas (round-robin).
    for (int i = 0; i < POOL_SIZE; ++i) {
        world->mut_ladder.replica_of[i] =
            static_cast<uint8_t>(i % PT_NUM_REPLICAS);
        world->organisms.replica_tag[i] = world->mut_ladder.replica_of[i];
    }
    // Stress slots tagged with high bit set (per integration/main_loop.cu).
    for (int i = 0; i < STRESS_POOL_SIZE; ++i) {
        int slot = POOL_SIZE + i;
        world->organisms.replica_tag[slot] =
            static_cast<uint8_t>(0x80 | (i / STRESS_SUBPOP_SIZE));
        // Role-balanced stress sub-pops: first half classifier, second half predictor.
        bool is_pred = (i % STRESS_SUBPOP_SIZE) >= (STRESS_SUBPOP_SIZE / 2);
        world->stress_ladder.role[i] =
            is_pred ? Role::Predictor : Role::Classifier;
        world->stress_ladder.subpop[i] =
            static_cast<uint8_t>(i / STRESS_SUBPOP_SIZE);
    }
    std::printf("coevo: world initialised. pool=%d stress=%d archive_cap=%d\n",
                POOL_SIZE, STRESS_POOL_SIZE, MAX_ARCHIVE);
}

void step_generation(World* world) {
    // Phase ordering follows I-001 pseudocode. Bodies are stubs while
    // construction sequence C-001 phases 3-8 are in progress.
    int gen = world->generation;

    // Bootstrap trigger: one-shot predictor founder spawn.
    if (!world->bootstrap_fired && archive::bootstrap_trigger(world->archive)) {
        world->bootstrap_fired = true;
        world->bootstrap_gen = gen;
        std::printf("coevo: bootstrap trigger at gen=%d (archive=%d)\n",
                    gen, archive::archive_size(world->archive));
        // predictor::spawn_predictor_founders(...);
    }

    // execution::launch_phase(&world->graphs, execution::Phase::Curriculum);
    // ... remaining phases ...

    // Update rolling correlation window + CUSUM each generation. Before the
    // predictor sub-population exists, s_predictor is undefined; pushing it
    // would seed the 100-generation Pearson window with pre-bootstrap zeros
    // that linger after the transition. Only feed the window once predictors
    // are live (A-601: before bootstrap r is treated as 0).
    float r = 0.f;
    if (world->bootstrap_fired) {
        predictor::push_correlation(&world->r_window,
                                    world->s_placeholder,
                                    world->s_predictor);
        r = predictor::pearson_r_clipped(world->r_window);
    }
    world->s_blended = predictor::blend_surprise(world->s_placeholder,
                                                 world->s_predictor, r);
    safety::cusum_update(&world->cusum_surprise, world->s_blended);
    safety::cusum_update(&world->cusum_r, r);

    // PT swap proposals every PT_SWAP_INTERVAL generations.
    if (gen > 0 && gen % PT_SWAP_INTERVAL == 0) {
        safety::pt::update_beta(&world->mut_ladder);
        // safety::pt::propose_swaps(&world->mut_ladder, ...);
    }

    world->generation = gen + 1;
}

int run(const char* /*checkpoint_path*/) {
    World* world = static_cast<World*>(std::calloc(1, sizeof(World)));
    if (!world) return 1;
    initialize_world(world);
    // Smoke-test loop: 10 stepped iterations. Real runs replace with the
    // off-switch / termination poll from S-002.
    for (int i = 0; i < 10; ++i) step_generation(world);
    std::printf("coevo: completed %d smoke generations\n", world->generation);
    std::free(world);
    return 0;
}

}  // namespace slime::integration

int main(int /*argc*/, char** /*argv*/) {
    return slime::integration::run(nullptr);
}
