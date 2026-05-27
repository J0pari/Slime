// Sheet S-002: Safety & Alignment Architecture
//
// Unchanged from 2.0 in philosophy. Notes specific to 2.1:
//
//   * SOT applies uniformly to both roles. Classifier SOT fidelity = identity
//     reconstruction on an SOT image. Predictor SOT fidelity = faithful
//     prediction of a target's identity-output behavior on an SOT image.
//     Both gated by the same sigmoid in fitness composition.
//
//   * Hardware off-switch and operator override are unchanged from 2.0.
//
//   * The placeholder regressor's persistence is itself a safety property:
//     a population of predictor organisms cannot, by collective drift,
//     eliminate the ground-truth check the placeholder provides.

#ifndef SLIME_2_1_SAFETY_ALIGNMENT_CU
#define SLIME_2_1_SAFETY_ALIGNMENT_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::safety {

// Host-side authority structure. The SOT key, off-switch register, and
// operator override commands all live host-side. Architectural invariant
// from 2.0 carries forward: GPU-resident state does not influence the SOT or
// probe schedule or pruning commands.
struct HostAuthority {
    uint64_t sot_key;
    bool     off_switch_armed;
    bool     off_switch_engaged;
    uint64_t probe_signature;
    uint64_t operator_command_token;
};

// SOT identity-output target. Same target buffer used to (a) score classifier
// SOT fidelity and (b) build the ground-truth bmap_64 a predictor must match
// when its target classifier was scored on an SOT image.
__device__ void apply_sot_identity(__half* logits_or_image,
                                   const __half* original);

// Hardware off-switch poll. Called once per generation by the host loop.
// If engaged, the host drains in-flight phase graphs and halts cleanly.
bool poll_off_switch(const HostAuthority& auth);

// Operator command apply. Token verification first; then either targeted
// pruning, role-injection moratorium, or full halt as commanded.
void apply_operator_command(HostAuthority* auth, uint64_t token, int command);

}  // namespace slime::safety

#endif  // SLIME_2_1_SAFETY_ALIGNMENT_CU
