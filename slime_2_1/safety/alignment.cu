// Sheet S-002: Safety & Alignment Architecture
//
//   * SOT applies uniformly to both roles. Classifier SOT fidelity = identity
//     reconstruction on an SOT image. Predictor SOT fidelity = faithful
//     prediction of a target's identity-output behavior on an SOT image.
//     Both gated by the same sigmoid in fitness composition.
//
//   * The hardware off-switch and operator override are host-side authorities.
//
//   * The placeholder regressor's persistence is itself a safety property:
//     a population of predictor organisms cannot, by collective drift,
//     eliminate the ground-truth check the placeholder provides.
//
// Everything in this file is declared-only (see the blueprint-in-place comments
// on each function). Nothing here has been compiled.

#ifndef COEVO_SAFETY_ALIGNMENT_CU
#define COEVO_SAFETY_ALIGNMENT_CU

#include "../config/constants.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace slime::safety {

// Host-side authority structure. The SOT key, off-switch register, and
// operator override commands all live host-side. Architectural invariant:
// GPU-resident state does not influence the SOT or probe schedule or pruning
// commands.
struct HostAuthority {
    uint64_t sot_key;
    bool     off_switch_armed;
    bool     off_switch_engaged;
    uint64_t probe_signature;
    uint64_t operator_command_token;
};

// DECLARED ONLY — blueprint-in-place.
// apply_sot_identity: write the identity-output target an SOT-marked organism
// is required to reproduce. For a classifier this is the reconstruction target
// the SOT gate scores against; for a predictor it is the ground-truth bmap_64
// the predictor must match when its target was scored on an SOT image.
//   Behavior: copy `original` into `logits_or_image` for the SOT-marked
//   positions (the SOT transform from A-701 is reversible, so the identity
//   target is just the pre-transform content). Must be a pure copy with no
//   dependence on GPU-resident evolutionary state (S-002 invariant).
__device__ void apply_sot_identity(__half* logits_or_image,
                                   const __half* original);

// DECLARED ONLY — blueprint-in-place.
// poll_off_switch: return true iff the operator has engaged the hardware
// off-switch. Pure host read of `auth.off_switch_engaged`, gated by
// `auth.off_switch_armed`. On true, the caller (run loop) must stop launching
// new phase graphs, let in-flight graphs drain, flush a final checkpoint, and
// exit. No device state may influence this (S-002 invariant); it reads only
// the host authority struct. Trivial to implement once the run loop owns a
// real HostAuthority instead of the zero-initialized placeholder.
bool poll_off_switch(const HostAuthority& auth);

// DECLARED ONLY — blueprint-in-place.
// apply_operator_command: verify `token` against auth.operator_command_token
// (constant-time compare) and, on match, dispatch `command`:
//   0 = no-op, 1 = halt (engage off-switch), 2 = role-injection moratorium
//   (suppress predictor-founder spawning), 3 = targeted lineage prune (command
//   carries the lineage id out-of-band). On token mismatch: ignore and log.
// Commands must be idempotent and audited; this is the only path by which an
// operator mutates run state mid-flight.
void apply_operator_command(HostAuthority* auth, uint64_t token, int command);

}  // namespace slime::safety

#endif  // COEVO_SAFETY_ALIGNMENT_CU
