# Architecture Signals

Non-normative observations and friction notes. This document records things
worth remembering about the architecture-control process; it never changes
behavior. The three binding specifications remain blueprint.md,
cuda_engineering.md, and construction_plan.md.

## Signals

- 2026-09-11: The first claim registry embedded 21 claims into the specs.
  Four claims are intentionally red: `A401.bin-capacity`,
  `A401.live-statistics-exact`, `S002.operator-command-effective`, and
  `A101.sot-schedule-independent` have mechanisms but no strong witness yet,
  and their semantics are known to be incomplete (PCA rebin can exceed bin
  caps; operator pause/checkpoint flags are transient locals). The report
  surfaces them until witnesses exist.
- 2026-09-11: The source gate `checked_cuda_calls` ships with an explicit
  allowlist of wrapper functions whose raw CUDA calls are caught by a later
  stream synchronize or are shutdown-only. `--strict` fails on allowlisted
  hits so the list can only shrink.
- 2026-09-11: The repository is not under git version control. Evidence
  manifests therefore rely on content hashes (mechanism/witness/binary)
  rather than commit ids for staleness.
