# CLAUDE.md — Binding constraints for all AI work on this project

## The docs folder is the spec. Period.

- `docs/blueprint.md` is the **binding specification**. Every function, every data flow, every architectural decision described there is what gets built. Not a subset. Not a "simpler version." Not "for now."
- `docs/construction_plan.md` defines the **build order and acceptance criteria**. Each stage's acceptance check must pass before moving to the next. No skipping, no partial credit.
- `docs/IMPLEMENTATION_STATUS.md` must be kept **honest and current**. If something is a stub, it says stub. If something works, it says works. No theatrical "smoke tests" that test nothing meaningful.
- `docs/bom.md` defines hardware/toolchain requirements.

## Zero technical debt. Zero shortcuts.

- **Never implement a "simpler version" of something the spec defines.** If the spec says checkpointed warp tape, you build a checkpointed warp tape. If you can't do it yet because dependencies aren't ready, you wait — you don't build a throwaway substitute.
- **Never create code that will need to be replaced.** Every line you write must be the final version of that line for its current scope. If the spec changes, that's different. But writing placeholder garbage that "works for now" is forbidden.
- **Never leave stubs, empty wrappers, or husks.** If a function exists, it has a complete body that does what the spec says. If it can't be implemented yet (missing dependencies from a later stage), it doesn't exist yet — remove the declaration entirely rather than leaving a fake shell.
- **Never add code that isn't called, tested, and justified by the spec.** Dead code is worse than no code.

## Testing requirements

- Tests must **fail unless the real implementation works**. A test that passes against a stub or empty state is a lie, not a test.
- Every function gets tested against spec behavior, not just "doesn't crash" or "returns finite numbers."
- The construction_plan.md acceptance criteria are the minimum bar, not the target.

## Build process

- `source env.sh` sets up the build environment (MSVC + CUDA + make in PATH). This is the only setup step.
- `make check` / `make forward-smoke` / `make all` are the build commands. The Makefile is the build system. Do not work around it.
- All targets must compile with **zero warnings**.

## How to work

1. **Read the spec first.** Before touching any file, read the relevant blueprint sheet, the construction plan stage, and the current implementation status.
2. **Write the failing test first.** The test encodes the spec's acceptance criteria. It must fail before implementation and pass after.
3. **Implement exactly what the spec says.** Not more, not less, not different.
4. **Verify the test passes.** Then update IMPLEMENTATION_STATUS.md honestly.
5. **Move to the next piece.** Follow construction_plan.md stage order.

## What "done" means

A function is done when:
- It implements every behavior described in its blueprint sheet
- It is called from the integration point the spec describes
- It has tests that would fail if the implementation were removed
- It compiles with zero warnings under nvcc + MSVC
- IMPLEMENTATION_STATUS.md accurately reflects its state

A stage is done when:
- Every function in that stage's scope has a body
- The stage's acceptance criteria from construction_plan.md pass on GPU
- No stubs remain from that stage's scope

## Never do these things

- Never say "for now" or "simpler but correct" or "we can upgrade later"
- Never create temporary/throwaway implementations
- Never write scripts then fail to test them before declaring success
- Never work around broken tooling — fix the tooling
- Never claim something works without running it on the actual GPU
- Never write a test that passes against empty/zero-initialized state
- Never proceed to stage N+1 while stage N has stubs or failures
