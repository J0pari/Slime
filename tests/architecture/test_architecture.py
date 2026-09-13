"""Architectural negative tests: plant the forbidden condition and prove the
guard catches it. A guard that has never been tested against a planted
violation is itself only an assumption.

Run from the repository root:

    python -m unittest discover -s tests/architecture -v

Witness markers below declare the claims these tests witness; the architecture
compiler verifies the bidirectional linkage.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from architecture import compiler, evidence, source_gates  # noqa: E402
from architecture.claims import Claim, WitnessRef, load_registry  # noqa: E402


def files_from(tree: dict[str, str]) -> dict[str, list[str]]:
    return {k: v.splitlines() for k, v in tree.items()}


class GateTests(unittest.TestCase):
    def test_gate_no_ambient_rng_catches_plant(self):
        # [claim:G100.single-prng]
        report = source_gates.GateReport()
        source_gates.gate_no_ambient_rng(
            files_from({"src/foo.cu": "float x = 1.f; std::mt19937 rng(7);"}), report)
        self.assertFalse(report.ok, "planted mt19937 was not caught")
        # A clean tree passes.
        report2 = source_gates.GateReport()
        source_gates.gate_no_ambient_rng(
            files_from({"src/foo.cu": "float x = pcg32_float(&rng);"}), report2)
        self.assertTrue(report2.ok)

    def test_gate_no_managed_memory_catches_plant(self):
        report = source_gates.GateReport()
        source_gates.gate_no_managed_memory(
            files_from({"src/foo.cu": "cudaMallocManaged(&p, n);"}), report)
        self.assertFalse(report.ok, "planted cudaMallocManaged was not caught")

    def test_gate_checked_cuda_calls_catches_plant(self):
        # [claim:S001.cuda-errors-fatal]
        report = source_gates.GateReport()
        source_gates.gate_checked_cuda_calls(
            files_from({"src/foo.cu": "cudaMalloc(&p, n);"}), report)
        self.assertFalse(report.ok, "planted unchecked cudaMalloc was not caught")
        # A checked call passes.
        report2 = source_gates.GateReport()
        source_gates.gate_checked_cuda_calls(
            files_from({"src/foo.cu": "CUDA_ABORT(cudaMalloc(&p, n), \"alloc\");"}),
            report2)
        self.assertTrue(report2.ok)

    def test_gate_named_tunables_catches_plant(self):
        report = source_gates.GateReport()
        source_gates.gate_named_tunables(
            files_from({"src/foo.cu": "if (generation % 50 == 0) swap();"}), report)
        self.assertFalse(report.ok, "planted bare % 50 tunable was not caught")
        report2 = source_gates.GateReport()
        source_gates.gate_named_tunables(
            files_from({"src/foo.cu": "if (generation % PT_SWAP_INTERVAL == 0) swap();"}),
            report2)
        self.assertTrue(report2.ok)

    def test_gate_rd_adjoint_present(self):
        # [claim:A202.rd-adjoint-present]
        report = source_gates.GateReport()
        source_gates.gate_rd_adjoint_present(
            files_from({"integration/host_main.cu":
                        "launch_forward_with_checkpoints(\n"
                        "    w->d_organisms, w->d_fwd_inputs, w->d_rd_coeffs,\n"
                        "    ...);",
                        "autodiff/warp_tape.cu": "no adjoint here"}), report)
        self.assertFalse(report.ok, "RD without its adjoint was not caught")
        report2 = source_gates.GateReport()
        source_gates.gate_rd_adjoint_present(
            files_from({"integration/host_main.cu":
                        "launch_forward_with_checkpoints(\n"
                        "    w->d_organisms, w->d_fwd_inputs, w->d_rd_coeffs,\n"
                        "    ...);",
                        "autodiff/warp_tape.cu":
                        "rd_step(rc, rn, coeffs[org]); "
                        "bwd_rd_gather_kernel; d_rd_g"}), report2)
        self.assertTrue(report2.ok)

    def test_gate_host_authority_catches_plant(self):
        # [claim:S002.host-authority]
        report = source_gates.GateReport()
        source_gates.gate_host_authority(
            files_from({"integration/host_main.cu": "void run() { step_generation(w); }"}),
            report)
        self.assertFalse(report.ok, "missing off-switch poll was not caught")
        report2 = source_gates.GateReport()
        source_gates.gate_host_authority(
            files_from({"integration/host_main.cu":
                        "void run() { if (poll_off_switch()) break; step_generation(w); }"}),
            report2)
        self.assertTrue(report2.ok)

    def test_gate_operator_polling_catches_plant(self):
        # [claim:S002.operator-command-effective]
        report = source_gates.GateReport()
        source_gates.gate_operator_polling(
            files_from({"integration/host_main.cu":
                        "void run() { step_generation(w); }"}), report)
        self.assertFalse(report.ok, "missing operator polling was not caught")
        self.assertTrue(any("paused" in f.text for f in report.findings))
        report2 = source_gates.GateReport()
        source_gates.gate_operator_polling(
            files_from({"integration/host_main.cu":
                        "void run() { poll_operator_commands(w);"
                        " while (w->operator_state.paused) {} }"}), report2)
        self.assertTrue(report2.ok)

    def test_gate_replay_before_spawn_catches_plant(self):
        # [claim:I001.replay-evaluation-identity]
        report = source_gates.GateReport()
        source_gates.gate_replay_before_spawn(
            files_from({"integration/host_main.cu":
                        "spawn_wave(w);\nreplay_buffer_push(&b, d);"}), report)
        self.assertFalse(report.ok, "post-spawn replay push was not caught")
        report2 = source_gates.GateReport()
        source_gates.gate_replay_before_spawn(
            files_from({"integration/host_main.cu":
                        "replay_buffer_push(&b, d);\nspawn_wave(w);"}), report2)
        self.assertTrue(report2.ok)

    def test_gate_surprise_before_spawn_catches_plant(self):
        report = source_gates.GateReport()
        source_gates.gate_surprise_before_spawn(
            files_from({"integration/host_main.cu":
                        "spawn_wave(w);\n"
                        "float s = evaluate_probe_reference(w);"}), report)
        self.assertFalse(report.ok, "post-spawn surprise was not caught")
        report2 = source_gates.GateReport()
        source_gates.gate_surprise_before_spawn(
            files_from({"integration/host_main.cu":
                        "float s = evaluate_probe_reference(w);\n"
                        "spawn_wave(w);"}), report2)
        self.assertTrue(report2.ok)

    def test_gate_schedule_host_only_catches_plant(self):
        # [claim:A101.sot-schedule-independent]
        report = source_gates.GateReport()
        source_gates.gate_schedule_host_only(
            files_from({"curriculum/problem_generator.cu":
                        "cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);"}),
            report)
        self.assertFalse(report.ok, "device access in the schedule was not caught")
        report2 = source_gates.GateReport()
        source_gates.gate_schedule_host_only(
            files_from({"curriculum/problem_generator.cu":
                        "float x = pcg32_float(&rng);"}), report2)
        self.assertTrue(report2.ok)

    def test_gate_numeric_policy_catches_plant(self):
        # [claim:G100.named-tunables]
        # One planted violation per rule: N1..N5.
        planted = files_from({"src/foo.cu": (
            "constexpr float TAU = 0.123f;\n"
            "if (x < 1e-12f) return;\n"
            "float y = ok ? 1.0f : 0.25f;\n"
            "void f(float a = 0.3f);\n"
            "int z = n % 50;\n")})
        report = source_gates.GateReport()
        source_gates.gate_numeric_policy(planted, report)
        self.assertGreaterEqual(len(report.errors), 5,
                                f"expected 5 planted violations, got "
                                f"{len(report.errors)}")

        clean = files_from({"src/foo.cu": (
            "if (x < EPS) return;\n"
            "float y = ok ? ONE : HALF;\n"
            "void f(float a = DEFAULT_RATE);\n"
            "int z = n % PT_SWAP_INTERVAL;\n")})
        report2 = source_gates.GateReport()
        source_gates.gate_numeric_policy(clean, report2)
        self.assertTrue(report2.ok, [str(f) for f in report2.errors])

        # The schema home is where numbers live: it is never scanned.
        schema = files_from({"config/constants.cuh":
                             "constexpr float TAU = 0.123f;\n"})
        report3 = source_gates.GateReport()
        source_gates.gate_numeric_policy(schema, report3)
        self.assertTrue(report3.ok)

    def test_numeric_exemptions_live(self):
        # [claim:G100.named-tunables]
        # Every exemption must still name real code (LLM-Trader's liveness
        # rule): a stale exemption is a lie about the codebase.
        import re as _re
        for (path, symbol), _reason in \
                source_gates.NUMERIC_POLICY_EXEMPTIONS.items():
            p = ROOT / path
            self.assertTrue(p.is_file(), f"exemption names missing file {path}")
            text = p.read_text(encoding="utf-8", errors="replace")
            self.assertRegex(text, _re.escape(symbol) + r"\s*\(",
                             f"exemption names {symbol!r} not defined in {path}")


class CompilerTests(unittest.TestCase):
    def test_phase_model_rejects_seed_before_pt(self):
        # [claim:I001.phase-order]
        invariants = [{"after": ["pt", "seed_transfer"]},
                      {"before": ["seed_transfer", "backward"]}]
        # The legal order passes.
        legal = compiler.validate_phase_model(
            ["pt", "seed_transfer", "backward"], invariants)
        self.assertFalse(legal, f"legal order was rejected: {legal}")
        # SeedTransfer -> PT -> Backward is illegal: the seed-gradient rows
        # would no longer match the checkpoints PT just moved.
        illegal = compiler.validate_phase_model(
            ["seed_transfer", "pt", "backward"], invariants)
        self.assertTrue(illegal, "seed-before-PT order was accepted")
        self.assertTrue(any("pt" in e and "after" in e for e in illegal))

    def test_transaction_completeness_enforced(self):
        transactions = {
            "generation_phases": {"order": ["a", "b"], "invariants": []},
            "organism_buffers": ["organism_state", "seed_grad", "brand_new_buffer"],
            "transactions": {"pt_swap": {"organism_identity":
                                         ["organism_state", "seed_grad"]}},
        }
        errors: list[str] = []
        compiler.check_phase_and_transactions(ROOT, transactions, errors)
        self.assertTrue(any("brand_new_buffer" in e for e in errors),
                        "undeclared organism buffer not caught")

    def test_build_status_checked_against_plan_and_code(self):
        _documents, transactions, _machine = compiler.load_configs(ROOT)
        build = compiler.load_build_status()
        errors: list[str] = []
        compiler.check_build_status(ROOT, build, transactions, errors)
        self.assertFalse(errors, f"real build inventory invalid: {errors}")

        broken = {"items": {k: dict(v) for k, v in build["items"].items()}}
        broken["items"]["I3"]["mechanisms"] = [
            "predictor/hybrid_surprise.cu::no_such_symbol"]
        errors = []
        compiler.check_build_status(ROOT, broken, transactions, errors)
        self.assertTrue(any("I3" in e for e in errors),
                        "bogus build mechanism not caught")

        del broken["items"]["I9"]
        errors = []
        compiler.check_build_status(ROOT, broken, transactions, errors)
        self.assertTrue(any("I9" in e for e in errors),
                        "inventory item with no build-status entry not caught")

        # Lifecycle-independent: force a known item to `missing`, then give it
        # mechanisms; the validator must reject that regardless of the real
        # build state.
        broken["items"]["I1"]["status"] = "missing"
        broken["items"]["I1"]["mechanisms"] = [
            "integration/host_main.cu::step_generation"]
        errors = []
        compiler.check_build_status(ROOT, broken, transactions, errors)
        self.assertTrue(any("I1" in e and "missing" in e for e in errors),
                        "missing item naming mechanisms not caught")

    def test_gpu_evidence_refused_while_build_incomplete(self):
        incomplete = {"items": {"I1": {"status": "implemented"},
                                "I5": {"status": "missing"}}}
        self.assertTrue(evidence.gpu_evidence_gate(incomplete))
        complete = {"items": {"I1": {"status": "implemented"}}}
        self.assertEqual(evidence.gpu_evidence_gate(complete), "")

    def test_prose_citations_resolve(self):
        _documents, _transactions, _machine = compiler.load_configs(ROOT)
        build = compiler.load_build_status()
        claims, _errors = load_registry(compiler.SPEC_DOCS)
        errors: list[str] = []
        compiler.check_prose_citations(ROOT, claims, build, errors)
        self.assertFalse(errors, f"prose citations unresolved: {errors}")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Makefile").write_text("check:\n\t@true\n", encoding="utf-8")
            (root / "README.md").write_text(
                "Run `make ghost` and `python architecture/ghost.py run`;\n"
                "see architecture/ghost.cu and claim A999.ghost.\n",
                encoding="utf-8")
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "TODO.md").write_text("", encoding="utf-8")
            for doc in ("docs/construction_plan.md", "docs/blueprint.md",
                        "docs/cuda_engineering.md"):
                p = root / doc
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("", encoding="utf-8")
            errors = []
            compiler.check_prose_citations(root, claims, {"items": {}}, errors)
            for needle in ("make ghost", "architecture/ghost.py",
                           "architecture/ghost.cu", "A999.ghost"):
                self.assertTrue(any(needle in e for e in errors),
                                f"citation {needle} not caught: {errors}")

    def test_todo_covers_incomplete_build_items(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Makefile").write_text("", encoding="utf-8")
            (root / "README.md").write_text("", encoding="utf-8")
            (root / "AGENTS.md").write_text("", encoding="utf-8")
            (root / "TODO.md").write_text("I1\n", encoding="utf-8")
            for doc in ("docs/construction_plan.md", "docs/blueprint.md",
                        "docs/cuda_engineering.md"):
                p = root / doc
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("", encoding="utf-8")
            errors = []
            build = {"items": {"I1": {"status": "implemented"},
                               "I2": {"status": "missing"}}}
            compiler.check_prose_citations(root, [], build, errors)
            self.assertTrue(any("I2" in e for e in errors),
                            "unlisted incomplete item not caught")

    def test_claim_grammar_doc_matches_registry(self):
        errors: list[str] = []
        compiler.check_claim_grammar_doc(ROOT, errors)
        self.assertFalse(errors, f"AGENTS grammar drifted: {errors}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "AGENTS.md").write_text(
                "@claim <id> <kind>            invariant|contract\n"
                "T <lifecycle>                 planned|provisional\n"
                "C <confidence>                unobserved|inferred\n",
                encoding="utf-8")
            errors = []
            compiler.check_claim_grammar_doc(root, errors)
            self.assertEqual(len(errors), 3, errors)

    def test_canonical_doc_list_matches_registry(self):
        documents, _transactions, _machine = compiler.load_configs(ROOT)
        errors: list[str] = []
        compiler.check_canonical_doc_list(ROOT, documents, errors)
        self.assertFalse(errors, f"AGENTS canonical list drifted: {errors}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "AGENTS.md").write_text("see docs/blueprint.md\n",
                                            encoding="utf-8")
            docs = {"documents": {
                "a": {"kind": "canonical",
                      "files": ["docs/blueprint.md", "docs/ghost.md"]},
                "b": {"kind": "generated", "files": ["docs/status.md"]},
            }}
            errors = []
            compiler.check_canonical_doc_list(root, docs, errors)
            self.assertTrue(any("docs/ghost.md" in e for e in errors),
                            "unlisted canonical document not caught")
            (root / "AGENTS.md").write_text("see docs/undeclared.md\n",
                                            encoding="utf-8")
            errors = []
            compiler.check_canonical_doc_list(root, docs, errors)
            self.assertTrue(any("docs/undeclared.md" in e for e in errors),
                            "undeclared document mention not caught")

    def test_experiment_registry_checked(self):
        claims, _errors = load_registry(compiler.SPEC_DOCS)
        errors: list[str] = []
        compiler.check_experiments(ROOT, claims, errors)
        self.assertFalse(errors, f"experiment registry invalid: {errors}")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "docs").mkdir()
            (root / "docs/construction_plan.md").write_text(
                "### E1 \u2014 Endogenous prediction pressure\n",
                encoding="utf-8")
            errors = []
            compiler.check_experiments(root, claims, errors)
            self.assertTrue(any("E2" in e for e in errors),
                            "registry entry absent from the plan not caught")

    def test_undeclared_root_document_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "docs").mkdir()
            (root / "AGENTS.md").write_text("authority", encoding="utf-8")
            documents = {"documents": {
                "a": {"kind": "operational", "files": ["AGENTS.md"]}}}
            errors: list[str] = []
            compiler.check_documents(root, documents, errors)
            self.assertFalse(errors, errors)
            (root / "NOTES.md").write_text("second authority",
                                           encoding="utf-8")
            errors = []
            compiler.check_documents(root, documents, errors)
            self.assertTrue(any("NOTES.md" in e for e in errors),
                            "undeclared root document not rejected")

    def test_crosses_annotation_completeness_both_directions(self):
        # A code field annotated [crosses:pt=rogue] that is missing from the
        # transaction registry must be caught — this is exactly how
        # d_eff_weights escaped the first version of the transaction model.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "integration"
            src.mkdir(parents=True)
            (src / "main_loop.cu").write_text(
                "float* d_eff_weights; // [identity:organism] [crosses:pt=rogue]\n",
                encoding="utf-8")
            transactions = {
                "generation_phases": {"order": ["a", "b"], "invariants": []},
                "organism_buffers": ["organism_state"],
                "transactions": {"pt_swap": {"organism_identity": ["organism_state"]}},
            }
            errors: list[str] = []
            compiler.check_phase_and_transactions(root, transactions, errors)
            self.assertTrue(any("crosses:pt=rogue" in e for e in errors),
                            "unregistered [crosses:pt=...] annotation not caught")
            # The reverse direction: a registry entry with no code annotation.
            errors2: list[str] = []
            transactions["transactions"]["pt_swap"]["organism_identity"].append(
                "effective_weights")
            transactions["organism_buffers"].append("effective_weights")
            compiler.check_phase_and_transactions(root, transactions, errors2)
            self.assertTrue(
                any("no [crosses:pt=effective_weights]" in e for e in errors2),
                "registry entry without code annotation not caught")

    def test_evidence_witness_attribution_enforced(self):
        # A claim=witness:result entry whose witness is not registered for the
        # claim must be rejected; an unregistered-but-executed binary is
        # accepted as integration-run attestation.
        claim = Claim(id="X001.foo", kind="invariant", doc="d", line=1)
        claim.witnesses = [WitnessRef("W+", "tests/t.cu::real", "X001.foo")]
        results, errors = evidence.parse_and_validate_results(
            ["X001.foo=tests/t.cu::impostor:pass"], [claim], "", ROOT)
        self.assertTrue(errors, "unregistered witness was accepted")
        results2, errors2 = evidence.parse_and_validate_results(
            ["X001.foo=build/coevo.exe:pass"], [claim], "build/coevo.exe", ROOT)
        self.assertFalse(errors2, "binary attestation was rejected")
        self.assertEqual(results2["X001.foo"]["witness"], "build/coevo.exe")

    def test_claim_hash_staleness(self):
        # Editing the proposition itself (statement/mechanisms/witnesses/deps)
        # must stale evidence even when no mechanism FILE changed.
        claim = Claim(id="X001.foo", kind="invariant", doc="d", line=1)
        claim.mechanisms = ["architecture/transactions.yaml::pt_swap"]
        h1 = compiler.claim_hash(claim)
        claim.statement = "a different proposition"
        h2 = compiler.claim_hash(claim)
        self.assertNotEqual(h1, h2)
        # Unrelated claims do not affect each other's hash.
        other = Claim(id="X002.bar", kind="invariant", doc="d", line=1)
        other.mechanisms = ["architecture/transactions.yaml::pt_swap"]
        h_other = compiler.claim_hash(other)
        self.assertEqual(compiler.claim_hash(claim), h2)
        self.assertNotEqual(h_other, h2)

    def test_manifest_records_witness_and_result(self):
        # Regression: new_manifest must store the ACTUAL witness anchor and
        # pass/fail result, not the dict keys ("witness"/"result").
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tests").mkdir()
            (root / "tests/t.cu").write_text(
                "// [claim:X001.foo]\nvoid real() {}\n", encoding="utf-8")
            (root / "src").mkdir()
            (root / "src/m.cu").write_text("void m_fn() {}\n", encoding="utf-8")
            claim = Claim(id="X001.foo", kind="invariant", doc="d", line=1)
            claim.mechanisms = ["src/m.cu::m_fn"]
            claim.witnesses = [WitnessRef("W+", "tests/t.cu::real", "X001.foo")]
            results, errs = evidence.parse_and_validate_results(
                ["X001.foo=tests/t.cu::real:pass"], [claim], "", root)
            self.assertFalse(errs)
            machine = {"schema": "slime-machine/v1"}
            man = evidence.new_manifest(
                name="t", root=root, binary="none", cuda="x", gpu="y",
                seed="s", results=results, claims=[claim], machine=machine)
            self.assertEqual(man["claims"]["X001.foo"]["result"], "pass")
            self.assertEqual(man["claims"]["X001.foo"]["witness"],
                             "tests/t.cu::real")
            v = evidence.verdict(claim, [man], root)
            self.assertEqual(v.state, "pass/current")

    def test_stale_evidence_detected(self):
        # [claim:C001.acceptance-evidence-current]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mech = root / "src/mech.cu"
            mech.parent.mkdir(parents=True)
            mech.write_text("int mech_fn() { return 1; }", encoding="utf-8")
            claim = Claim(id="X001.foo", kind="invariant", doc="d", line=1)
            claim.mechanisms = ["src/mech.cu::mech_fn"]
            claim.witnesses = [WitnessRef("W+", "tests/x.cu::t", "X001.foo")]
            manifest = {
                "schema": "slime-evidence/v1",
                "generatedAt": "2026-01-01T00:00:00+00:00",
                "claims": {"X001.foo": {"witness": "t", "result": "pass"}},
                "claimHashes": {claim.id: compiler.claim_hash(claim)},
                "mechanismHashes": {claim.id: "0" * 64},
                "witnessHashes": {claim.id: "0" * 64},
            }
            v = evidence.verdict(claim, [manifest], root)
            self.assertEqual(v.state, "pass/stale",
                             "changed mechanism did not mark evidence stale")
            manifest["mechanismHashes"][claim.id] = \
                evidence.mechanism_file_hashes([claim], root)[claim.id]
            manifest["witnessHashes"][claim.id] = \
                evidence.witness_file_hashes([claim], root)[claim.id]
            v2 = evidence.verdict(claim, [manifest], root)
            self.assertEqual(v2.state, "pass/current")
            # Editing the proposition itself stales evidence even when no
            # mechanism or witness file changed.
            claim.statement = "a changed proposition"
            v3 = evidence.verdict(claim, [manifest], root)
            self.assertEqual(v3.state, "pass/stale")

    def test_claim_parser_and_validation(self):
        text = (
            "@claim X001.demo invariant\n"
            "S demo statement spanning\n"
            "  two lines\n"
            "M src/a.cu::fn\n"
            "W+ tests/t.cu::demo\n"
            "T established\n"
            "C observed\n"
            "D X002.parent\n"
        )
        claims = load_registry.__wrapped__ if hasattr(load_registry, "__wrapped__") else None
        from architecture.claims import parse_claim_blocks
        parsed = parse_claim_blocks(text, "doc.md")
        self.assertEqual(len(parsed), 1)
        c = parsed[0]
        self.assertEqual(c.id, "X001.demo")
        self.assertEqual(c.statement, "demo statement spanning two lines")
        self.assertEqual(c.mechanisms, ["src/a.cu::fn"])
        self.assertEqual(c.witnesses[0].anchor, "tests/t.cu::demo")
        self.assertEqual(c.lifecycle, "established")
        self.assertEqual(c.deps, ["X002.parent"])

    def test_compiler_catches_missing_witness_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tests").mkdir()
            wit = root / "tests/t.cu"
            wit.write_text("void demo() {}\n", encoding="utf-8")
            claim = Claim(id="X001.demo", kind="invariant", doc="d", line=1)
            claim.mechanisms = []
            claim.witnesses = [WitnessRef("W+", "tests/t.cu::demo", "X001.demo")]
            ok, msg = compiler.resolve_witness(claim.witnesses[0], root, claim.id)
            self.assertFalse(ok, "witness without [claim:] marker was accepted")
            self.assertIn("does not declare", msg)
            wit.write_text("// [claim:X001.demo]\nvoid demo() {}\n", encoding="utf-8")
            ok2, _ = compiler.resolve_witness(claim.witnesses[0], root, claim.id)
            self.assertTrue(ok2)

    def test_compiler_catches_broken_mechanism(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src/a.cu").write_text("void real_fn() {}\n", encoding="utf-8")
            ok, _ = compiler.resolve_mechanism("src/a.cu::missing_fn", root, {})
            self.assertFalse(ok, "stale mechanism anchor was accepted")
            ok2, _ = compiler.resolve_mechanism("src/a.cu::real_fn", root, {})
            self.assertTrue(ok2)

    def test_compiler_catches_undeclared_document(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            docs = root / "docs"
            docs.mkdir()
            (docs / "known.md").write_text("x", encoding="utf-8")
            (docs / "rogue.md").write_text("x", encoding="utf-8")
            documents = {"documents": {
                "c": {"kind": "canonical", "files": ["docs/known.md"]}}}
            errors: list[str] = []
            compiler.check_documents(root, documents, errors)
            self.assertTrue(any("rogue.md" in e for e in errors),
                            "undeclared document was not caught")
            # A generated document without the golden block is also caught.
            (docs / "generated.md").write_text("hand-authored", encoding="utf-8")
            documents["documents"]["g"] = {
                "kind": "generated", "files": ["docs/generated.md"]}
            errors2: list[str] = []
            compiler.check_documents(root, documents, errors2)
            self.assertTrue(any("generated.md" in e for e in errors2),
                            "generated doc without markers was not caught")

    def test_capability_drift_catches_present_tense(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "README.md").write_text(
                "predictors are scored against targets.\n", encoding="utf-8")
            documents = {"capability-drift": {
                "predictor-role": {
                    "status": "planned",
                    "surface-docs": ["README.md"],
                    "banned-present-tense": ["predictors are scored"],
                }}}
            errors: list[str] = []
            compiler.check_capability_drift(root, documents, errors)
            self.assertTrue(errors, "present-tense planned capability was not caught")


if __name__ == "__main__":
    unittest.main(verbosity=2)
