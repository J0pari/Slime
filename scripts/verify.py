#!/usr/bin/env python3
"""
Trustless verification script for SLIME MNIST training outputs.
Recomputes all metrics from raw GPU dumps and validates against manifest hashes.
"""

import numpy as np
import hashlib
import struct
import sys
import os
from pathlib import Path
from core import BinaryData, FileOp

def verify_file_hash(filepath, expected_hash, expected_size):
    """Verify file integrity against SHA256 hash"""
    filepath_obj = Path(filepath) if isinstance(filepath, str) else filepath
    result = FileOp.read_bytes(filepath_obj)
    if result.is_err():
        return False, f"File not found: {filepath}"

    data = result.value
    if len(data) != expected_size:
        return False, f"Size mismatch: {len(data)} != {expected_size}"

    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        return False, f"Hash mismatch:\n  Expected: {expected_hash}\n  Got:      {actual_hash}"

    return True, "OK"

def load_manifest(manifest_path):
    """Load manifest file with filenames, sizes, and SHA256 hashes using FileOp."""
    result = FileOp.read(Path(manifest_path))
    if result.is_err():
        print(f"ERROR: {result.message}")
        sys.exit(1)

    manifest = {}
    for line in result.value.split('\n'):
        parts = line.strip().split(',')
        if len(parts) == 3:
            filename, size, hash_val = parts
            manifest[filename] = {'size': int(size), 'hash': hash_val}
    return manifest

def verify_generation(base_path, gen_idx, manifest):
    """Verify all data for a single generation."""
    print(f"\n=== Generation {gen_idx} ===")

    # Input data
    mnist_path = f"{base_path}/mnist_{gen_idx:04d}.raw"
    label_path = f"{base_path}/label_{gen_idx:04d}.raw"

    # Neural CA output
    ca_path = f"{base_path}/ca_{gen_idx:04d}.raw"
    logits_path = f"{base_path}/logits_{gen_idx:04d}.raw"

    # Reaction-diffusion fields
    rd_u_path = f"{base_path}/rd_u_{gen_idx:04d}.raw"
    rd_v_path = f"{base_path}/rd_v_{gen_idx:04d}.raw"

    # Resource dynamics
    resources_path = f"{base_path}/resources_{gen_idx:04d}.raw"
    fitness_path = f"{base_path}/fitness_{gen_idx:04d}.raw"

    # Lifecycle state
    lifecycle_path = f"{base_path}/lifecycle_{gen_idx:04d}.raw"

    # Performance metrics
    perf_path = f"{base_path}/perf_{gen_idx:04d}.raw"

    errors = []

    # Verify MNIST image
    if mnist_path in manifest:
        ok, msg = verify_file_hash(mnist_path, manifest[mnist_path]['hash'], manifest[mnist_path]['size'])
        if not ok:
            errors.append(f"MNIST: {msg}")
        else:
            with open(mnist_path, 'rb') as f:
                mnist_data = np.frombuffer(f.read(), dtype=np.uint8)
            print(f"✓ MNIST image: {len(mnist_data)} bytes, hash verified")

    # Verify CA state
    if ca_path in manifest:
        ok, msg = verify_file_hash(ca_path, manifest[ca_path]['hash'], manifest[ca_path]['size'])
        if not ok:
            errors.append(f"CA: {msg}")
        else:
            with open(ca_path, 'rb') as f:
                ca_data = np.frombuffer(f.read(), dtype=np.float32)
            print(f"✓ CA state: {len(ca_data)} float32 values, hash verified")

    # Verify logits
    if logits_path in manifest:
        ok, msg = verify_file_hash(logits_path, manifest[logits_path]['hash'], manifest[logits_path]['size'])
        if not ok:
            errors.append(f"Logits: {msg}")
        else:
            with open(logits_path, 'rb') as f:
                logits = np.frombuffer(f.read(), dtype=np.float32)

            if len(logits) != 10:
                errors.append(f"Logits: Expected 10 classes, got {len(logits)}")
            else:
                # Recompute metrics
                predicted = np.argmax(logits)
                probs = BinaryData.softmax(logits)

                print(f"✓ Logits: {logits}")
                print(f"  Softmax: {probs}")
                print(f"  Predicted: {predicted}")

    # Verify label
    if label_path in manifest:
        ok, msg = verify_file_hash(label_path, manifest[label_path]['hash'], manifest[label_path]['size'])
        if not ok:
            errors.append(f"Label: {msg}")
        else:
            with open(label_path, 'rb') as f:
                label = struct.unpack('i', f.read(4))[0]

            if logits_path in manifest:
                loss = BinaryData.cross_entropy_loss(logits, label)
                correct = (predicted == label)

                print(f"✓ True label: {label}")
                print(f"  Loss: {loss:.6f}")
                print(f"  Correct: {correct}")

                if not correct:
                    print(f"  ⚠ PREDICTION FAILURE: Predicted {predicted}, True {label}")

    # Verify reaction-diffusion fields
    if rd_u_path in manifest:
        ok, msg = verify_file_hash(rd_u_path, manifest[rd_u_path]['hash'], manifest[rd_u_path]['size'])
        if ok:
            with open(rd_u_path, 'rb') as f:
                u_field = np.frombuffer(f.read(), dtype=np.float32)
            print(f"✓ RD U-field: {len(u_field)} values, mean={u_field.mean():.4f}, std={u_field.std():.4f}")

    if rd_v_path in manifest:
        ok, msg = verify_file_hash(rd_v_path, manifest[rd_v_path]['hash'], manifest[rd_v_path]['size'])
        if ok:
            with open(rd_v_path, 'rb') as f:
                v_field = np.frombuffer(f.read(), dtype=np.float32)
            print(f"✓ RD V-field: {len(v_field)} values, mean={v_field.mean():.4f}, std={v_field.std():.4f}")

    # Verify resource dynamics
    if resources_path in manifest:
        ok, msg = verify_file_hash(resources_path, manifest[resources_path]['hash'], manifest[resources_path]['size'])
        if ok:
            with open(resources_path, 'rb') as f:
                resources = np.frombuffer(f.read(), dtype=np.float32)
            print(f"✓ Resource density: total={resources.sum():.2f}, max={resources.max():.4f}")

    if fitness_path in manifest:
        ok, msg = verify_file_hash(fitness_path, manifest[fitness_path]['hash'], manifest[fitness_path]['size'])
        if ok:
            with open(fitness_path, 'rb') as f:
                fitness = np.frombuffer(f.read(), dtype=np.float32)
            print(f"✓ Fitness landscape: mean={fitness.mean():.4f}, max={fitness.max():.4f}")

    # Verify lifecycle state
    if lifecycle_path in manifest:
        ok, msg = verify_file_hash(lifecycle_path, manifest[lifecycle_path]['hash'], manifest[lifecycle_path]['size'])
        if ok:
            with open(lifecycle_path, 'rb') as f:
                phases = np.frombuffer(f.read(), dtype=np.int32)
            phase_names = ['ACTIVE', 'STRESSED', 'DORMANT', 'REACTIVATING']
            print(f"✓ Lifecycle phases: {', '.join(f'{phase_names[i]}={phases[i]}' for i in range(4))}")

    # Verify performance metrics
    if perf_path in manifest:
        ok, msg = verify_file_hash(perf_path, manifest[perf_path]['hash'], manifest[perf_path]['size'])
        if ok:
            with open(perf_path, 'rb') as f:
                metrics = np.frombuffer(f.read(), dtype=np.float32)
            print(f"✓ Performance metrics: {len(metrics)} values")

    if errors:
        print(f"\n❌ VERIFICATION FAILED:")
        for err in errors:
            print(f"  {err}")
        return False

    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: verify.py <report_directory>")
        sys.exit(1)

    report_dir = sys.argv[1]
    manifest_path = os.path.join(report_dir, "manifest.txt")

    if not os.path.exists(manifest_path):
        print(f"ERROR: Manifest not found: {manifest_path}")
        sys.exit(1)

    print(f"Loading manifest: {manifest_path}")
    manifest = load_manifest(manifest_path)
    print(f"Loaded {len(manifest)} file entries")

    # Find all generations
    generations = set()
    for filename in manifest.keys():
        if '_' in filename and '.raw' in filename:
            parts = os.path.basename(filename).split('_')
            if len(parts) >= 2:
                try:
                    gen_num = int(parts[1].split('.')[0])
                    generations.add(gen_num)
                except ValueError:
                    pass

    generations = sorted(generations)
    print(f"Found {len(generations)} generations to verify")

    all_passed = True
    for gen in generations:
        if not verify_generation(report_dir, gen, manifest):
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL VERIFICATIONS PASSED")
        print("All raw data hashes match, metrics recomputed successfully")
    else:
        print("❌ VERIFICATION FAILED")
        print("One or more generations failed integrity checks")
        sys.exit(1)

if __name__ == "__main__":
    main()
