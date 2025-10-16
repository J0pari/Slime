
import hashlib
import json
from pathlib import Path

def compute_result_hash(result_dict):
    """Compute cryptographic hash of results (tamper-proof)."""
    # Sort keys for deterministic hashing
    canonical = json.dumps(result_dict, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()

def verify_results(results_file):
    """Verify results haven't been tampered with."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    if 'hash' not in data:
        return False, 'No hash found'
    
    stored_hash = data.pop('hash')
    computed_hash = compute_result_hash(data)
    
    if stored_hash == computed_hash:
        return True, 'Results verified'
    else:
        return False, f'Hash mismatch: {stored_hash[:8]} != {computed_hash[:8]}'
