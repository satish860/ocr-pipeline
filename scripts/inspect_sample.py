"""Quick script to inspect raw dataset structure"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.dataset import load_benchmark_dataset
import json

# Load dataset
dataset = load_benchmark_dataset()

# Get first sample
sample = dataset[0]

# Print all keys
print("=" * 60)
print("Sample 0 - Available Keys:")
print("=" * 60)
print(f"Keys: {list(sample.keys())}")
print()

# Print each field type and preview
for key in sample.keys():
    value = sample[key]
    value_type = type(value).__name__

    print(f"[{key}]")
    print(f"   Type: {value_type}")

    if isinstance(value, str):
        preview = value[:200] if len(value) > 200 else value
        print(f"   Preview: {repr(preview)}")
    elif isinstance(value, dict):
        print(f"   Keys: {list(value.keys())[:10]}")
        print(f"   Preview: {json.dumps(value, indent=2)[:200]}")
    elif isinstance(value, list):
        print(f"   Length: {len(value)}")
        if len(value) > 0:
            print(f"   First item: {repr(value[0])[:100]}")
    else:
        print(f"   Value: {repr(value)[:100]}")

    print()

print("=" * 60)
print("Sample 100 - Checking different sample:")
print("=" * 60)
sample2 = dataset[100]
print(f"Keys: {list(sample2.keys())}")
print()

# Check metadata specifically
print("Metadata field details:")
metadata = sample2.get('metadata', None)
print(f"   Type: {type(metadata).__name__}")
print(f"   Value: {repr(metadata)[:200]}")
