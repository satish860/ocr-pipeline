"""Exploration script for OCR benchmark dataset (Phase 1)

This script loads the dataset and displays minimal statistics:
- Total sample count
- Category breakdown
- JSON schema statistics
- Sampling method validation
"""

import sys
from pathlib import Path

# Add parent directory to path to import benchmark module
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import (
    get_dataset_stats,
    get_random_samples,
    get_samples_by_category,
    get_stratified_samples,
    get_all_categories,
)


def main():
    print("=" * 60)
    print("OCR Benchmark Dataset Exploration")
    print("=" * 60)
    print()

    # Get dataset statistics
    print("Loading dataset and computing statistics...")
    stats = get_dataset_stats()

    # Display overview
    print("[Dataset Overview]")
    print(f"   Total samples: {stats['total_samples']:,}")
    print()

    # Display categories
    print("[Categories - by metadata tags]:")
    categories = stats['categories']

    # Sort by count (descending)
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)

    for category, count in sorted_categories:
        percentage = (count / stats['total_samples']) * 100
        print(f"   - {category:20s}: {count:4d} samples ({percentage:5.1f}%)")
    print()

    # Display schema statistics
    print("[JSON Schema Statistics]:")
    print(f"   - Average fields per schema: {stats['avg_schema_fields']}")
    print(f"   - Min fields: {stats['min_schema_fields']}")
    print(f"   - Max fields: {stats['max_schema_fields']}")
    print()

    # Test sampling methods
    print("=" * 60)
    print("Testing Sampling Methods")
    print("=" * 60)
    print()

    # Test 1: Random sampling
    print("[Test 1] Random Sampling (5 samples):")
    random_samples = get_random_samples(5, seed=42)
    print(f"   [OK] Retrieved {len(random_samples)} samples")
    print(f"   Sample IDs: {[s['id'] for s in random_samples]}")
    print()

    # Test 2: Category filtering
    # Pick the most common category (excluding very broad ones)
    test_category = None
    for category, count in sorted_categories:
        if category not in ['annotated', 'synthetic'] and count >= 10:
            test_category = category
            break

    if test_category:
        print(f"[Test 2] Category Filtering (5 samples from '{test_category}'):")
        category_samples = get_samples_by_category(test_category, 5, seed=42)
        print(f"   [OK] Retrieved {len(category_samples)} samples")

        # Verify all have the correct category
        def has_category(sample):
            metadata = sample.get('metadata', {})
            if not isinstance(metadata, dict):
                return False
            format_type = metadata.get('format', '')
            quality_type = metadata.get('documentQuality', '')
            return (test_category.upper() == format_type.upper() or
                   test_category.upper() == quality_type.upper())

        all_correct = all(has_category(s) for s in category_samples)
        print(f"   [OK] All samples have '{test_category}' tag: {all_correct}")
        print(f"   Sample IDs: {[s['id'] for s in category_samples]}")
    else:
        print("[Test 2] Category Filtering: Skipped (no suitable category)")
    print()

    # Test 3: Stratified sampling
    print("[Test 3] Stratified Sampling (20 samples):")
    stratified_samples = get_stratified_samples(20, seed=42)
    print(f"   [OK] Retrieved {len(stratified_samples)} samples")

    # Show distribution
    from collections import Counter
    sample_categories = []
    for s in stratified_samples:
        metadata = s.get('metadata', {})

        # Metadata is now parsed as dict
        if isinstance(metadata, dict):
            primary_category = metadata.get('format', 'uncategorized')
        else:
            primary_category = 'uncategorized'

        sample_categories.append(primary_category)

    distribution = Counter(sample_categories)
    print("   Distribution:")
    for category, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(stratified_samples)) * 100
        print(f"      - {category:20s}: {count:2d} samples ({percentage:5.1f}%)")
    print()

    # Success message
    print("=" * 60)
    print("[SUCCESS] Dataset Exploration Complete!")
    print("=" * 60)
    print()
    print("[Next Steps]:")
    print("   - Phase 2: Run QwenVL baseline on 10 samples")
    print("   - Phase 3: Add Claude JSON extraction")
    print("   - Phase 4: Implement evaluation metrics")
    print("   - Phase 5: Full benchmark analysis")
    print()


if __name__ == "__main__":
    main()
