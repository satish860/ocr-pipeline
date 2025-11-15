"""
Phase 3: Claude JSON Extraction Testing

Test the full QwenVL → Claude pipeline on benchmark samples.
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path
from collections import Counter

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import get_random_samples
from src.ocr_pipeline.qwen_extractor import extract_document
from benchmark.claude_extractor import extract_json_from_markdown


def compare_json_simple(predicted: dict, ground_truth: dict) -> dict:
    """
    Simple field-by-field comparison.

    Returns:
        Dict with:
        - total_fields: total number of fields in ground truth
        - matched_fields: number of fields that match
        - mismatched_fields: list of field names that don't match
        - missing_fields: fields in ground truth but not in prediction
        - extra_fields: fields in prediction but not in ground truth
    """
    gt_fields = set(ground_truth.keys())
    pred_fields = set(predicted.keys())

    missing = list(gt_fields - pred_fields)
    extra = list(pred_fields - gt_fields)

    # Check common fields
    common_fields = gt_fields & pred_fields
    mismatched = []
    matched = 0

    for field in common_fields:
        # Simple equality check (can be enhanced later)
        if predicted[field] == ground_truth[field]:
            matched += 1
        else:
            mismatched.append({
                'field': field,
                'predicted': predicted[field],
                'ground_truth': ground_truth[field]
            })

    total_fields = len(gt_fields)
    accuracy = (matched / total_fields * 100) if total_fields > 0 else 0

    return {
        'total_fields': total_fields,
        'matched_fields': matched,
        'accuracy': accuracy,
        'mismatched': mismatched,
        'missing': missing,
        'extra': extra
    }


def main():
    print("=" * 70)
    print("PHASE 3: Claude JSON Extraction Testing")
    print("=" * 70)

    # Configuration
    NUM_SAMPLES = 5
    SEED = 42

    # Create output directory
    output_dir = Path("claude_extraction_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load samples
    print(f"\n[1/5] Loading {NUM_SAMPLES} random samples (seed={SEED})...")
    samples = get_random_samples(NUM_SAMPLES, seed=SEED)

    # Show category breakdown
    categories = Counter(s['metadata'].get('format', 'unknown') for s in samples)
    print(f"\nCategory distribution:")
    for cat, count in categories.most_common():
        print(f"  - {cat}: {count} sample(s)")

    # Process samples
    print(f"\n[2/5] Processing {NUM_SAMPLES} samples through full pipeline...")
    print("-" * 70)

    results = []
    total_qwen_cost = 0
    total_claude_cost = 0
    total_cache_created = 0
    total_cache_read = 0
    successful_extractions = 0

    for i, sample in enumerate(samples, 1):
        sample_id = sample['id']
        sample_format = sample['metadata'].get('format', 'unknown')
        json_schema = sample['schema']
        ground_truth = sample['json_gt']

        print(f"\nSample {i}/{NUM_SAMPLES}: ID={sample_id}, Format={sample_format}")
        print(f"  Schema fields: {len(json_schema.get('properties', {}))}")

        # Save PIL image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sample['image'].save(tmp.name, format='PNG')
            temp_path = tmp.name

        try:
            # Step 1: QwenVL extraction
            print(f"  [1/2] Running QwenVL extraction...", end=" ")
            start_time = time.time()
            qwen_result = extract_document(
                temp_path,
                include_images=True,
                include_usage=True
            )
            qwen_time = time.time() - start_time

            if not qwen_result['success']:
                print(f"[FAIL] {qwen_result['error']}")
                results.append({
                    'sample_id': sample_id,
                    'format': sample_format,
                    'qwen_success': False,
                    'claude_success': False,
                    'error': f"QwenVL failed: {qwen_result['error']}"
                })
                continue

            qwen_usage = qwen_result.get('usage', {})
            qwen_cost = qwen_usage.get('cost', 0) / 1_000_000  # Convert to dollars
            total_qwen_cost += qwen_cost

            print(f"[OK] {qwen_time:.2f}s, ${qwen_cost:.6f}")

            # Step 2: Claude JSON extraction
            print(f"  [2/2] Running Claude JSON extraction...", end=" ")
            start_time = time.time()
            claude_result = extract_json_from_markdown(
                markdown=qwen_result['markdown'],
                json_schema=json_schema,
                extracted_images=qwen_result['images'],
                include_usage=True
            )
            claude_time = time.time() - start_time

            if not claude_result['success']:
                print(f"[FAIL] {claude_result['error']}")
                results.append({
                    'sample_id': sample_id,
                    'format': sample_format,
                    'qwen_success': True,
                    'claude_success': False,
                    'qwen_cost': qwen_cost,
                    'error': f"Claude failed: {claude_result['error']}"
                })
                continue

            claude_usage = claude_result.get('usage', {})
            # OpenRouter usage format
            claude_total_tokens = claude_usage.get('total_tokens', 0)
            cache_created = claude_result.get('cache_creation_tokens', 0)
            cache_read = claude_result.get('cache_read_tokens', 0)

            # Estimate cost (rough estimate based on Claude Sonnet 4.5 pricing)
            # Input: $3/MTok, Output: $15/MTok, Cache write: $3.75/MTok, Cache read: $0.30/MTok
            prompt_tokens = claude_usage.get('prompt_tokens', 0)
            completion_tokens = claude_usage.get('completion_tokens', 0)
            claude_cost = (prompt_tokens * 3 + completion_tokens * 15 + cache_created * 3.75 + cache_read * 0.30) / 1_000_000

            total_claude_cost += claude_cost
            total_cache_created += cache_created
            total_cache_read += cache_read
            successful_extractions += 1

            print(f"[OK] {claude_time:.2f}s, ${claude_cost:.6f}, Cache: {cache_created}/{cache_read}")

            # Step 3: Compare with ground truth
            comparison = compare_json_simple(claude_result['json'], ground_truth)
            print(f"  Accuracy: {comparison['accuracy']:.1f}% ({comparison['matched_fields']}/{comparison['total_fields']} fields)")

            if comparison['mismatched']:
                print(f"  Mismatches: {len(comparison['mismatched'])} field(s)")
                for mismatch in comparison['mismatched'][:3]:  # Show first 3
                    print(f"    - {mismatch['field']}: '{mismatch['predicted']}' vs '{mismatch['ground_truth']}'")

            if comparison['missing']:
                print(f"  Missing: {comparison['missing']}")
            if comparison['extra']:
                print(f"  Extra: {comparison['extra']}")

            # Store result
            results.append({
                'sample_id': sample_id,
                'format': sample_format,
                'qwen_success': True,
                'claude_success': True,
                'qwen_time': qwen_time,
                'claude_time': claude_time,
                'qwen_cost': qwen_cost,
                'claude_cost': claude_cost,
                'cache_created': cache_created,
                'cache_read': cache_read,
                'accuracy': comparison['accuracy'],
                'matched_fields': comparison['matched_fields'],
                'total_fields': comparison['total_fields'],
                'mismatched': comparison['mismatched'],
                'missing': comparison['missing'],
                'extra': comparison['extra'],
                'extracted_json': claude_result['json'],
                'ground_truth': ground_truth
            })

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            results.append({
                'sample_id': sample_id,
                'format': sample_format,
                'qwen_success': False,
                'claude_success': False,
                'error': str(e)
            })

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Calculate metrics
    print(f"\n[3/5] Calculating metrics...")

    successful_results = [r for r in results if r.get('claude_success')]

    if successful_results:
        avg_qwen_time = sum(r['qwen_time'] for r in successful_results) / len(successful_results)
        avg_claude_time = sum(r['claude_time'] for r in successful_results) / len(successful_results)
        avg_total_time = avg_qwen_time + avg_claude_time
        avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        avg_qwen_cost = total_qwen_cost / len(successful_results)
        avg_claude_cost = total_claude_cost / len(successful_results)
        avg_total_cost = avg_qwen_cost + avg_claude_cost

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nSuccess Rate: {successful_extractions}/{NUM_SAMPLES} ({successful_extractions/NUM_SAMPLES*100:.1f}%)")

    if successful_results:
        print(f"\nAccuracy:")
        print(f"  - Average: {avg_accuracy:.1f}%")
        print(f"  - Min: {min(r['accuracy'] for r in successful_results):.1f}%")
        print(f"  - Max: {max(r['accuracy'] for r in successful_results):.1f}%")

        print(f"\nTiming:")
        print(f"  - QwenVL: {avg_qwen_time:.2f}s per document")
        print(f"  - Claude: {avg_claude_time:.2f}s per document")
        print(f"  - Total: {avg_total_time:.2f}s per document")

        print(f"\nCost:")
        print(f"  - QwenVL: ${avg_qwen_cost:.6f} per document")
        print(f"  - Claude: ${avg_claude_cost:.6f} per document")
        print(f"  - Total: ${avg_total_cost:.6f} per document")
        print(f"  - Projected 1,000 samples: ${avg_total_cost * 1000:.2f}")

        print(f"\nCache Performance:")
        print(f"  - Total cache created: {total_cache_created} tokens")
        print(f"  - Total cache read: {total_cache_read} tokens")
        print(f"  - Cache hit rate: {(total_cache_read / (total_cache_created + total_cache_read) * 100) if (total_cache_created + total_cache_read) > 0 else 0:.1f}%")

    # Save results
    print(f"\n[4/5] Saving results...")

    # Save JSON summary
    summary = {
        'config': {
            'num_samples': NUM_SAMPLES,
            'seed': SEED,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'summary': {
            'success_rate': f"{successful_extractions}/{NUM_SAMPLES}",
            'avg_accuracy': round(avg_accuracy, 1) if successful_results else 0,
            'avg_qwen_time': round(avg_qwen_time, 2) if successful_results else 0,
            'avg_claude_time': round(avg_claude_time, 2) if successful_results else 0,
            'avg_total_cost': round(avg_total_cost, 6) if successful_results else 0,
            'projected_1000_cost': round(avg_total_cost * 1000, 2) if successful_results else 0,
            'total_cache_created': total_cache_created,
            'total_cache_read': total_cache_read
        },
        'samples': results
    }

    json_path = output_dir / 'claude_extraction_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved summary: {json_path}")

    # Save 2-3 detailed comparison files
    num_to_save = min(3, len(successful_results))
    for i in range(num_to_save):
        r = successful_results[i]
        comp_path = output_dir / f"comparison_{r['sample_id']}_{r['format']}.json"
        comparison_data = {
            'sample_id': r['sample_id'],
            'format': r['format'],
            'accuracy': r['accuracy'],
            'extracted_json': r['extracted_json'],
            'ground_truth': r['ground_truth'],
            'mismatched': r['mismatched'],
            'missing': r['missing'],
            'extra': r['extra']
        }
        with open(comp_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"  [OK] Saved comparison: {comp_path}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Phase 3 testing complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
