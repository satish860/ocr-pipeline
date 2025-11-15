"""
Phase 4: Evaluation Metrics Testing

Test the strict matching evaluator on 10 benchmark samples using getomni-ai methodology.
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
from benchmark.evaluator import compare_json, calculate_accuracy, format_diff_report


def main():
    print("=" * 70)
    print("PHASE 4: Evaluation Metrics Testing (Strict Matching)")
    print("=" * 70)

    # Configuration
    NUM_SAMPLES = 10
    SEED = 42

    # Create output directory
    output_dir = Path("evaluation_outputs")
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
            print(f"  [1/3] Running QwenVL extraction...", end=" ")
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
            qwen_cost = qwen_usage.get('cost', 0) / 1_000_000
            total_qwen_cost += qwen_cost

            print(f"[OK] {qwen_time:.2f}s, ${qwen_cost:.6f}")

            # Step 2: Claude JSON extraction
            print(f"  [2/3] Running Claude JSON extraction...", end=" ")
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
            prompt_tokens = claude_usage.get('prompt_tokens', 0)
            completion_tokens = claude_usage.get('completion_tokens', 0)
            cache_created = claude_result.get('cache_creation_tokens', 0)
            cache_read = claude_result.get('cache_read_tokens', 0)

            # Estimate Claude cost
            claude_cost = (prompt_tokens * 3 + completion_tokens * 15 +
                          cache_created * 3.75 + cache_read * 0.30) / 1_000_000

            total_claude_cost += claude_cost
            successful_extractions += 1

            print(f"[OK] {claude_time:.2f}s, ${claude_cost:.6f}")

            # Step 3: Strict evaluation using getomni methodology
            print(f"  [3/3] Running strict evaluation...", end=" ")
            evaluation = compare_json(claude_result['json'], ground_truth)

            total_fields = evaluation['total_fields']
            diff_fields = evaluation['different_fields']
            accuracy = evaluation['accuracy'] * 100

            print(f"[OK]")
            print(f"  Total fields: {total_fields}, Different: {diff_fields}, Accuracy: {accuracy:.1f}%")

            if evaluation['differences']:
                print(f"  Top differences ({min(3, len(evaluation['differences']))}):")
                for diff in evaluation['differences'][:3]:
                    pred_str = str(diff['predicted'])[:50]
                    gt_str = str(diff['ground_truth'])[:50]
                    print(f"    - {diff['path']}: '{pred_str}' vs '{gt_str}'")

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
                'total_fields': total_fields,
                'different_fields': diff_fields,
                'accuracy': accuracy,
                'evaluation': evaluation,
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
    print(f"\n[3/5] Calculating aggregate metrics...")

    successful_results = [r for r in results if r.get('claude_success')]

    if successful_results:
        avg_qwen_time = sum(r['qwen_time'] for r in successful_results) / len(successful_results)
        avg_claude_time = sum(r['claude_time'] for r in successful_results) / len(successful_results)
        avg_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
        min_accuracy = min(r['accuracy'] for r in successful_results)
        max_accuracy = max(r['accuracy'] for r in successful_results)

        # Count documents by accuracy tiers
        perfect_docs = sum(1 for r in successful_results if r['accuracy'] == 100.0)
        high_acc_docs = sum(1 for r in successful_results if r['accuracy'] >= 90.0)
        good_acc_docs = sum(1 for r in successful_results if r['accuracy'] >= 70.0)

    # Print summary
    print("\n" + "=" * 70)
    print("PHASE 4 RESULTS SUMMARY (STRICT MATCHING)")
    print("=" * 70)
    print(f"\nPipeline Success Rate: {successful_extractions}/{NUM_SAMPLES} ({successful_extractions/NUM_SAMPLES*100:.1f}%)")

    if successful_results:
        print(f"\nStrict Accuracy (getomni methodology):")
        print(f"  - Average: {avg_accuracy:.1f}%")
        print(f"  - Min: {min_accuracy:.1f}%")
        print(f"  - Max: {max_accuracy:.1f}%")

        print(f"\nDocument Quality Breakdown:")
        print(f"  - 100% accurate: {perfect_docs}/{len(successful_results)} documents")
        print(f"  - ≥90% accurate: {high_acc_docs}/{len(successful_results)} documents")
        print(f"  - ≥70% accurate: {good_acc_docs}/{len(successful_results)} documents")

        print(f"\nTiming:")
        print(f"  - QwenVL: {avg_qwen_time:.2f}s per document")
        print(f"  - Claude: {avg_claude_time:.2f}s per document")
        print(f"  - Total: {avg_qwen_time + avg_claude_time:.2f}s per document")

        avg_total_cost = (total_qwen_cost + total_claude_cost) / len(successful_results)
        print(f"\nCost:")
        print(f"  - Total cost: ${total_qwen_cost + total_claude_cost:.6f}")
        print(f"  - Average: ${avg_total_cost:.6f} per document")
        print(f"  - Projected 1,000 samples: ${avg_total_cost * 1000:.2f}")

        print(f"\nComparison to Benchmark:")
        print(f"  - GPT-4o (baseline): ~75%")
        print(f"  - Our pipeline: {avg_accuracy:.1f}%")
        if avg_accuracy > 75:
            print(f"  - Difference: +{avg_accuracy - 75:.1f}% ✅")
        else:
            print(f"  - Difference: {avg_accuracy - 75:.1f}%")

    # Save results
    print(f"\n[4/5] Saving results...")

    # Save JSON summary
    summary = {
        'config': {
            'num_samples': NUM_SAMPLES,
            'seed': SEED,
            'evaluation_method': 'strict_matching (getomni methodology)',
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'summary': {
            'success_rate': f"{successful_extractions}/{NUM_SAMPLES}",
            'avg_accuracy': round(avg_accuracy, 1) if successful_results else 0,
            'min_accuracy': round(min_accuracy, 1) if successful_results else 0,
            'max_accuracy': round(max_accuracy, 1) if successful_results else 0,
            'perfect_documents': perfect_docs if successful_results else 0,
            'high_accuracy_documents': high_acc_docs if successful_results else 0,
            'avg_cost': round(avg_total_cost, 6) if successful_results else 0,
            'projected_1000_cost': round(avg_total_cost * 1000, 2) if successful_results else 0
        },
        'samples': [
            {k: v for k, v in r.items() if k not in ['evaluation', 'extracted_json', 'ground_truth']}
            for r in results
        ]
    }

    json_path = output_dir / 'evaluation_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved summary: {json_path}")

    # Save detailed comparison files for samples
    num_to_save = min(3, len(successful_results))
    for i in range(num_to_save):
        r = successful_results[i]
        comp_path = output_dir / f"eval_detail_{r['sample_id']}_{r['format']}.txt"

        with open(comp_path, 'w') as f:
            f.write(f"Sample ID: {r['sample_id']}\n")
            f.write(f"Format: {r['format']}\n")
            f.write(f"Accuracy: {r['accuracy']:.1f}%\n\n")
            f.write(format_diff_report(r['evaluation']))

        print(f"  [OK] Saved detailed evaluation: {comp_path}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Phase 4 evaluation testing complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
