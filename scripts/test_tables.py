"""
Process 15 TABLE samples through the OCR pipeline.

This script loads 15 samples from the TABLE category and runs them through
the full pipeline (QwenVL extraction + Claude JSON extraction + evaluation).
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path
from collections import defaultdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import get_samples_by_category
from src.ocr_pipeline.qwen_extractor import extract_document
from benchmark.claude_extractor import extract_json_from_markdown
from benchmark.evaluator import compare_json


def main():
    print("=" * 70)
    print("TABLE SAMPLES ANALYSIS")
    print("=" * 70)

    # Configuration
    CATEGORY = 'TABLE'
    SAMPLE_COUNT = 15
    SEED = 42

    # Create output directory
    output_dir = Path("table_analysis_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load samples
    print(f"\n[1/4] Loading {SAMPLE_COUNT} samples from '{CATEGORY}' category (seed={SEED})...")
    samples = get_samples_by_category(CATEGORY, SAMPLE_COUNT, seed=SEED)

    if not samples:
        print(f"\n[ERROR] No samples found for category '{CATEGORY}'")
        return

    print(f"[OK] Loaded {len(samples)} samples")

    # Process each sample
    print(f"\n[2/4] Processing {len(samples)} samples through full pipeline...")
    print("-" * 70)

    results = []
    total_qwen_cost = 0
    total_claude_cost = 0

    for i, sample in enumerate(samples, 1):
        sample_id = sample['id']
        sample_format = sample['metadata'].get('format', 'unknown')
        json_schema = sample['schema']
        ground_truth = sample['json_gt']

        print(f"\n[{i}/{len(samples)}] Processing Sample ID={sample_id}")

        # Create temp file for image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sample['image'].save(tmp.name, format='PNG')
            temp_path = tmp.name

        try:
            # Step 1: QwenVL extraction
            print(f"  [1/3] QwenVL extraction...", end=" ")
            start_time = time.time()
            qwen_result = extract_document(
                temp_path,
                include_images=True,
                include_usage=True
            )
            qwen_time = time.time() - start_time

            if not qwen_result['success']:
                print(f"FAILED - {qwen_result['error']}")
                results.append({
                    'sample_id': sample_id,
                    'format': sample_format,
                    'qwen_success': False,
                    'error': f"QwenVL failed: {qwen_result['error']}"
                })
                continue

            qwen_usage = qwen_result.get('usage', {})
            qwen_cost = qwen_usage.get('cost', 0) / 1_000_000
            total_qwen_cost += qwen_cost
            print(f"OK ({qwen_time:.1f}s, ${qwen_cost:.6f})")

            # Step 2: Claude JSON extraction
            print(f"  [2/3] Claude JSON extraction...", end=" ")
            start_time = time.time()
            claude_result = extract_json_from_markdown(
                markdown=qwen_result['markdown'],
                json_schema=json_schema,
                extracted_images=qwen_result['images'],
                include_usage=True
            )
            claude_time = time.time() - start_time

            if not claude_result['success']:
                print(f"FAILED - {claude_result['error']}")
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

            # Estimate Claude cost (rough estimate for GPT-5-mini pricing)
            claude_cost = (prompt_tokens * 3 + completion_tokens * 15) / 1_000_000
            total_claude_cost += claude_cost
            print(f"OK ({claude_time:.1f}s, ${claude_cost:.6f})")

            # Step 3: Evaluation
            print(f"  [3/3] Evaluating accuracy...", end=" ")
            evaluation = compare_json(claude_result['json'], ground_truth)

            total_fields = evaluation['total_fields']
            diff_fields = evaluation['different_fields']
            accuracy = evaluation['accuracy'] * 100

            print(f"Accuracy: {accuracy:.1f}% ({total_fields - diff_fields}/{total_fields} fields correct)")

            # Show top errors if any
            if diff_fields > 0 and len(evaluation['differences']) > 0:
                print(f"      Top errors:")
                for j, diff in enumerate(evaluation['differences'][:3], 1):
                    print(f"        {j}. {diff['path']}: '{diff['predicted']}' vs '{diff['ground_truth']}'")

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
                'top_errors': evaluation['differences'][:5]  # Save top 5 errors
            })

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            results.append({
                'sample_id': sample_id,
                'format': sample_format,
                'error': str(e)
            })

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Calculate summary statistics
    print("\n" + "-" * 70)
    print("\n[3/4] Summary Statistics")
    print("=" * 70)

    successful_results = [r for r in results if r.get('claude_success')]
    success_count = len(successful_results)

    if success_count > 0:
        avg_accuracy = sum(r['accuracy'] for r in successful_results) / success_count
        min_accuracy = min(r['accuracy'] for r in successful_results)
        max_accuracy = max(r['accuracy'] for r in successful_results)
        avg_time = sum(r['qwen_time'] + r['claude_time'] for r in successful_results) / success_count

        print(f"\nOverall Performance:")
        print(f"  - Success rate: {success_count}/{len(samples)} ({success_count/len(samples)*100:.1f}%)")
        print(f"  - Average accuracy: {avg_accuracy:.1f}%")
        print(f"  - Accuracy range: {min_accuracy:.1f}% - {max_accuracy:.1f}%")
        print(f"  - Average processing time: {avg_time:.1f}s per document")
        print(f"  - Total cost: ${total_qwen_cost + total_claude_cost:.6f}")
        print(f"      - QwenVL: ${total_qwen_cost:.6f}")
        print(f"      - Claude: ${total_claude_cost:.6f}")

        # Show breakdown by accuracy
        print(f"\nAccuracy Distribution:")
        perfect = sum(1 for r in successful_results if r['accuracy'] == 100)
        high = sum(1 for r in successful_results if 90 <= r['accuracy'] < 100)
        medium = sum(1 for r in successful_results if 70 <= r['accuracy'] < 90)
        low = sum(1 for r in successful_results if r['accuracy'] < 70)

        print(f"  - Perfect (100%): {perfect} samples")
        print(f"  - High (90-99%): {high} samples")
        print(f"  - Medium (70-89%): {medium} samples")
        print(f"  - Low (<70%): {low} samples")

        # Show samples with lowest accuracy
        if success_count > 0:
            print(f"\nSamples with lowest accuracy:")
            sorted_results = sorted(successful_results, key=lambda x: x['accuracy'])
            for r in sorted_results[:3]:
                print(f"  - ID {r['sample_id']}: {r['accuracy']:.1f}% ({r['different_fields']} errors)")

    else:
        print("\n[ERROR] No successful extractions!")

    # Save results
    print(f"\n[4/4] Saving results...")

    # Save JSON summary
    summary = {
        'config': {
            'category': CATEGORY,
            'sample_count': len(samples),
            'seed': SEED,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'overall': {
            'success_rate': f"{success_count}/{len(samples)}",
            'avg_accuracy': round(avg_accuracy, 1) if success_count > 0 else 0,
            'min_accuracy': round(min_accuracy, 1) if success_count > 0 else 0,
            'max_accuracy': round(max_accuracy, 1) if success_count > 0 else 0,
            'total_cost': round(total_qwen_cost + total_claude_cost, 6)
        },
        'samples': results
    }

    json_path = output_dir / 'table_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved summary: {json_path}")

    # Save detailed report
    report_path = output_dir / 'table_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TABLE SAMPLES ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Configuration:\n")
        f.write(f"  - Category: {CATEGORY}\n")
        f.write(f"  - Samples: {len(samples)}\n")
        f.write(f"  - Seed: {SEED}\n\n")

        if success_count > 0:
            f.write(f"Overall Performance:\n")
            f.write(f"  - Success rate: {success_count}/{len(samples)} ({success_count/len(samples)*100:.1f}%)\n")
            f.write(f"  - Average accuracy: {avg_accuracy:.1f}%\n")
            f.write(f"  - Accuracy range: {min_accuracy:.1f}% - {max_accuracy:.1f}%\n")
            f.write(f"  - Total cost: ${total_qwen_cost + total_claude_cost:.6f}\n\n")

            f.write("Sample Results:\n\n")
            for i, r in enumerate(results, 1):
                if r.get('claude_success'):
                    f.write(f"{i}. Sample ID {r['sample_id']}:\n")
                    f.write(f"   - Accuracy: {r['accuracy']:.1f}%\n")
                    f.write(f"   - Fields: {r['total_fields'] - r['different_fields']}/{r['total_fields']} correct\n")
                    f.write(f"   - Time: {r['qwen_time'] + r['claude_time']:.1f}s\n")
                    if r['different_fields'] > 0:
                        f.write(f"   - Top errors:\n")
                        for err in r['top_errors'][:3]:
                            f.write(f"      * {err['path']}: '{err['predicted']}' vs '{err['ground_truth']}'\n")
                    f.write("\n")

    print(f"  [OK] Saved report: {report_path}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] TABLE analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
