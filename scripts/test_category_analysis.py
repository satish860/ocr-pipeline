"""
Phase 5: Category Analysis (Lightweight - Find Weaknesses)

Test 15-20 documents from challenging categories to identify pipeline weaknesses.
Uses parallel processing for speed.
"""

import sys
import time
import json
import tempfile
import os
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import get_samples_by_category, get_random_samples
from src.ocr_pipeline.qwen_extractor import extract_document
from benchmark.claude_extractor import extract_json_from_markdown
from benchmark.evaluator import compare_json

# Thread-safe print lock
print_lock = threading.Lock()


def process_sample(sample, sample_num, total_samples, categories_to_test):
    """Process a single sample through the pipeline (thread-safe).

    Args:
        sample: Sample dictionary with image, schema, ground_truth, metadata
        sample_num: Sample number for display
        total_samples: Total number of samples
        categories_to_test: List of category names for categorization

    Returns:
        Result dictionary with metrics
    """
    sample_id = sample['id']
    sample_format = sample['metadata'].get('format', 'unknown')
    sample_quality = sample['metadata'].get('documentQuality', 'unknown')
    json_schema = sample['schema']
    ground_truth = sample['json_gt']

    # Determine category (format or quality)
    category = sample_format if sample_format in [c for c, _ in categories_to_test] else sample_quality

    # Create temp file for image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        sample['image'].save(tmp.name, format='PNG')
        temp_path = tmp.name

    try:
        with print_lock:
            print(f"[{sample_num}/{total_samples}] Processing ID={sample_id}, Category={category}")

        # Step 1: QwenVL extraction
        start_time = time.time()
        qwen_result = extract_document(
            temp_path,
            include_images=True,
            include_usage=True
        )
        qwen_time = time.time() - start_time

        if not qwen_result['success']:
            with print_lock:
                print(f"  [{sample_num}] QwenVL FAILED: {qwen_result['error']}")
            return {
                'sample_id': sample_id,
                'category': category,
                'format': sample_format,
                'quality': sample_quality,
                'qwen_success': False,
                'claude_success': False,
                'error': f"QwenVL failed: {qwen_result['error']}"
            }

        qwen_usage = qwen_result.get('usage', {})
        qwen_cost = qwen_usage.get('cost', 0) / 1_000_000

        # Step 2: Claude JSON extraction
        start_time = time.time()
        claude_result = extract_json_from_markdown(
            markdown=qwen_result['markdown'],
            json_schema=json_schema,
            extracted_images=qwen_result['images'],
            include_usage=True
        )
        claude_time = time.time() - start_time

        if not claude_result['success']:
            with print_lock:
                print(f"  [{sample_num}] Claude FAILED: {claude_result['error']}")
            return {
                'sample_id': sample_id,
                'category': category,
                'format': sample_format,
                'quality': sample_quality,
                'qwen_success': True,
                'claude_success': False,
                'qwen_cost': qwen_cost,
                'error': f"Claude failed: {claude_result['error']}"
            }

        claude_usage = claude_result.get('usage', {})
        prompt_tokens = claude_usage.get('prompt_tokens', 0)
        completion_tokens = claude_usage.get('completion_tokens', 0)
        cache_created = claude_result.get('cache_creation_tokens', 0)
        cache_read = claude_result.get('cache_read_tokens', 0)

        # Estimate Claude cost
        claude_cost = (prompt_tokens * 3 + completion_tokens * 15 +
                      cache_created * 3.75 + cache_read * 0.30) / 1_000_000

        # Step 3: Strict evaluation
        evaluation = compare_json(claude_result['json'], ground_truth)

        total_fields = evaluation['total_fields']
        diff_fields = evaluation['different_fields']
        accuracy = evaluation['accuracy'] * 100

        with print_lock:
            print(f"  [{sample_num}] OK Accuracy: {accuracy:.1f}% ({total_fields - diff_fields}/{total_fields} fields) | {qwen_time + claude_time:.1f}s | ${qwen_cost + claude_cost:.6f}")

        # Store result
        return {
            'sample_id': sample_id,
            'category': category,
            'format': sample_format,
            'quality': sample_quality,
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
        }

    except Exception as e:
        with print_lock:
            print(f"  [{sample_num}] ERROR: {str(e)}")
        return {
            'sample_id': sample_id,
            'category': category,
            'format': sample_format,
            'quality': sample_quality,
            'qwen_success': False,
            'claude_success': False,
            'error': str(e)
        }

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    print("=" * 70)
    print("PHASE 5: Category Analysis (Find Weaknesses)")
    print("=" * 70)

    # Configuration: Focus on challenging categories
    CATEGORIES_TO_TEST = [
        ('PHOTO', 4),           # Photo captures (harder OCR)
        ('LOW_QUALITY', 4),     # Poor scans
        ('PATIENT_INTAKE', 3),  # Often handwritten
        ('TABLE', 3),           # Complex layouts
        ('CHART', 3),           # Visual elements
    ]
    SEED = 42

    # Create output directory
    output_dir = Path("category_analysis_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Calculate total samples
    total_samples = sum(count for _, count in CATEGORIES_TO_TEST)
    print(f"\nPlanned samples: {total_samples} across {len(CATEGORIES_TO_TEST)} categories")

    # Load samples by category
    print(f"\n[1/5] Loading samples by category (seed={SEED})...")
    all_samples = []
    category_sample_counts = {}

    for category, count in CATEGORIES_TO_TEST:
        print(f"  Loading {count} samples from '{category}'...", end=" ")
        samples = get_samples_by_category(category, count, seed=SEED)

        if samples:
            all_samples.extend(samples)
            category_sample_counts[category] = len(samples)
            print(f"[OK] {len(samples)} samples loaded")
        else:
            print(f"[SKIP] No samples found")
            category_sample_counts[category] = 0

    actual_total = len(all_samples)
    print(f"\nActual samples loaded: {actual_total}")

    if actual_total == 0:
        print("\n[ERROR] No samples loaded. Exiting.")
        return

    # Show category breakdown
    print(f"\nCategory distribution:")
    for cat, count in category_sample_counts.items():
        print(f"  - {cat}: {count} sample(s)")

    # Process samples IN PARALLEL
    print(f"\n[2/5] Processing {actual_total} samples through full pipeline (IN PARALLEL)...")
    print(f"  Using ThreadPoolExecutor with {min(actual_total, 10)} workers")
    print("-" * 70)

    results = []
    start_all = time.time()

    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(actual_total, 10)  # Limit to 10 concurrent requests
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(process_sample, sample, i, actual_total, CATEGORIES_TO_TEST): (i, sample)
            for i, sample in enumerate(all_samples, 1)
        }

        # Collect results as they complete
        for future in as_completed(future_to_sample):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_all

    # Calculate costs
    total_qwen_cost = sum(r.get('qwen_cost', 0) for r in results if r.get('qwen_success'))
    total_claude_cost = sum(r.get('claude_cost', 0) for r in results if r.get('claude_success'))
    successful_extractions = sum(1 for r in results if r.get('claude_success'))

    print("-" * 70)
    print(f"\n>> Parallel processing complete in {total_time:.1f}s (vs ~{actual_total * 30:.0f}s serial)")
    print(f"  Success: {successful_extractions}/{actual_total} documents")
    print(f"  Total cost: ${total_qwen_cost + total_claude_cost:.6f}")

    # Group results by category
    print(f"\n[3/5] Analyzing results by category...")

    successful_results = [r for r in results if r.get('claude_success')]

    if not successful_results:
        print("\n[ERROR] No successful extractions. Cannot analyze.")
        return

    # Calculate per-category metrics
    category_metrics = defaultdict(lambda: {
        'accuracies': [],
        'times': [],
        'costs': [],
        'total_fields': [],
        'different_fields': [],
        'samples': []
    })

    for r in successful_results:
        cat = r['category']
        category_metrics[cat]['accuracies'].append(r['accuracy'])
        category_metrics[cat]['times'].append(r['qwen_time'] + r['claude_time'])
        category_metrics[cat]['costs'].append(r['qwen_cost'] + r['claude_cost'])
        category_metrics[cat]['total_fields'].append(r['total_fields'])
        category_metrics[cat]['different_fields'].append(r['different_fields'])
        category_metrics[cat]['samples'].append(r['sample_id'])

    # Calculate aggregate metrics
    category_stats = {}
    for cat, metrics in category_metrics.items():
        accuracies = metrics['accuracies']
        category_stats[cat] = {
            'sample_count': len(accuracies),
            'mean_accuracy': sum(accuracies) / len(accuracies),
            'min_accuracy': min(accuracies),
            'max_accuracy': max(accuracies),
            'mean_time': sum(metrics['times']) / len(metrics['times']),
            'mean_cost': sum(metrics['costs']) / len(metrics['costs']),
            'total_fields': sum(metrics['total_fields']),
            'total_different': sum(metrics['different_fields']),
            'sample_ids': metrics['samples']
        }

    # Sort by mean accuracy (ascending - worst first)
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]['mean_accuracy'])

    # Print category performance report
    print("\n" + "=" * 70)
    print("CATEGORY PERFORMANCE REPORT (WEAKNESSES FOCUS)")
    print("=" * 70)

    overall_accuracy = sum(r['accuracy'] for r in successful_results) / len(successful_results)
    print(f"\nOverall Performance:")
    print(f"  - Success rate: {successful_extractions}/{actual_total} ({successful_extractions/actual_total*100:.1f}%)")
    print(f"  - Average accuracy: {overall_accuracy:.1f}%")
    print(f"  - Total cost: ${total_qwen_cost + total_claude_cost:.6f}")

    print(f"\n[!] Weakest Categories (sorted by accuracy):")
    for i, (cat, stats) in enumerate(sorted_categories, 1):
        print(f"\n{i}. {cat} ({stats['sample_count']} samples):")
        print(f"   - Mean accuracy: {stats['mean_accuracy']:.1f}%")
        print(f"   - Range: {stats['min_accuracy']:.1f}% - {stats['max_accuracy']:.1f}%")
        print(f"   - Mean time: {stats['mean_time']:.2f}s")
        print(f"   - Mean cost: ${stats['mean_cost']:.6f}")
        print(f"   - Total errors: {stats['total_different']}/{stats['total_fields']} fields")

    # Identify common failure patterns
    print(f"\n[*] Failure Pattern Analysis:")

    # Analyze differences for weakest categories
    weakest_categories = [cat for cat, _ in sorted_categories[:3]]  # Top 3 weakest
    failure_patterns = defaultdict(int)

    for r in successful_results:
        if r['category'] in weakest_categories and r.get('evaluation'):
            for diff in r['evaluation'].get('differences', []):
                # Categorize errors
                if diff['predicted'] is None and diff['ground_truth'] is not None:
                    failure_patterns['Missing fields'] += 1
                elif diff['predicted'] is not None and diff['ground_truth'] is None:
                    failure_patterns['Extra fields'] += 1
                elif isinstance(diff['ground_truth'], str) and isinstance(diff['predicted'], str):
                    if diff['ground_truth'].replace(' ', '') == diff['predicted'].replace(' ', ''):
                        failure_patterns['Whitespace differences'] += 1
                    elif any(c.isdigit() for c in str(diff['ground_truth'])) and any(c.isdigit() for c in str(diff['predicted'])):
                        failure_patterns['Number/date format mismatches'] += 1
                    else:
                        failure_patterns['Text OCR errors'] += 1
                else:
                    failure_patterns['Type mismatches'] += 1

    if failure_patterns:
        print(f"\nTop error types in weakest categories:")
        for pattern, count in sorted(failure_patterns.items(), key=lambda x: -x[1])[:5]:
            print(f"  - {pattern}: {count} occurrences")
    else:
        print(f"\n  (No specific patterns identified)")

    # Save results
    print(f"\n[4/5] Saving results...")

    # Save JSON summary
    summary = {
        'config': {
            'categories_tested': CATEGORIES_TO_TEST,
            'total_samples': actual_total,
            'seed': SEED,
            'focus': 'find_weaknesses',
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'overall': {
            'success_rate': f"{successful_extractions}/{actual_total}",
            'avg_accuracy': round(overall_accuracy, 1),
            'total_cost': round(total_qwen_cost + total_claude_cost, 6)
        },
        'category_stats': {
            cat: {k: v for k, v in stats.items() if k != 'sample_ids'}
            for cat, stats in sorted_categories
        },
        'failure_patterns': dict(failure_patterns),
        'samples': [
            {k: v for k, v in r.items() if k not in ['evaluation']}
            for r in results
        ]
    }

    json_path = output_dir / 'category_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved summary: {json_path}")

    # Save detailed report
    report_path = output_dir / 'category_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CATEGORY PERFORMANCE REPORT (WEAKNESSES FOCUS)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Overall Performance:\n")
        f.write(f"  - Success rate: {successful_extractions}/{actual_total} ({successful_extractions/actual_total*100:.1f}%)\n")
        f.write(f"  - Average accuracy: {overall_accuracy:.1f}%\n")
        f.write(f"  - Total cost: ${total_qwen_cost + total_claude_cost:.6f}\n\n")

        f.write("Categories (sorted by accuracy, weakest first):\n\n")
        for i, (cat, stats) in enumerate(sorted_categories, 1):
            f.write(f"{i}. {cat} ({stats['sample_count']} samples):\n")
            f.write(f"   - Mean accuracy: {stats['mean_accuracy']:.1f}%\n")
            f.write(f"   - Range: {stats['min_accuracy']:.1f}% - {stats['max_accuracy']:.1f}%\n")
            f.write(f"   - Samples: {', '.join(str(s) for s in stats['sample_ids'])}\n\n")

        if failure_patterns:
            f.write("Top error types in weakest categories:\n")
            for pattern, count in sorted(failure_patterns.items(), key=lambda x: -x[1])[:5]:
                f.write(f"  - {pattern}: {count} occurrences\n")

    print(f"  [OK] Saved report: {report_path}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Phase 5 category analysis complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
