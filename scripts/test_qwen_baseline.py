"""
Phase 2: QwenVL Baseline Testing

Test the existing QwenVL pipeline on benchmark samples to establish
baseline performance metrics.
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


def main():
    print("=" * 70)
    print("PHASE 2: QwenVL Baseline Testing")
    print("=" * 70)

    # Configuration
    NUM_SAMPLES = 5
    SEED = 42

    # Create output directory
    output_dir = Path("baseline_outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load samples
    print(f"\n[1/4] Loading {NUM_SAMPLES} random samples (seed={SEED})...")
    samples = get_random_samples(NUM_SAMPLES, seed=SEED)

    # Show category breakdown
    categories = Counter(s['metadata'].get('format', 'unknown') for s in samples)
    print(f"\nCategory distribution:")
    for cat, count in categories.most_common():
        print(f"  - {cat}: {count} sample(s)")

    # Process samples
    print(f"\n[2/4] Processing {NUM_SAMPLES} samples...")
    print("-" * 70)

    results = []
    total_cost = 0
    successful = 0

    for i, sample in enumerate(samples, 1):
        sample_id = sample['id']
        sample_format = sample['metadata'].get('format', 'unknown')

        print(f"\nSample {i}/{NUM_SAMPLES}: ID={sample_id}, Format={sample_format}")

        # Save PIL image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            sample['image'].save(tmp.name, format='PNG')
            temp_path = tmp.name

        try:
            # Process with timing
            start_time = time.time()
            result = extract_document(
                temp_path,
                include_images=True,
                include_usage=True  # Enable cost tracking
            )
            elapsed = time.time() - start_time

            # Extract metrics
            usage = result.get('usage', {})
            cost_credits = usage.get('cost', 0)
            cost_dollars = cost_credits / 1_000_000  # Convert to dollars

            total_cost += cost_dollars

            if result['success']:
                successful += 1
                status = "[OK]"
            else:
                status = "[FAIL]"

            # Print progress
            print(f"  {status} Time: {elapsed:.2f}s | Cost: ${cost_dollars:.6f} | "
                  f"Tokens: {usage.get('total_tokens', 0)} | "
                  f"Elements: {len(result['elements'])} | Images: {len(result['images'])}")

            # Store result
            results.append({
                'sample_id': sample_id,
                'format': sample_format,
                'quality': sample['metadata'].get('documentQuality', 'unknown'),
                'success': result['success'],
                'error': result.get('error'),
                'processing_time': elapsed,
                'cost_dollars': cost_dollars,
                'usage': {
                    'total_tokens': usage.get('total_tokens', 0),
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'cached_tokens': usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
                },
                'markdown_length': len(result['markdown']),
                'element_count': len(result['elements']),
                'image_count': len(result['images']),
                'elements': [{'type': e['type'], 'bbox': e['bbox']} for e in result['elements']],
                'markdown': result['markdown']  # Save for potential file output
            })

        except Exception as e:
            print(f"  [ERROR] {str(e)}")
            results.append({
                'sample_id': sample_id,
                'format': sample_format,
                'success': False,
                'error': str(e),
                'processing_time': 0,
                'cost_dollars': 0
            })

        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Calculate metrics
    print(f"\n[3/4] Calculating metrics...")

    avg_time = sum(r['processing_time'] for r in results) / len(results)
    avg_cost = total_cost / len(results)
    projected_1000_cost = avg_cost * 1000

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nSuccess Rate: {successful}/{NUM_SAMPLES} ({successful/NUM_SAMPLES*100:.1f}%)")
    print(f"\nTiming:")
    print(f"  - Average time: {avg_time:.2f}s per document")
    print(f"  - Total time: {sum(r['processing_time'] for r in results):.2f}s")
    print(f"\nCost:")
    print(f"  - Total cost: ${total_cost:.6f}")
    print(f"  - Average cost: ${avg_cost:.6f} per document")
    print(f"  - Projected 1,000 samples: ${projected_1000_cost:.2f}")

    # Token statistics (only for successful runs)
    successful_results = [r for r in results if r.get('success')]
    if successful_results:
        avg_total = sum(r.get('usage', {}).get('total_tokens', 0) for r in successful_results) / len(successful_results)
        avg_prompt = sum(r.get('usage', {}).get('prompt_tokens', 0) for r in successful_results) / len(successful_results)
        avg_completion = sum(r.get('usage', {}).get('completion_tokens', 0) for r in successful_results) / len(successful_results)
        print(f"\nTokens (average):")
        print(f"  - Total: {avg_total:.0f}")
        print(f"  - Prompt: {avg_prompt:.0f}")
        print(f"  - Completion: {avg_completion:.0f}")

        avg_elements = sum(r.get('element_count', 0) for r in successful_results) / len(successful_results)
        avg_images = sum(r.get('image_count', 0) for r in successful_results) / len(successful_results)
        print(f"\nElements:")
        print(f"  - Average elements detected: {avg_elements:.1f}")
        print(f"  - Average images extracted: {avg_images:.1f}")

    # Save results
    print(f"\n[4/4] Saving results...")

    # Save JSON summary (without markdown content to keep it small)
    summary = {
        'config': {
            'num_samples': NUM_SAMPLES,
            'seed': SEED,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'summary': {
            'success_rate': f"{successful}/{NUM_SAMPLES}",
            'avg_time_seconds': round(avg_time, 2),
            'total_cost_dollars': round(total_cost, 6),
            'avg_cost_dollars': round(avg_cost, 6),
            'projected_1000_cost_dollars': round(projected_1000_cost, 2)
        },
        'samples': [
            {k: v for k, v in r.items() if k != 'markdown'}  # Exclude markdown from JSON
            for r in results
        ]
    }

    json_path = output_dir / 'baseline_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved summary: {json_path}")

    # Save 2-3 markdown files for manual inspection
    num_to_save = min(3, len(successful_results))

    for i in range(num_to_save):
        r = successful_results[i]
        md_path = output_dir / f"sample_{r['sample_id']}_{r['format']}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Sample {r['sample_id']} - {r['format']}\n\n")
            f.write(f"**Processing Time:** {r['processing_time']:.2f}s\n")
            f.write(f"**Cost:** ${r['cost_dollars']:.6f}\n")
            f.write(f"**Elements:** {r['element_count']}\n")
            f.write(f"**Images:** {r['image_count']}\n\n")
            f.write("---\n\n")
            f.write(r['markdown'])
        print(f"  [OK] Saved markdown: {md_path}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] Phase 2 baseline testing complete!")
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
