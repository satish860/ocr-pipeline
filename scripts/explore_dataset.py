"""
Dataset Exploration Script for Omni AI OCR Benchmark

This script loads the HuggingFace dataset and allows exploration of samples.
Usage: uv run python scripts/explore_dataset.py [--num-samples 3]
"""

import json
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import argparse


def explore_dataset(num_samples: int = 3, save_samples: bool = True):
    """
    Load and explore the Omni AI benchmark dataset.

    Args:
        num_samples: Number of samples to inspect
        save_samples: Whether to save sample images locally
    """
    print("Loading Omni AI OCR Benchmark dataset from HuggingFace...")
    print("Dataset: getomni-ai/ocr-benchmark")
    print()

    # Load dataset
    try:
        dataset = load_dataset("getomni-ai/ocr-benchmark")
        print("Dataset loaded successfully!")
        print(f"Splits: {list(dataset.keys())}")

        # Use test split
        test_split = dataset["test"]
        print(f"Test samples: {len(test_split)}")
        print()

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Analyze dataset structure
    print("Dataset Structure:")
    print("=" * 60)

    if len(test_split) > 0:
        first_sample = test_split[0]
        print(f"Available fields: {list(first_sample.keys())}")
        print()

    # Analyze document types and quality levels
    print("Dataset Composition:")
    print("=" * 60)

    doc_types = {}
    quality_levels = {}

    for sample in test_split:
        metadata = json.loads(sample.get("metadata", "{}"))
        doc_type = metadata.get("format", "unknown")
        quality = metadata.get("quality", "unknown")

        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        quality_levels[quality] = quality_levels.get(quality, 0) + 1

    print("\nDocument Types:")
    for doc_type, count in sorted(doc_types.items(), key=lambda x: -x[1]):
        print(f"  {doc_type}: {count} samples")

    print("\nQuality Levels:")
    for quality, count in sorted(quality_levels.items(), key=lambda x: -x[1]):
        print(f"  {quality}: {count} samples")

    print()

    # Explore individual samples
    print(f"Inspecting {num_samples} Sample(s):")
    print("=" * 60)

    # Create output directory for samples
    if save_samples:
        samples_dir = Path("input/benchmark_samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving samples to: {samples_dir}/")
        print()

    for i in range(min(num_samples, len(test_split))):
        sample = test_split[i]
        metadata = json.loads(sample.get("metadata", "{}"))
        doc_type = metadata.get("format", "unknown")
        quality = metadata.get("quality", "unknown")

        print(f"\nSample {i+1}:")
        print("-" * 60)
        print(f"  ID: {sample['id']}")
        print(f"  Document Type: {doc_type}")
        print(f"  Quality: {quality}")

        # Image info
        image = sample["image"]
        print(f"  Image Size: {image.size}")
        print(f"  Image Mode: {image.mode}")

        # Ground truth info
        true_markdown = sample["true_markdown_output"]
        print(f"  Markdown Length: {len(true_markdown)} chars")
        print(f"  Markdown Preview: {true_markdown[:100]}...")

        # JSON schema info
        try:
            json_schema = json.loads(sample["json_schema"])
            print(f"  JSON Schema Keys: {list(json_schema.keys())}")

            true_json = json.loads(sample["true_json_output"])
            print(f"  True JSON Keys: {list(true_json.keys())}")
        except Exception as e:
            print(f"  JSON Schema/Data: Error parsing - {e}")

        # Save sample if requested
        if save_samples:
            # Save image
            image_path = samples_dir / f"sample_{i+1}_{doc_type}.png"
            image.save(image_path)
            print(f"  Saved image: {image_path}")

            # Save ground truth markdown
            markdown_path = samples_dir / f"sample_{i+1}_{doc_type}_truth.md"
            markdown_path.write_text(true_markdown, encoding="utf-8")
            print(f"  Saved markdown: {markdown_path}")

            # Save JSON schema and ground truth JSON
            schema_path = samples_dir / f"sample_{i+1}_{doc_type}_schema.json"
            schema_path.write_text(sample["json_schema"], encoding="utf-8")
            print(f"  Saved schema: {schema_path}")

            json_path = samples_dir / f"sample_{i+1}_{doc_type}_truth.json"
            json_path.write_text(sample["true_json_output"], encoding="utf-8")
            print(f"  Saved JSON: {json_path}")

    print()
    print("=" * 60)
    print("Dataset exploration complete!")

    if save_samples:
        print(f"\nNext steps:")
        print(f"  1. Review saved samples in: input/benchmark_samples/")
        print(f"  2. Run your pipeline on a sample image")
        print(f"  3. Compare output with ground truth using metrics")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Explore Omni AI OCR Benchmark dataset")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of samples to inspect")
    parser.add_argument("--no-save", action="store_true", help="Don't save samples locally")

    args = parser.parse_args()

    explore_dataset(
        num_samples=args.num_samples,
        save_samples=not args.no_save
    )


if __name__ == "__main__":
    main()
