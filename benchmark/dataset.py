"""Dataset loader for getomni-ai/ocr-benchmark from HuggingFace"""

import random
import json
from typing import List, Dict, Any, Optional
from collections import Counter

from datasets import load_dataset

# Cache the dataset globally to avoid reloading
_DATASET_CACHE = None


def load_benchmark_dataset(force_reload: bool = False):
    """Load the OCR benchmark dataset from HuggingFace.

    Args:
        force_reload: If True, reload the dataset even if cached

    Returns:
        Dataset object from HuggingFace (test split)
    """
    global _DATASET_CACHE

    if _DATASET_CACHE is None or force_reload:
        print("[Loading] Dataset from HuggingFace (this may take a minute)...")
        _DATASET_CACHE = load_dataset("getomni-ai/ocr-benchmark", split="test")
        print(f"[OK] Dataset loaded: {len(_DATASET_CACHE)} samples")

    return _DATASET_CACHE


def _extract_sample_dict(sample) -> Dict[str, Any]:
    """Convert HuggingFace dataset sample to a simple dictionary.

    Args:
        sample: HuggingFace dataset sample

    Returns:
        Dictionary with keys: image, markdown_gt, json_gt, schema, metadata, id
    """
    # Parse JSON strings
    metadata = sample.get('metadata', '{}')
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    json_schema = sample.get('json_schema', '{}')
    if isinstance(json_schema, str):
        try:
            json_schema = json.loads(json_schema)
        except json.JSONDecodeError:
            json_schema = {}

    json_output = sample.get('true_json_output', '{}')
    if isinstance(json_output, str):
        try:
            json_output = json.loads(json_output)
        except json.JSONDecodeError:
            json_output = {}

    markdown_output = sample.get('true_markdown_output', '')
    # Markdown might also be JSON-encoded string
    if isinstance(markdown_output, str) and markdown_output.startswith('"'):
        try:
            markdown_output = json.loads(markdown_output)
        except json.JSONDecodeError:
            pass  # Keep as is

    return {
        'id': sample.get('id', None),
        'image': sample['image'],  # PIL Image
        'markdown_gt': markdown_output,
        'json_gt': json_output,
        'schema': json_schema,
        'metadata': metadata,
    }


def get_random_samples(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get N random samples from the dataset.

    Args:
        n: Number of samples to retrieve
        seed: Random seed for reproducibility (optional)

    Returns:
        List of sample dictionaries
    """
    dataset = load_benchmark_dataset()

    if seed is not None:
        random.seed(seed)

    # Sample indices
    total_samples = len(dataset)
    n = min(n, total_samples)  # Don't sample more than available
    indices = random.sample(range(total_samples), n)

    # Extract samples
    samples = [_extract_sample_dict(dataset[i]) for i in indices]

    return samples


def get_samples_by_category(category: str, n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get N samples from a specific category.

    Args:
        category: Category tag to filter by (e.g., 'tables', 'handwriting', 'photo')
        n: Number of samples to retrieve
        seed: Random seed for reproducibility (optional)

    Returns:
        List of sample dictionaries from the specified category
    """
    dataset = load_benchmark_dataset()

    if seed is not None:
        random.seed(seed)

    # Filter samples by category
    filtered_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        metadata = sample.get('metadata', '{}')

        # Parse JSON string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Extract tags (format or other fields might contain category info)
        tags = []
        if 'format' in metadata:
            tags.append(metadata['format'])
        if 'documentQuality' in metadata:
            tags.append(metadata['documentQuality'])

        if category.upper() in [tag.upper() for tag in tags]:
            filtered_indices.append(i)

    if not filtered_indices:
        print(f"[WARNING] No samples found with category '{category}'")
        return []

    # Sample from filtered indices
    n = min(n, len(filtered_indices))
    sampled_indices = random.sample(filtered_indices, n)

    samples = [_extract_sample_dict(dataset[i]) for i in sampled_indices]

    return samples


def get_stratified_samples(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get N samples with proportional representation across categories.

    This ensures that if 20% of the dataset has 'tables' tag, then ~20% of
    the returned samples will also have the 'tables' tag.

    Args:
        n: Total number of samples to retrieve
        seed: Random seed for reproducibility (optional)

    Returns:
        List of sample dictionaries with stratified distribution
    """
    dataset = load_benchmark_dataset()

    if seed is not None:
        random.seed(seed)

    # Group samples by their primary category
    # Note: Samples can have multiple tags, so we'll use the first tag as primary
    category_indices = {}

    for i in range(len(dataset)):
        sample = dataset[i]
        metadata = sample.get('metadata', '{}')

        # Parse JSON string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Use 'format' as primary category (e.g., TABLE, DELIVERY_NOTE, etc.)
        primary_category = metadata.get('format', 'uncategorized')

        if primary_category not in category_indices:
            category_indices[primary_category] = []
        category_indices[primary_category].append(i)

    # Calculate samples per category (proportional)
    total_samples = len(dataset)
    n = min(n, total_samples)

    sampled_indices = []
    for category, indices in category_indices.items():
        # Calculate proportion
        proportion = len(indices) / total_samples
        samples_for_category = max(1, int(n * proportion))  # At least 1 sample

        # Sample from this category
        samples_for_category = min(samples_for_category, len(indices))
        sampled = random.sample(indices, samples_for_category)
        sampled_indices.extend(sampled)

    # If we have too many samples (due to rounding), randomly remove some
    if len(sampled_indices) > n:
        sampled_indices = random.sample(sampled_indices, n)

    # If we have too few samples (unlikely), add more randomly
    if len(sampled_indices) < n:
        remaining = n - len(sampled_indices)
        all_indices = set(range(total_samples))
        available = list(all_indices - set(sampled_indices))
        sampled_indices.extend(random.sample(available, remaining))

    # Extract samples
    samples = [_extract_sample_dict(dataset[i]) for i in sampled_indices]

    return samples


def get_dataset_stats() -> Dict[str, Any]:
    """Get statistics about the benchmark dataset.

    Returns:
        Dictionary with dataset statistics:
        - total_samples: Total number of samples
        - categories: Dict of category -> count
        - avg_schema_fields: Average number of fields in JSON schemas
        - min_schema_fields: Minimum number of fields
        - max_schema_fields: Maximum number of fields
    """
    dataset = load_benchmark_dataset()

    # Collect all tags
    all_tags = []
    schema_field_counts = []

    for i in range(len(dataset)):
        sample = dataset[i]

        # Extract metadata
        metadata = sample.get('metadata', '{}')

        # Parse JSON string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Collect format and quality tags
        if 'format' in metadata:
            all_tags.append(metadata['format'])
        if 'documentQuality' in metadata:
            all_tags.append(metadata['documentQuality'])

        # Count schema fields
        schema = sample.get('json_schema', '{}')

        # Parse JSON string
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except json.JSONDecodeError:
                schema = {}

        if isinstance(schema, dict):
            properties = schema.get('properties', {})
            schema_field_counts.append(len(properties))
        else:
            schema_field_counts.append(0)

    # Count category occurrences
    category_counts = Counter(all_tags)

    # Calculate schema stats
    avg_fields = sum(schema_field_counts) / len(schema_field_counts) if schema_field_counts else 0
    min_fields = min(schema_field_counts) if schema_field_counts else 0
    max_fields = max(schema_field_counts) if schema_field_counts else 0

    return {
        'total_samples': len(dataset),
        'categories': dict(category_counts),
        'avg_schema_fields': round(avg_fields, 1),
        'min_schema_fields': min_fields,
        'max_schema_fields': max_fields,
    }


def get_all_categories() -> List[str]:
    """Get a list of all unique category tags in the dataset.

    Returns:
        List of category strings
    """
    dataset = load_benchmark_dataset()

    all_tags = set()

    for i in range(len(dataset)):
        sample = dataset[i]
        metadata = sample.get('metadata', '{}')

        # Parse JSON string
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        # Collect all metadata fields as tags
        if 'format' in metadata:
            all_tags.add(metadata['format'])
        if 'documentQuality' in metadata:
            all_tags.add(metadata['documentQuality'])
        if 'fontFamily' in metadata:
            all_tags.add(metadata['fontFamily'])

    return sorted(list(all_tags))
