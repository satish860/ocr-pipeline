# OCR Pipeline Benchmark Plan

## Overview
This document outlines the benchmarking strategy for our OCR pipeline using the [Omni AI benchmark dataset](https://huggingface.co/datasets/getomni-ai/ocr-benchmark). The goal is to measure performance on both markdown quality and JSON extraction accuracy, with baseline comparison against GPT-4o and Gemini 2.0.

## Benchmark Framework: Omni AI

**Repository**: https://github.com/getomni-ai/benchmark
**Dataset**: https://huggingface.co/datasets/getomni-ai/ocr-benchmark

### Key Features
- **1,000 diverse document images** across 8+ document types
- **Dual evaluation metrics**: Text similarity (markdown) + JSON accuracy (structured data)
- **Quality tiers**: HIGH_QUALITY, CLEAN, PHOTO, LOW_QUALITY
- **Ground truth provided**: Both markdown and structured JSON outputs
- **Standardized schemas**: Pre-defined JSON schemas for each document type

### Evaluation Metrics

#### 1. Text Similarity (Markdown Quality)
- **Method**: Normalized Levenshtein distance
- **Formula**: `1 - (edit_distance / max_length)`
- **Range**: 0.0 - 1.0 (higher is better)
- **Purpose**: Measures how accurately we extract and format text in markdown

#### 2. JSON Accuracy (Structured Data Extraction)
- **Method**: Modified JSON-diff algorithm
- **Formula**: `1 - (difference_fields / total_fields)`
- **Range**: 0.0 - 1.0 (higher is better)
- **Purpose**: Measures accuracy of structured data extraction matching schemas

---

## Phase 1: Infrastructure Setup

### 1.1 Create Benchmark Module Structure

Create a new benchmark module in the project:

```
scripts/
  benchmark/
    __init__.py              # Package initialization
    dataset_loader.py        # HuggingFace dataset loader
    evaluator.py             # Metrics implementation (Levenshtein + JSON-diff)
    runner.py                # Main benchmark orchestration
    subset_selector.py       # Smart stratified sampling
    baseline_runner.py       # GPT-4o/Gemini baseline comparison
    visualizer.py            # Results visualization and reporting
    config.yaml              # Benchmark configuration
```

### 1.2 Install Dependencies ✅ COMPLETED (2025-01-13)

**Status**: All 6 dependencies installed and verified
- datasets (4.4.1)
- python-Levenshtein (0.27.3)
- google-generativeai (0.8.5)
- pandas (2.3.3)
- matplotlib (3.10.7)
- seaborn (0.13.2)

Add to `pyproject.toml`:

```toml
[project.dependencies]
# Existing dependencies...
datasets = "^2.14.0"           # HuggingFace datasets
python-Levenshtein = "^0.21.0" # Text similarity
openai = "^1.0.0"              # GPT-4o baseline
google-generativeai = "^0.3.0" # Gemini baseline
pandas = "^2.0.0"              # Results analysis
matplotlib = "^3.7.0"          # Visualization
seaborn = "^0.12.0"            # Statistical visualization
```

**Installation command**:
```bash
uv add datasets python-Levenshtein openai google-generativeai pandas matplotlib seaborn
```

---

## Phase 2: Metrics Implementation

### 2.1 Text Similarity Metric (evaluator.py)

Implement normalized Levenshtein distance:

```python
from Levenshtein import distance

def calculate_text_similarity(predicted: str, ground_truth: str) -> float:
    """
    Calculate normalized Levenshtein distance between markdown outputs.

    Args:
        predicted: Your pipeline's markdown output
        ground_truth: Ground truth markdown from dataset

    Returns:
        Score from 0.0 to 1.0 (1.0 = perfect match)
    """
    edit_dist = distance(predicted, ground_truth)
    max_len = max(len(predicted), len(ground_truth))

    if max_len == 0:
        return 1.0

    return 1.0 - (edit_dist / max_len)
```

### 2.2 JSON Accuracy Metric (evaluator.py)

Implement modified JSON-diff algorithm:

```python
def calculate_json_accuracy(predicted: dict, ground_truth: dict) -> float:
    """
    Calculate field-level accuracy for structured JSON extraction.

    Args:
        predicted: Your pipeline's JSON output
        ground_truth: Ground truth JSON from dataset

    Returns:
        Score from 0.0 to 1.0 (1.0 = perfect match)
    """
    total_fields = count_fields(ground_truth)
    diff_fields = count_differences(predicted, ground_truth)

    if total_fields == 0:
        return 1.0

    return 1.0 - (diff_fields / total_fields)

def count_fields(obj) -> int:
    """Recursively count all fields in nested JSON."""
    if isinstance(obj, dict):
        return sum(1 + count_fields(v) for v in obj.values())
    elif isinstance(obj, list):
        return sum(count_fields(item) for item in obj)
    else:
        return 1

def count_differences(pred, truth) -> int:
    """Recursively count differing fields."""
    # Implementation handles:
    # - Missing/extra keys
    # - Type mismatches
    # - Value differences
    # - Array length differences
    # - Nested object comparisons
```

### 2.3 Additional Metrics

Track supplementary performance indicators:

```python
class BenchmarkMetrics:
    def __init__(self):
        self.text_similarity_scores = []
        self.json_accuracy_scores = []
        self.processing_times = []
        self.api_costs = []
        self.error_count = 0
        self.per_category_scores = {}

    def calculate_summary(self):
        return {
            "overall_text_similarity": mean(self.text_similarity_scores),
            "overall_json_accuracy": mean(self.json_accuracy_scores),
            "avg_processing_time": mean(self.processing_times),
            "total_cost": sum(self.api_costs),
            "error_rate": self.error_count / total_samples,
            "by_document_type": self.per_category_scores
        }
```

---

## Phase 3: Dataset Preparation

### 3.1 Smart Subset Selection (subset_selector.py)

Create stratified sample ensuring diversity:

**Target**: 80-100 samples covering all document types and quality levels

**Document Type Distribution**:
- TABLE: 12 samples (structured data tables)
- CHART: 12 samples (business charts, graphs)
- DELIVERY_NOTE: 10 samples (shipping documents)
- EQUIPMENT_INSPECTION: 10 samples (checklists)
- BANK_CHECK: 10 samples (financial documents)
- REAL_ESTATE: 8 samples (transaction summaries)
- SHIFT_SCHEDULE: 8 samples (employee schedules)
- COMMERCIAL_LEASE_AGREEMENT: 8 samples (legal contracts)
- Other types: 12 samples (glossaries, invoices, etc.)

**Quality Level Distribution** (across all types):
- HIGH_QUALITY: 25%
- CLEAN: 30%
- PHOTO: 25%
- LOW_QUALITY: 20%

**Special Cases** (must include):
- At least 5 rotated documents (test rotation correction)
- At least 5 multi-page complex documents
- At least 5 documents with dense tables
- At least 5 documents with charts/graphs

```python
def select_benchmark_subset(dataset, target_size=80):
    """
    Select stratified sample from full dataset.

    Returns:
        List of sample indices ensuring diversity
    """
    # Stratify by document type
    # Balance quality levels
    # Include edge cases
    # Ensure reproducibility (fixed seed)
```

### 3.2 Dataset Loader (dataset_loader.py)

```python
from datasets import load_dataset
from PIL import Image
import json

class BenchmarkDataset:
    def __init__(self, subset_indices=None):
        """Load Omni AI benchmark dataset."""
        self.dataset = load_dataset("getomni-ai/ocr-benchmark")
        self.test_split = self.dataset["test"]

        if subset_indices:
            self.test_split = self.test_split.select(subset_indices)

    def get_sample(self, idx):
        """Get a single sample with all ground truth data."""
        sample = self.test_split[idx]

        return {
            "id": sample["id"],
            "image": sample["image"],  # PIL Image
            "format": sample["format"],  # Document type
            "quality": sample["quality"],  # Quality tier
            "json_schema": json.loads(sample["json_schema"]),
            "true_json": json.loads(sample["true_json"]),
            "true_markdown": sample["true_markdown"]
        }

    def __len__(self):
        return len(self.test_split)
```

### 3.3 Cache Ground Truth

Pre-load and cache all ground truth data to avoid repeated parsing:

```python
def cache_ground_truth(dataset, cache_dir="results/cache"):
    """Pre-process and cache all ground truth data."""
    for idx in range(len(dataset)):
        sample = dataset.get_sample(idx)

        # Save parsed JSON
        # Save markdown
        # Save image metadata
        # Index by document ID
```

---

## Phase 4: Pipeline Enhancement for JSON Extraction

### 4.1 Two-Stage Processing Architecture

**Current**: Image → Markdown
**Enhanced**: Image → Markdown → JSON

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────┐
│ Stage 1: Your Pipeline          │
│ - ImagePreprocessor             │
│ - LayoutDetector (Qwen-30B)     │
│ - RegionExtractor               │
│ - OCRExtractor (Gemini Flash)   │
│ - SpatialAnalyzer               │
└──────┬──────────────────────────┘
       │
       ↓ [Markdown Output]
       │
┌──────┴──────────────────────────┐
│ Stage 2: JSON Extraction        │
│ - Gemini Flash 2.5              │
│ - Input: Markdown + JSON Schema │
│ - Output: Structured JSON       │
└──────┬──────────────────────────┘
       │
       ↓
┌─────────────┐
│ JSON Output │
└─────────────┘
```

### 4.2 JSON Extractor Implementation

Create new module: `src/ocr_pipeline/json_extractor.py`

```python
import google.generativeai as genai
from typing import Dict, Any
import json

class JSONExtractor:
    """Converts markdown to structured JSON using LLM."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def extract(self, markdown: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract structured JSON from markdown according to schema.

        Args:
            markdown: Markdown text from OCR pipeline
            json_schema: JSON schema defining expected structure

        Returns:
            Structured JSON matching the schema
        """
        prompt = self._build_extraction_prompt(markdown, json_schema)

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,  # Low temp for consistency
                "response_mime_type": "application/json"
            }
        )

        return json.loads(response.text)

    def _build_extraction_prompt(self, markdown: str, schema: Dict) -> str:
        """Build prompt for JSON extraction."""
        return f"""Extract structured data from the following markdown text according to the provided JSON schema.

**JSON Schema:**
```json
{json.dumps(schema, indent=2)}
```

**Markdown Content:**
```markdown
{markdown}
```

Extract all fields defined in the schema. If a field is not present in the markdown, use null or an appropriate default value based on the schema type.

Return ONLY valid JSON matching the schema structure."""
```

### 4.3 Unified Pipeline Wrapper

Create: `scripts/benchmark/pipeline_wrapper.py`

```python
import requests
from io import BytesIO
import json

class OCRPipelineWrapper:
    """Wrapper for OCR pipeline with JSON extraction."""

    def __init__(self, api_url="http://localhost:8000", openrouter_key=None):
        self.api_url = api_url
        self.json_extractor = JSONExtractor(openrouter_key)

    def process(self, image_path: str, json_schema: Dict = None):
        """
        Process image through full pipeline.

        Args:
            image_path: Path to input image
            json_schema: Optional JSON schema for structured extraction

        Returns:
            {
                "markdown": str,
                "json": dict or None,
                "metadata": {...}
            }
        """
        # Stage 1: Call existing OCR pipeline
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.api_url}/ocr",
                files={"file": f}
            )

        result = response.json()
        markdown = result["markdown"]

        # Stage 2: Extract JSON if schema provided
        json_output = None
        if json_schema:
            json_output = self.json_extractor.extract(markdown, json_schema)

        return {
            "markdown": markdown,
            "json": json_output,
            "metadata": {
                "detected_elements": result["detected_elements"],
                "rotation_correction": result["rotation_correction_degrees"],
                "processing_time": result.get("processing_time", 0)
            }
        }
```

---

## Phase 5: Baseline Comparison

### 5.1 Baseline Model Implementations

Create: `scripts/benchmark/baseline_runner.py`

```python
import openai
import google.generativeai as genai
import base64
import json

class GPT4oBaseline:
    """Direct image-to-JSON extraction using GPT-4o."""

    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def process(self, image_path: str, json_schema: Dict):
        """Process image directly to JSON."""
        with open(image_path, 'rb') as f:
            image_b64 = base64.b64encode(f.read()).decode()

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_prompt(json_schema)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

class Gemini2Baseline:
    """Direct image-to-JSON extraction using Gemini 2.0 Flash."""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    def process(self, image_path: str, json_schema: Dict):
        """Process image directly to JSON."""
        from PIL import Image

        image = Image.open(image_path)
        prompt = self._build_prompt(json_schema)

        response = self.model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )

        return json.loads(response.text)
```

### 5.2 Comparison Configuration

Define which models to compare:

```yaml
# scripts/benchmark/config.yaml

models:
  - name: "ocr-pipeline-2stage"
    enabled: true
    type: "custom"
    description: "Our OCR Pipeline (Qwen-30B + Gemini Flash) + JSON extraction"

  - name: "gpt-4o"
    enabled: true
    type: "baseline"
    description: "OpenAI GPT-4o (direct image-to-JSON)"

  - name: "gemini-2.0-flash"
    enabled: true
    type: "baseline"
    description: "Google Gemini 2.0 Flash (direct image-to-JSON)"

benchmark:
  subset_size: 80
  random_seed: 42
  output_dir: "results/benchmark"
  cache_dir: "results/cache"
```

---

## Phase 6: Benchmark Execution

### 6.1 Main Runner (runner.py)

```python
import time
from tqdm import tqdm
import json
from pathlib import Path

class BenchmarkRunner:
    """Orchestrates the full benchmark execution."""

    def __init__(self, config_path="scripts/benchmark/config.yaml"):
        self.config = self._load_config(config_path)
        self.dataset = BenchmarkDataset()
        self.evaluator = Evaluator()
        self.results = []

    def run(self):
        """Execute full benchmark."""
        print("🚀 Starting OCR Pipeline Benchmark")
        print(f"📊 Dataset: {len(self.dataset)} samples")
        print(f"🤖 Models: {len(self.config['models'])} models")

        # Select subset
        subset_indices = select_benchmark_subset(
            self.dataset,
            target_size=self.config['benchmark']['subset_size']
        )

        print(f"✅ Selected {len(subset_indices)} samples")

        # Run each model
        for model_config in self.config['models']:
            if not model_config['enabled']:
                continue

            print(f"\n🔄 Running: {model_config['name']}")
            model_results = self._run_model(model_config, subset_indices)
            self.results.append(model_results)

            # Save intermediate results
            self._save_results()

        # Generate report
        self._generate_report()

        print("\n✅ Benchmark complete!")

    def _run_model(self, model_config, subset_indices):
        """Run a single model on all samples."""
        model = self._initialize_model(model_config)
        results = []

        for idx in tqdm(subset_indices, desc=f"{model_config['name']}"):
            sample = self.dataset.get_sample(idx)

            try:
                start_time = time.time()

                # Process image
                output = model.process(
                    image_path=self._save_temp_image(sample['image']),
                    json_schema=sample['json_schema']
                )

                processing_time = time.time() - start_time

                # Evaluate
                metrics = {
                    "text_similarity": self.evaluator.calculate_text_similarity(
                        output.get('markdown', ''),
                        sample['true_markdown']
                    ),
                    "json_accuracy": self.evaluator.calculate_json_accuracy(
                        output.get('json', {}),
                        sample['true_json']
                    ),
                    "processing_time": processing_time
                }

                results.append({
                    "sample_id": sample['id'],
                    "format": sample['format'],
                    "quality": sample['quality'],
                    "metrics": metrics,
                    "success": True
                })

            except Exception as e:
                results.append({
                    "sample_id": sample['id'],
                    "format": sample['format'],
                    "quality": sample['quality'],
                    "error": str(e),
                    "success": False
                })

        return {
            "model": model_config['name'],
            "results": results,
            "summary": self._calculate_summary(results)
        }
```

### 6.2 Progress Tracking

Features:
- Real-time progress bar with tqdm
- Save intermediate results every 10 samples
- Error logging with full traceback
- Estimated time and cost tracking
- Console output with color-coded status

```python
def run_with_checkpoints(self, checkpoint_interval=10):
    """Run benchmark with periodic checkpoints."""
    for i, sample in enumerate(tqdm(subset)):
        result = self.process_sample(sample)
        self.results.append(result)

        # Checkpoint
        if (i + 1) % checkpoint_interval == 0:
            self._save_checkpoint(i + 1)
            self._log_progress(i + 1)
```

---

## Phase 7: Results Analysis

### 7.1 Generate Comparison Report (visualizer.py)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BenchmarkVisualizer:
    """Generate comprehensive benchmark reports."""

    def generate_report(self, results, output_dir):
        """Create full benchmark report."""

        # 1. Overall Summary Table
        self._create_summary_table(results)

        # 2. Per-Category Breakdown
        self._create_category_breakdown(results)

        # 3. Quality Tier Analysis
        self._create_quality_analysis(results)

        # 4. Speed vs Accuracy Plot
        self._create_speed_accuracy_plot(results)

        # 5. Cost Comparison
        self._create_cost_analysis(results)

        # 6. Generate HTML Report
        self._create_html_report(results, output_dir)
```

### 7.2 Report Components

#### Summary Table

| Model | Text Similarity | JSON Accuracy | Avg Time (s) | Total Cost ($) | Error Rate |
|-------|-----------------|---------------|--------------|----------------|------------|
| OCR Pipeline (2-stage) | 0.XX | 0.XX | X.XX | $X.XX | X.X% |
| GPT-4o | 0.XX | 0.XX | X.XX | $X.XX | X.X% |
| Gemini 2.0 Flash | 0.XX | 0.XX | X.XX | $X.XX | X.X% |

#### Per-Category Breakdown

```python
def create_category_breakdown(results):
    """Analyze performance by document type."""

    categories = ["TABLE", "CHART", "DELIVERY_NOTE", "BANK_CHECK", ...]

    for category in categories:
        category_results = filter_by_category(results, category)

        print(f"\n{category}:")
        print(f"  Text Similarity: {mean(scores):.3f}")
        print(f"  JSON Accuracy: {mean(json_scores):.3f}")
        print(f"  Sample Count: {len(category_results)}")
```

#### Visualization Examples

**1. Accuracy by Document Type (Bar Chart)**
```python
def plot_accuracy_by_type(results):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped bar chart
    # X-axis: Document types
    # Y-axis: Accuracy scores
    # Groups: Different models

    plt.savefig("accuracy_by_type.png")
```

**2. Score Distribution (Box Plots)**
```python
def plot_score_distribution(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Text similarity distribution
    # Right: JSON accuracy distribution
    # Box plots for each model

    plt.savefig("score_distribution.png")
```

**3. Time vs Accuracy Scatter**
```python
def plot_time_vs_accuracy(results):
    # Scatter plot: processing time vs accuracy
    # Color by model
    # Size by document complexity

    plt.savefig("time_vs_accuracy.png")
```

### 7.3 Error Analysis Tools

```python
class ErrorAnalyzer:
    """Tools for analyzing failures and low-scoring samples."""

    def identify_failure_patterns(self, results):
        """Find common patterns in failed/low-scoring samples."""

        failures = [r for r in results if r['metrics']['json_accuracy'] < 0.5]

        patterns = {
            "by_document_type": self._group_by_field(failures, 'format'),
            "by_quality": self._group_by_field(failures, 'quality'),
            "by_error_type": self._classify_errors(failures)
        }

        return patterns

    def export_failed_cases(self, results, output_dir):
        """Export failed samples for manual review."""

        for failure in failures:
            # Save: input image, your output, ground truth, diff
            self._create_comparison_view(failure, output_dir)

    def highlight_field_mismatches(self, predicted_json, truth_json):
        """Create visual diff showing which fields differ."""

        diff = self._recursive_diff(predicted_json, truth_json)

        # Generate HTML with highlighted differences
        return self._render_diff_html(diff)
```

---

## Phase 8: Iteration Framework

### 8.1 Regression Testing

After each pipeline improvement:

```python
def run_regression_test(baseline_results_path, new_run_name):
    """Compare new run against previous baseline."""

    baseline = load_results(baseline_results_path)
    new_results = run_benchmark()

    comparison = {
        "text_similarity": {
            "baseline": baseline['avg_text_similarity'],
            "new": new_results['avg_text_similarity'],
            "change": new_results['avg_text_similarity'] - baseline['avg_text_similarity']
        },
        "json_accuracy": {
            "baseline": baseline['avg_json_accuracy'],
            "new": new_results['avg_json_accuracy'],
            "change": new_results['avg_json_accuracy'] - baseline['avg_json_accuracy']
        }
    }

    # Check for regressions
    if comparison['json_accuracy']['change'] < -0.05:
        print("⚠️  WARNING: Regression detected in JSON accuracy!")

    return comparison
```

### 8.2 Improvement Tracking

Track improvements over time:

```python
class ImprovementTracker:
    """Track benchmark scores across multiple runs."""

    def __init__(self, history_path="results/history.json"):
        self.history = self._load_history(history_path)

    def add_run(self, run_name, results):
        """Add new benchmark run to history."""
        self.history[run_name] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": results['summary'],
            "git_commit": self._get_git_commit()
        }
        self._save_history()

    def plot_improvement_trend(self):
        """Plot metric improvements over time."""
        runs = sorted(self.history.items(), key=lambda x: x[1]['timestamp'])

        # Line plot: accuracy over runs
        # Show trend line
        # Highlight significant improvements
```

---

## Expected Deliverables

### Files Created

1. **`scripts/benchmark/runner.py`** - Main executable
2. **`scripts/benchmark/evaluator.py`** - Metrics implementation
3. **`scripts/benchmark/dataset_loader.py`** - Dataset interface
4. **`scripts/benchmark/pipeline_wrapper.py`** - Pipeline integration
5. **`scripts/benchmark/baseline_runner.py`** - Baseline models
6. **`scripts/benchmark/visualizer.py`** - Report generation
7. **`scripts/benchmark/config.yaml`** - Configuration
8. **`src/ocr_pipeline/json_extractor.py`** - JSON extraction module

### Output Files

1. **`results/benchmark_TIMESTAMP/`** - Run directory
   - `results.json` - Raw results
   - `summary.json` - Aggregated metrics
   - `report.html` - Interactive report
   - `plots/` - Visualizations

2. **`results/error_analysis/`** - Failed cases
   - Side-by-side comparisons
   - Error categorization
   - Field-level diffs

3. **`results/history.json`** - Historical tracking

---

## Success Metrics

### Minimum Viable Benchmark (MVP)
- ✅ Successfully processes 80+ samples without crashes
- ✅ Calculates both text similarity and JSON accuracy metrics
- ✅ Generates comparison report vs GPT-4o and Gemini baselines
- ✅ Identifies top 3-5 improvement areas with specific examples
- ✅ Reproducible results (same scores on re-run)

### Target Performance Goals
- 🎯 **Text Similarity (Markdown)**: > 80% avg accuracy
- 🎯 **JSON Accuracy**: > 70% avg accuracy (acceptable for first iteration)
- 🎯 **Processing Speed**: < 30s per document average
- 🎯 **Cost Efficiency**: Competitive with GPT-4o (< $0.10 per document)
- 🎯 **Reliability**: < 5% error rate

### Stretch Goals
- 🌟 Text similarity > 85%
- 🌟 JSON accuracy > 80%
- 🌟 Faster than baselines (leveraging parallelization)
- 🌟 Lower cost than GPT-4o/Gemini
- 🌟 Best-in-class performance on specific categories (e.g., forms, tables)

---

## Estimated Effort & Cost

### Development Time

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Infrastructure setup | 4-6 hours |
| 2 | Metrics implementation | 2-3 hours |
| 3 | Dataset preparation | 1-2 hours |
| 4 | JSON extraction integration | 3-4 hours |
| 5 | Baseline implementations | 2-3 hours |
| 6 | Benchmark execution | 2 hours (mostly runtime) |
| 7 | Analysis & visualization | 3-4 hours |
| 8 | Documentation | 1-2 hours |
| **Total** | **Full benchmark setup** | **18-26 hours** |

### API Costs (80 sample subset)

**Your Pipeline (2-stage)**:
- Layout detection (Qwen-30B): 80 images × $0.XX = $X.XX
- OCR (Gemini Flash 2.5): 80 images × ~10 regions × $0.XX = $X.XX
- JSON extraction (Gemini Flash): 80 extractions × $0.XX = $X.XX
- **Subtotal**: ~$10-15

**GPT-4o Baseline**:
- Direct image-to-JSON: 80 images × $0.XX = $X.XX
- **Subtotal**: ~$5-8

**Gemini 2.0 Baseline**:
- Direct image-to-JSON: 80 images × $0.XX = $X.XX
- **Subtotal**: ~$3-5

**Total Estimated Cost**: $18-28 for complete benchmark with all 3 models

---

## Key Advantages of This Approach

### ✅ Standalone & Simple
- Pure Python (no TypeScript/Node.js complexity)
- Minimal dependencies
- Easy to debug and iterate

### ✅ Fast Iteration
- Small subset (80 samples) allows rapid experimentation
- Checkpoint system prevents losing progress
- Quick turnaround for testing improvements

### ✅ Comprehensive Evaluation
- Tests both markdown quality and JSON extraction
- Compares against state-of-the-art baselines
- Multiple metrics and visualizations

### ✅ Actionable Insights
- Clear identification of weak points
- Per-category breakdown shows where to focus
- Error analysis with specific examples

### ✅ Reproducible
- Fixed random seed for subset selection
- Standardized metrics
- Versioned configuration
- Git commit tracking

### ✅ Extensible
- Easy to add new models for comparison
- Can expand to full 1,000 sample dataset
- Can add custom metrics
- Can integrate with CI/CD for continuous benchmarking

---

## Next Steps

1. **Phase 1**: Set up infrastructure and install dependencies
2. **Phase 2**: Implement evaluation metrics
3. **Phase 3**: Load and prepare dataset subset
4. **Phase 4**: Add JSON extraction capability
5. **Phase 5**: Implement baseline models
6. **Phase 6**: Run initial benchmark
7. **Phase 7**: Analyze results and identify improvements
8. **Phase 8**: Iterate and optimize

Let's start with Phase 1! 🚀
