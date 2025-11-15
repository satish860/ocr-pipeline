# OCR Pipeline Benchmark Implementation Plan

## Overview
Evaluate the QwenVL + Claude pipeline against the Omni OCR benchmark dataset (1,000 documents) to identify **strengths** and **weaknesses** in document extraction performance.

**Architecture:**
```
Input Image → QwenVL (markdown + base64) → Claude Sonnet 4.5 (JSON extraction) → Compare with ground truth
```

**Key Principles:**
- ✅ No changes to `qwen_extractor.py` (keeps OCR layer clean)
- ✅ Separate JSON extraction using Claude Sonnet 4.5
- ✅ Phase-by-phase implementation with clear success criteria
- ✅ Test immediately after each phase
- ✅ Cost-conscious (start small, scale up)

---

## Phase 1: Dataset Integration 📊

### Goal
Load and explore the benchmark dataset to understand structure, categories, and ground truth format.

### Tasks
1. Install `datasets` library via `uv add datasets`
2. Create `src/ocr_pipeline/benchmark/dataset.py`:
   - Load `getomni-ai/ocr-benchmark` from HuggingFace
   - Function to get N random samples
   - Function to get samples by category
   - Extract image, JSON schema, ground truth JSON, metadata
3. Create exploration script `explore_dataset.py`:
   - Display 5 random samples
   - Show image, metadata, JSON schema, ground truth
   - List all document categories
   - Count samples per category

### Success Criteria
- ✅ `uv run python explore_dataset.py` runs without errors
- ✅ Can view 5 sample images from dataset
- ✅ Can see ground truth JSON schemas and outputs
- ✅ Document categories identified (financial, medical, commercial, etc.)
- ✅ Understand dataset structure (1,000 samples, fields, formats)

### Testing Steps
```bash
# Install dependency
uv add datasets

# Run exploration
uv run python explore_dataset.py

# Expected output:
# - 5 sample images displayed/saved
# - Category breakdown printed
# - Sample JSON schemas shown
# - Total dataset size confirmed
```

### Cost Impact
**$0** - Just loading and viewing data

### Deliverables
- [ ] `src/ocr_pipeline/benchmark/__init__.py`
- [ ] `src/ocr_pipeline/benchmark/dataset.py`
- [ ] `explore_dataset.py` (temporary exploration script)
- [ ] Understanding of dataset structure documented

---

## Phase 2: QwenVL Baseline 🔍

### Goal
Run existing QwenVL pipeline on 10 benchmark samples to establish baseline and verify compatibility.

### Tasks
1. Create `test_qwen_baseline.py`:
   - Load 10 samples from dataset
   - Run `extract_document()` on each
   - Save markdown outputs
   - Display extracted images count
   - Calculate per-document processing time
   - Estimate cost for full 1,000 sample run
2. Manual review of outputs:
   - Read 3-5 markdown outputs
   - Verify special elements detected (tables, signatures, etc.)
   - Check if markdown is reasonably accurate

### Success Criteria
- ✅ Successfully process 10 benchmark images through existing pipeline
- ✅ All 10 return `success: True`
- ✅ Markdown output looks reasonable (manual inspection)
- ✅ Base64 images extracted for special elements
- ✅ Cost per document calculated (QwenVL API costs)
- ✅ Average processing time per document measured
- ✅ **Zero changes to `qwen_extractor.py`**

### Testing Steps
```bash
# Run baseline test
uv run python test_qwen_baseline.py

# Expected output:
# Processing sample 1/10... ✓ (3.2s)
# Processing sample 2/10... ✓ (2.8s)
# ...
#
# Results:
# - Success rate: 10/10 (100%)
# - Avg time: 3.1s per document
# - Avg cost: $0.XX per document
# - Total cost for 1,000 samples: ~$XX.XX
#
# Markdown outputs saved to: ./baseline_outputs/
```

### Cost Impact
**~$0.50 - $2.00** for 10 samples (QwenVL only)

### Deliverables
- [ ] `test_qwen_baseline.py`
- [ ] `./baseline_outputs/` folder with 10 markdown files
- [ ] Cost estimate for full run
- [ ] Performance metrics (time, success rate)

---

## Phase 3: Claude JSON Extraction 🤖

### Goal
Build Claude Sonnet 4.5 JSON extractor to convert markdown → structured JSON.

### Tasks
1. Install Anthropic SDK: `uv add anthropic`
2. Add `ANTHROPIC_API_KEY` to `.env.example`
3. Create `src/ocr_pipeline/benchmark/claude_extractor.py`:
   - Function: `extract_json_from_markdown(markdown: str, json_schema: dict) -> dict`
   - Use Claude Sonnet 4.5 via Anthropic API
   - Prompt: "Extract structured data from this markdown according to the provided JSON schema"
   - Parse and validate JSON response
   - Handle API errors with retries
4. Create `test_claude_extraction.py`:
   - Load 5 samples
   - Run QwenVL → markdown
   - Run Claude → JSON
   - Display extracted JSON vs ground truth (side by side)
   - Calculate cost per extraction

### Success Criteria
- ✅ Claude successfully extracts JSON from markdown
- ✅ JSON structure matches schema format (valid JSON)
- ✅ Handles nested objects and arrays
- ✅ Error handling works (retry on failure)
- ✅ Cost per extraction tracked
- ✅ Manual inspection: 3/5 extractions look reasonable

### Testing Steps
```bash
# Set API key
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Test extraction
uv run python test_claude_extraction.py

# Expected output:
# Sample 1: Bank Check
# - QwenVL extraction: ✓ (2.5s)
# - Claude JSON extraction: ✓ (1.2s)
# - Ground truth fields: 12
# - Extracted fields: 12
# - Cost: $0.XX
#
# [Shows JSON comparison]
```

### Cost Impact
**~$1.00 - $3.00** for 5 samples (QwenVL + Claude)

### Deliverables
- [ ] `src/ocr_pipeline/benchmark/claude_extractor.py`
- [ ] `test_claude_extraction.py`
- [ ] Updated `.env.example` with `ANTHROPIC_API_KEY`
- [ ] Cost estimate for Claude extraction

---

## Phase 4: Evaluation Metrics 📏

### Goal
Implement JSON comparison and accuracy calculation to measure extraction quality.

### Tasks
1. Create `src/ocr_pipeline/benchmark/evaluator.py`:
   - Function: `compare_json(predicted: dict, ground_truth: dict) -> dict`
   - Calculate field-level accuracy: `1 - (mismatches / total_fields)`
   - Identify specific mismatched fields
   - Handle different data types (string/number normalization)
   - Support nested objects and arrays
   - Return detailed diff report
2. Create `test_evaluation.py`:
   - Load 10 samples
   - Run full pipeline: Image → QwenVL → Claude → JSON
   - Compare with ground truth
   - Display accuracy per document
   - Show mismatched fields for failed extractions
3. Validate metrics:
   - Manually verify 3 samples
   - Check if accuracy metric makes sense
   - Review mismatched fields

### Success Criteria
- ✅ Accuracy metric calculated for 10 documents
- ✅ Accuracy matches manual validation (spot-check 3 documents)
- ✅ Mismatched fields correctly identified
- ✅ Handles edge cases: missing fields, null values, type mismatches
- ✅ Detailed error report generated
- ✅ Average accuracy > 50% (baseline sanity check)

### Testing Steps
```bash
# Run evaluation test
uv run python test_evaluation.py

# Expected output:
# Processing 10 samples...
#
# Document 1: Bank Check (financial)
#   Accuracy: 85.7% (12/14 fields correct)
#   Mismatches:
#     - amount: "1234.56" vs "1,234.56" (format diff)
#     - date: "2024-01-15" vs "January 15, 2024" (format diff)
#
# Document 2: Invoice (commercial)
#   Accuracy: 92.3% (12/13 fields correct)
#   Mismatches:
#     - total: missing in extraction
#
# ...
#
# Summary:
# - Mean accuracy: 78.5%
# - Median accuracy: 82.0%
# - Min accuracy: 45.0%
# - Max accuracy: 100.0%
```

### Cost Impact
**~$2.00 - $5.00** for 10 samples (QwenVL + Claude)

### Deliverables
- [ ] `src/ocr_pipeline/benchmark/evaluator.py`
- [ ] `test_evaluation.py`
- [ ] Validation report (manual check of 3 samples)
- [ ] Confirmed accuracy metric is working correctly

---

## Phase 5: Category Analysis 📈

### Goal
Run large-scale evaluation and analyze performance by document category to identify strengths/weaknesses.

### Tasks
1. Create `src/ocr_pipeline/benchmark/analyzer.py`:
   - Function: `analyze_by_category(results: List[dict]) -> dict`
   - Group results by document category
   - Calculate per-category metrics:
     - Mean/median/min/max accuracy
     - Success rate (% above 80% threshold)
     - Common failure patterns
   - Identify strengths (>90% accuracy)
   - Identify weaknesses (<70% accuracy)
2. Create `src/ocr_pipeline/benchmark/runner.py`:
   - Function: `run_benchmark(sample_size: int, cache_results: bool)`
   - Orchestrate full pipeline for N samples
   - Progress tracking with ETA
   - Result caching to avoid re-processing
   - Checkpoint/resume support
   - Cost tracking
3. Create `src/ocr_pipeline/benchmark/cli.py`:
   - `benchmark run --sample-size N`
   - `benchmark analyze`
   - `benchmark report --format [json|text]`
4. Run evaluation:
   - Start with 50 samples
   - Review category breakdown
   - Expand to 100 samples if results look good
   - Run full 1,000 samples for final report

### Success Criteria
- ✅ Successfully process 50-100 samples end-to-end
- ✅ Category breakdown displayed (e.g., financial: 85%, medical: 72%, charts: 65%)
- ✅ Strengths identified (categories with >90% accuracy)
- ✅ Weaknesses identified (categories with <70% accuracy)
- ✅ Actionable insights generated (e.g., "Tables: strong, Handwritten: weak")
- ✅ Results cached for re-analysis without re-processing
- ✅ CLI works: `uv run python -m ocr_pipeline.benchmark.cli run --sample-size 50`

### Testing Steps
```bash
# Run benchmark on 50 samples
uv run python -m ocr_pipeline.benchmark.cli run --sample-size 50

# Expected output:
# Estimating cost... ~$5.00 for 50 samples
# Proceed? [y/N]: y
#
# Processing samples: ████████░░ 45/50 (90%) | ETA: 2m 15s
#
# Results cached to: ./benchmark_results/2025-01-15_143022.json

# Analyze results
uv run python -m ocr_pipeline.benchmark.cli analyze

# Expected output:
# 📊 Category Performance Report
#
# Overall: 76.3% mean accuracy
#
# 🟢 Strengths (>90% accuracy):
#   - Clean financial documents: 93.2%
#   - Simple forms: 91.5%
#
# 🟡 Moderate (70-90% accuracy):
#   - Invoices: 82.1%
#   - Medical reports: 78.5%
#   - Tables: 74.2%
#
# 🔴 Weaknesses (<70% accuracy):
#   - Low-quality scans: 62.3%
#   - Handwritten forms: 58.7%
#   - Complex charts: 55.1%
#
# 💡 Insights:
#   - QwenVL+Claude excels at clean typed documents
#   - Struggles with handwritten content
#   - Table extraction needs improvement for complex layouts
```

### Cost Impact
- **50 samples**: ~$5.00 - $10.00
- **100 samples**: ~$10.00 - $20.00
- **1,000 samples**: ~$100.00 - $200.00

### Deliverables
- [ ] `src/ocr_pipeline/benchmark/analyzer.py`
- [ ] `src/ocr_pipeline/benchmark/runner.py`
- [ ] `src/ocr_pipeline/benchmark/cli.py`
- [ ] `src/ocr_pipeline/benchmark/reporter.py`
- [ ] Benchmark results for 50-100 samples
- [ ] Category performance report
- [ ] Strengths and weaknesses documented

---

## Project Structure (Final)

```
ocr-pipeline/
├── benchmarkplan.md              # This file
├── src/
│   └── ocr_pipeline/
│       ├── benchmark/
│       │   ├── __init__.py
│       │   ├── dataset.py        # HuggingFace loader
│       │   ├── claude_extractor.py  # Claude JSON extraction
│       │   ├── evaluator.py      # JSON diff + accuracy
│       │   ├── analyzer.py       # Category analysis
│       │   ├── runner.py         # Pipeline orchestration
│       │   ├── reporter.py       # Result formatting
│       │   └── cli.py            # Benchmark CLI
│       ├── qwen_extractor.py     # UNCHANGED
│       └── cli.py                # UNCHANGED
├── explore_dataset.py            # Phase 1 script
├── test_qwen_baseline.py         # Phase 2 script
├── test_claude_extraction.py     # Phase 3 script
└── test_evaluation.py            # Phase 4 script
```

---

## Timeline Estimate

- **Phase 1**: 2-3 hours (dataset integration + exploration)
- **Phase 2**: 1-2 hours (baseline testing)
- **Phase 3**: 3-4 hours (Claude integration + testing)
- **Phase 4**: 2-3 hours (evaluation metrics + validation)
- **Phase 5**: 4-6 hours (full pipeline + analysis)

**Total**: 12-18 hours over 3-4 days

---

## Dependencies to Add

```toml
# pyproject.toml
[project.dependencies]
datasets = "*"          # Phase 1
anthropic = "*"         # Phase 3
```

---

## Next Steps

1. ✅ **Read this plan thoroughly**
2. ⏭️ **Start with Phase 1** - Dataset Integration
3. ⏸️ Pause after each phase to review results
4. 🔁 Only proceed to next phase after success criteria met
5. 💰 Monitor costs closely (start small, scale up)

---

## Notes

- All test scripts (`explore_dataset.py`, `test_*.py`) are **temporary** for validation
- Core functionality goes in `src/ocr_pipeline/benchmark/`
- Results are cached to avoid expensive re-processing
- Start with 5-10 samples per phase for testing
- Scale to 50-100 samples in Phase 5
- Full 1,000 sample run is optional (costly but comprehensive)

---

**Ready to start Phase 1? Let's go! 🚀**
