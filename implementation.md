# LaTeX Validator & Corrector Implementation Plan

## Overview

**Goal:** Improve OCR accuracy by adding validation and correction layers to the LaTeX table extraction pipeline.

**Current Baseline:** 84.8% accuracy (LaTeX-only with normalization)

**Target:** 88-92% accuracy (5-8% improvement)

---

## Architecture

```
┌─────────────┐
│   Image     │
└──────┬──────┘
       │
       ↓
┌─────────────────────┐
│ QwenVL Extraction   │ → Markdown with LaTeX tables + table images (base64)
└──────┬──────────────┘
       │
       ↓
┌─────────────────────┐
│ Validator (Gemini)  │ ← Table image (base64)
│                     │ ← LaTeX table
│ Detection Only      │
└──────┬──────────────┘
       │
       ↓ (Validation Report)
┌─────────────────────┐
│ Corrector           │ ← Validation errors
│                     │ ← Table image (base64)
│ Apply Fixes         │ ← Original LaTeX
└──────┬──────────────┘
       │
       ↓ (Corrected LaTeX)
┌─────────────────────┐
│ Return Markdown     │ → Improved LaTeX quality
└─────────────────────┘
```

---

## Phase 1: Create Validator Module

### File: `src/ocr_pipeline/latex_validator.py`

**Purpose:** Detect errors in LaTeX tables by comparing with original table images.

**Key Functions:**

#### 1. `validate_latex_table()`
```python
def validate_latex_table(
    table_image_base64: str,
    latex_table: str,
    prompt: str = VALIDATION_PROMPT_TEMPLATE
) -> dict:
    """
    Validate LaTeX table against image using Gemini 2.0 Flash.

    Args:
        table_image_base64: Base64-encoded PNG of table region
        latex_table: LaTeX table string from QwenVL
        prompt: Validation instructions template

    Returns:
        {
            "valid": bool,
            "confidence": float,
            "errors": [
                {
                    "location": {"row": int, "column": int},
                    "latex_value": str,
                    "image_value": str,
                    "error_type": str,
                    "confidence": float
                }
            ],
            "summary": {
                "total_errors": int,
                "critical": int,
                "high": int,
                "low": int
            }
        }
    """
```

#### 2. `parse_validation_response()`
```python
def parse_validation_response(response_text: str) -> dict:
    """
    Parse Gemini's JSON response into validation report.

    Handles:
    - Clean JSON extraction (remove markdown fences)
    - Error categorization by severity
    - Confidence scoring
    """
```

#### 3. `VALIDATION_PROMPT_TEMPLATE`
```python
VALIDATION_PROMPT_TEMPLATE = """
You are a LaTeX table validator.

TASK: Compare the LaTeX table below with the table shown in the image.

LATEX TABLE:
```
{latex_table}
```

VALIDATION STEPS:
1. Count rows in image vs LaTeX (exact count)
2. Count columns in image vs LaTeX (exact count)
3. For each cell: compare LaTeX value with image value character-by-character
4. Check for missing/extra rows (especially empty rows at start/end)
5. Check for digit errors (common OCR mistakes: 0/O, 1/I/l, 5/S, 7/1, 8/B, 6/G)
6. Check for spelling errors
7. Check for value swaps (e.g., rate vs amount fields)

CRITICAL RULES:
- Only report DEFINITE errors (confidence > 0.7)
- If a cell is unclear/blurry in the image, mark error_type="unclear" and confidence < 0.5
- Be precise with locations (zero-indexed row/column)
- Compare digit-by-digit for numbers

OUTPUT FORMAT (JSON only, no explanations):
{{
    "valid": true/false,
    "errors": [
        {{
            "location": {{"row": <int>, "column": <int>}},
            "latex_value": "<exact value from LaTeX>",
            "image_value": "<exact value you see in image>",
            "error_type": "digit_mismatch|spelling|missing_row|extra_row|value_swap|unclear",
            "confidence": <0.0-1.0>
        }}
    ]
}}

RETURN ONLY THE JSON OBJECT ABOVE. NO OTHER TEXT.
"""
```

**API Integration:**
- Model: `google/gemini-2.0-flash` via OpenRouter
- Temperature: 0 (deterministic)
- Clean separation: image, text, and prompt are separate parameters
- Timeout: 60 seconds

**Error Categories:**
- `digit_mismatch`: OCR digit confusion (150 vs 180)
- `spelling`: Text misspelling (Litco vs Liteo)
- `missing_row`: Row in image but not in LaTeX
- `extra_row`: Row in LaTeX but not in image
- `value_swap`: Fields swapped (rate/amount confusion)
- `unclear`: Image too blurry to verify

**Confidence Levels:**
- `>0.8`: High confidence (definite error)
- `0.6-0.8`: Medium confidence (likely error)
- `<0.6`: Low confidence (uncertain, needs review)

---

### Phase 1 Success Criteria

**Must Pass Before Moving to Phase 2:**

1. ✅ **Module Import**: `from ocr_pipeline.latex_validator import validate_latex_table` works
2. ✅ **Function Signature**: Accepts all required parameters (table_image_base64, latex_table, prompt)
3. ✅ **Return Structure**: Returns valid dict with keys: `valid`, `errors`, `confidence`, `summary`
4. ✅ **API Connection**: Successfully calls Gemini 2.0 Flash via OpenRouter (no connection errors)
5. ✅ **Error Detection**: Detects at least 1 error type on manual test (digit_mismatch or spelling)
6. ✅ **Real Sample Test**: Manual test on sample ID=20 correctly identifies known errors
7. ✅ **Error Handling**: Handles API errors gracefully (timeout → returns `{"valid": True, "errors": []}`)

**Exit Gate:** Can validate one table image and return a structured error report.

**How to Test:**
```python
# Manual test script
from ocr_pipeline.latex_validator import validate_latex_table
import base64

# Load sample ID=20 table image
with open("test_data/sample_20_table.png", "rb") as f:
    table_image_base64 = base64.b64encode(f.read()).decode()

# Test LaTeX (with known error: "M. Maskos & Sons" instead of "Sico Serve")
latex_table = r"\begin{tabular}{|l|l|} ... \end{tabular}"

# Run validator
result = validate_latex_table(table_image_base64, latex_table)

# Verify
assert result['valid'] == False  # Should detect error
assert len(result['errors']) > 0  # Should find at least one error
assert result['errors'][0]['error_type'] in ['digit_mismatch', 'spelling', 'missing_row']
print("✅ Phase 1 Success!")
```

---

## Phase 2: Create Corrector Module

### File: `src/ocr_pipeline/latex_corrector.py`

**Purpose:** Apply corrections to LaTeX based on validation report.

**Key Functions:**

#### 1. `correct_latex_table()`
```python
def correct_latex_table(
    latex_table: str,
    validation_report: dict,
    table_image_base64: str,
    strategy: str = "conservative"
) -> dict:
    """
    Correct errors in LaTeX table based on validation report.

    Args:
        latex_table: Original LaTeX table
        validation_report: Output from validate_latex_table()
        table_image_base64: Original table image for re-extraction
        strategy: "conservative" | "auto" | "aggressive"

    Returns:
        {
            "corrected_latex": str,
            "corrections_applied": [
                {
                    "error_id": str,
                    "location": dict,
                    "original_value": str,
                    "corrected_value": str,
                    "method": str
                }
            ],
            "corrections_skipped": [
                {
                    "error_id": str,
                    "reason": str
                }
            ]
        }
    """
```

#### 2. `apply_correction()`
```python
def apply_correction(
    latex: str,
    error: dict,
    image_base64: str
) -> str:
    """
    Apply a single correction to LaTeX.

    Strategies by error type:
    - digit_mismatch: Direct value replacement
    - spelling: Direct value replacement
    - missing_row: Re-extract with focused prompt
    - value_swap: Swap field values
    - unclear: Skip (needs manual review)
    """
```

#### 3. `decide_correction()`
```python
def decide_correction(
    error: dict,
    strategy: str
) -> bool:
    """
    Decide if an error should be corrected based on strategy.

    Strategies:
    - conservative: Only fix errors with confidence > 0.8
    - auto: Fix errors with confidence > 0.6
    - aggressive: Fix all errors except "unclear"
    """
```

**Correction Methods:**

1. **Direct Replacement** (digit_mismatch, spelling)
   - Replace wrong value with correct value in LaTeX
   - Simple string replacement

2. **Focused Re-extraction** (missing_row)
   - Call Gemini with focused prompt
   - Target specific rows/areas
   - Merge with existing LaTeX

3. **Structural Fix** (extra_row, value_swap)
   - Parse LaTeX structure
   - Remove/rearrange rows
   - Rebuild LaTeX table

**Correction Strategies:**

| Strategy | Threshold | Use Case |
|----------|-----------|----------|
| conservative | >0.8 | Production (safe) |
| auto | >0.6 | Balanced |
| aggressive | >0.3 | Experimental |

---

### Phase 2 Success Criteria

**Must Pass Before Moving to Phase 3:**

1. ✅ **Module Import**: `from ocr_pipeline.latex_corrector import correct_latex_table` works
2. ✅ **Function Signature**: Accepts validation_report, latex_table, table_image_base64, strategy
3. ✅ **Return Structure**: Returns dict with `corrected_latex`, `corrections_applied`, `corrections_skipped`
4. ✅ **Strategy Implementation**: All 3 strategies work (conservative, auto, aggressive)
5. ✅ **Correction Types**: Can apply at least 2 correction types (digit_mismatch + spelling)
6. ✅ **Detailed Logging**: Returns complete logs of what was changed and what was skipped
7. ✅ **Real Improvement**: Manual test on sample ID=20 shows measurable accuracy gain (>10% improvement)
8. ✅ **Edge Cases**: Handles empty errors list, low confidence errors, unclear errors gracefully

**Exit Gate:** Can correct errors in one table and demonstrate measurable improvement.

**How to Test:**
```python
# Manual test script (continues from Phase 1)
from ocr_pipeline.latex_corrector import correct_latex_table

# Use validation_report from Phase 1 test
validation_report = {
    "valid": False,
    "errors": [
        {
            "location": {"row": 2, "column": 0},
            "latex_value": "M. Maskos & Sons",
            "image_value": "Sico Serve",
            "error_type": "spelling",
            "confidence": 0.9
        }
    ]
}

# Run corrector with conservative strategy
result = correct_latex_table(
    latex_table=latex_table,
    validation_report=validation_report,
    table_image_base64=table_image_base64,
    strategy="conservative"
)

# Verify
assert result['corrected_latex'] != latex_table  # Should be different
assert len(result['corrections_applied']) > 0  # Should have applied corrections
assert "Sico Serve" in result['corrected_latex']  # Should contain corrected value
print(f"✅ Phase 2 Success! Applied {len(result['corrections_applied'])} corrections")

# Test strategy differences
conservative = correct_latex_table(..., strategy="conservative")
aggressive = correct_latex_table(..., strategy="aggressive")
assert len(aggressive['corrections_applied']) >= len(conservative['corrections_applied'])
```

---

## Phase 3: Integrate into Pipeline

### File: `src/ocr_pipeline/qwen_extractor.py`

**Changes to `extract_document()`:**

```python
def extract_document(
    image_path: str,
    min_pixels: int = 512 * 32 * 32,
    max_pixels: int = 2048 * 32 * 32,
    include_images: bool = True,
    include_usage: bool = False,
    validate_tables: bool = False,  # NEW PARAMETER
    correction_strategy: str = "conservative"  # NEW PARAMETER
) -> Dict:
```

**Integration Flow:**

1. Run QwenVL extraction (existing code)
2. If `validate_tables=True`:
   - Extract LaTeX tables from markdown
   - For each table:
     - Run validator
     - If errors found, run corrector
     - Replace LaTeX in markdown
3. Return improved markdown

**Code Structure:**

```python
# After QwenVL extraction
if validate_tables and extracted_images:
    # Get table images
    table_images = [img for img in extracted_images if img['type'] == 'Table']

    if table_images:
        # Extract LaTeX tables from markdown
        latex_tables = extract_latex_from_markdown(markdown_with_images)

        corrected_tables = []
        for i, latex_table in enumerate(latex_tables):
            # Validate
            validation_report = validate_latex_table(
                table_image_base64=table_images[i]['base64'],
                latex_table=latex_table
            )

            # Correct if needed
            if not validation_report['valid']:
                correction_result = correct_latex_table(
                    latex_table=latex_table,
                    validation_report=validation_report,
                    table_image_base64=table_images[i]['base64'],
                    strategy=correction_strategy
                )
                corrected_tables.append(correction_result['corrected_latex'])
            else:
                corrected_tables.append(latex_table)

        # Replace in markdown
        markdown_with_images = replace_latex_in_markdown(
            markdown_with_images,
            latex_tables,
            corrected_tables
        )
```

**Helper Functions Needed:**

```python
def extract_latex_from_markdown(markdown: str) -> List[str]:
    """Extract all LaTeX table blocks from markdown."""

def replace_latex_in_markdown(
    markdown: str,
    old_tables: List[str],
    new_tables: List[str]
) -> str:
    """Replace LaTeX tables in markdown."""
```

---

### Phase 3 Success Criteria

**Must Pass Before Moving to Phase 4:**

1. ✅ **Helper Functions**: `extract_latex_from_markdown()` and `replace_latex_in_markdown()` work correctly
2. ✅ **Parameter Addition**: New parameters `validate_tables` and `correction_strategy` added to `extract_document()`
3. ✅ **Backward Compatibility**: Existing API still works: `extract_document(image_path)` without new params
4. ✅ **End-to-End Test**: Process 1 complete sample with `validate_tables=True` successfully
5. ✅ **Image Matching**: Table images correctly matched to LaTeX tables (same count, same order)
6. ✅ **No Regression**: Test with `validate_tables=False` still works exactly as before
7. ✅ **Exports Updated**: New functions exported in `src/ocr_pipeline/__init__.py`
8. ✅ **Integration Flow**: Validation → Correction → Replacement pipeline works end-to-end

**Exit Gate:** One complete sample processes end-to-end with validation enabled.

**How to Test:**
```python
# End-to-end integration test
from ocr_pipeline import extract_document

# Test 1: Backward compatibility (should work unchanged)
result_no_validation = extract_document("test_data/sample_01.png", include_images=True)
assert result_no_validation['success'] == True
print("✅ Backward compatibility maintained")

# Test 2: With validation enabled
result_with_validation = extract_document(
    "test_data/sample_20.png",
    include_images=True,
    validate_tables=True,
    correction_strategy="conservative"
)
assert result_with_validation['success'] == True
assert 'markdown' in result_with_validation
assert 'images' in result_with_validation
print("✅ Validation pipeline works end-to-end")

# Test 3: Verify table extraction/replacement
markdown = result_with_validation['markdown']
assert r'\begin{tabular}' in markdown  # LaTeX tables present
assert len(result_with_validation['images']) > 0  # Table images extracted

# Test 4: Compare with and without validation (should be different for sample 20)
if result_with_validation['markdown'] != result_no_validation['markdown']:
    print("✅ Validation made corrections (expected for sample 20)")
```

---

## Phase 4: Test on Worst Samples

### File: `scripts/test_worst_samples.py` (update existing)

**Test Cases:**

1. **Sample ID=4** (Complex table)
   - Baseline: 93.5% (LaTeX-only)
   - Previous HTML attempt: 3.8% (catastrophic failure)
   - Expected: Validator should detect structure errors
   - Target: >90% with validation

2. **Sample ID=20** (Empty rows)
   - Baseline: 21.1% (missing rows problem)
   - Issue: Reads "M. Maskos & Sons" instead of "Sico Serve"
   - Expected: Validator should detect missing rows
   - Target: >50% with validation

**Update Script:**

```python
# Add validation
qwen_result = extract_document(
    temp_path,
    include_images=True,
    include_usage=True,
    validate_tables=True,  # ENABLE VALIDATION
    correction_strategy="conservative"
)
```

**Success Criteria:**
- Validator detects known errors
- Corrector improves accuracy
- No regression on other samples

---

### Phase 4 Success Criteria

**Must Pass Before Moving to Phase 5:**

1. ✅ **Script Execution**: `scripts/test_worst_samples.py` runs without crashes on samples ID=4 and ID=20
2. ✅ **Error Detection**: Validator detects errors in both samples (errors list not empty for at least one sample)
3. ✅ **Accuracy Improvement**: At least 1 sample shows accuracy improvement over baseline
4. ✅ **No Regression (ID=4)**: Sample ID=4 maintains >90% accuracy (no regression from 93.5% baseline)
5. ✅ **Target Improvement (ID=20)**: Sample ID=20 shows improvement from 21.1% baseline (target >30%)
6. ✅ **Results Documented**: Before/after comparison documented with specific accuracy numbers
7. ✅ **No Catastrophic Failures**: No sample drops to <10% accuracy (validation shouldn't make things worse)
8. ✅ **Validator Quality**: Validation reports are sensible (errors detected match actual errors in image)

**Exit Gate:** Proven improvement on at least 1 worst-case sample with documented results.

**How to Test:**
```bash
# Run test on worst samples
uv run python scripts/test_worst_samples.py

# Expected output for sample ID=20:
# Baseline (no validation): 21.1%
# With validation: >30% (target), >50% (stretch goal)
```

**Documentation Required:**
```markdown
## Phase 4 Test Results

### Sample ID=4 (Complex Table)
- Baseline: 93.5%
- With Validation: ___%
- Errors Detected: [list error types]
- Corrections Applied: [list corrections]
- Status: ✅ No regression / ⚠️ Needs review

### Sample ID=20 (Missing Rows)
- Baseline: 21.1%
- With Validation: ___%
- Errors Detected: [list error types]
- Corrections Applied: [list corrections]
- Improvement: ___% (absolute gain)
- Status: ✅ Target met / ⚠️ Needs improvement
```

---

## Phase 5: Full Benchmark Test

### File: `scripts/test_tables.py` (enable validation)

**Changes:**

```python
qwen_result = extract_document(
    temp_path,
    include_images=True,
    include_usage=True,
    validate_tables=True,  # ENABLE VALIDATION
    correction_strategy="conservative"
)
```

**Metrics to Track:**

1. **Accuracy:**
   - Overall accuracy (target: 88-92%)
   - Per-sample improvement
   - Error reduction by category

2. **Performance:**
   - Latency per sample (validator + corrector)
   - API call count
   - Cost per sample

3. **Validation Quality:**
   - False positive rate (flagged correct values)
   - False negative rate (missed errors)
   - Correction success rate

**Expected Results:**

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| Overall Accuracy | 84.8% | 88-92% | >92% |
| Sample ID=4 | 93.5% | >90% | >95% |
| Sample ID=20 | 21.1% | >50% | >70% |
| Avg Time/Sample | 45s | <60s | <50s |

---

### Phase 5 Success Criteria

**Final Validation Before Production:**

1. ✅ **Complete Processing**: All 15 samples process successfully without crashes
2. ✅ **Accuracy Measurement**: Overall accuracy measured and compared to 84.8% baseline
3. ✅ **Target Achievement**: 88-92% accuracy achieved OR documented root cause analysis if not
4. ✅ **Performance Acceptable**: <75s average per sample (vs 45s baseline = +67% max overhead)
5. ✅ **Cost Acceptable**: <$0.60 total for 15 samples (vs $0.407 baseline = +47% max increase)
6. ✅ **Metrics Tracked**: Accuracy, latency, API calls, and cost per sample all recorded
7. ✅ **Error Analysis**: Breakdown by error type (digit_mismatch, spelling, missing_row, etc.)
8. ✅ **Results Documented**: Complete benchmark report with before/after comparison

**Exit Gate:** Full validation system proven on entire benchmark suite with production-ready metrics.

**How to Test:**
```bash
# Run full benchmark with validation
uv run python scripts/test_tables.py

# Expected output:
# Processing 15 samples with validation enabled...
# Overall accuracy: ___%
# Baseline: 84.8%
# Improvement: +___% (absolute)
# Average time: ___s per sample
# Total cost: $___
```

**Required Documentation:**
```markdown
## Phase 5 Benchmark Results

### Overall Performance
- **Accuracy**: ___% (baseline: 84.8%, target: 88-92%)
- **Samples Processed**: 15/15
- **Average Time**: ___s per sample (baseline: 45s)
- **Total Cost**: $___ (baseline: $0.407)
- **Status**: ✅ Production Ready / ⚠️ Needs Optimization / ❌ Needs Investigation

### Accuracy Breakdown by Sample
| Sample ID | Baseline | With Validation | Change | Status |
|-----------|----------|----------------|---------|---------|
| 1 | __% | __% | +__% | ✅/⚠️/❌ |
| ... | ... | ... | ... | ... |
| 20 | 21.1% | __% | +__% | ✅/⚠️/❌ |

### Error Detection Analysis
| Error Type | Detected | Corrected | Success Rate |
|------------|----------|-----------|--------------|
| digit_mismatch | __ | __ | __% |
| spelling | __ | __ | __% |
| missing_row | __ | __ | __% |
| value_swap | __ | __ | __% |
| **Total** | __ | __ | __% |

### Cost & Performance Analysis
- Validator API calls: ___
- Corrector API calls: ___
- Average latency added: ___s per sample
- Cost per sample: $___
- ROI: Accuracy gain ___% / Cost increase ___%

### Recommendations
- ✅ Ready for production (if targets met)
- ⚠️ Needs tuning (if close to targets)
- ❌ Requires investigation (if targets missed)
```

**Decision Criteria:**
- If accuracy ≥88%: **Deploy to production** ✅
- If accuracy 85-88%: **Optimize and re-test** ⚠️
- If accuracy <85%: **Root cause analysis required** ❌

---

## Implementation Checklist

### Phase 1: Validator
- [ ] Create `src/ocr_pipeline/latex_validator.py`
- [ ] Implement `validate_latex_table()`
- [ ] Implement `parse_validation_response()`
- [ ] Write validation prompt template
- [ ] Test validator on sample ID=20 manually

### Phase 2: Corrector
- [ ] Create `src/ocr_pipeline/latex_corrector.py`
- [ ] Implement `correct_latex_table()`
- [ ] Implement `apply_correction()`
- [ ] Implement `decide_correction()`
- [ ] Test corrector on sample ID=20 manually

### Phase 3: Integration
- [ ] Add helper functions to extract/replace LaTeX
- [ ] Update `src/ocr_pipeline/qwen_extractor.py`
- [ ] Add `validate_tables` parameter
- [ ] Add `correction_strategy` parameter
- [ ] Update `src/ocr_pipeline/__init__.py` exports

### Phase 4: Test Worst Samples
- [ ] Update `scripts/test_worst_samples.py`
- [ ] Run on sample ID=4 and ID=20
- [ ] Verify validator detects errors
- [ ] Verify corrector improves accuracy
- [ ] Document results

### Phase 5: Full Benchmark
- [ ] Update `scripts/test_tables.py`
- [ ] Run on all 15 samples
- [ ] Measure accuracy improvement
- [ ] Analyze cost/latency
- [ ] Document final results

### Documentation
- [ ] Update README with validation feature
- [ ] Add usage examples
- [ ] Document configuration options
- [ ] Create performance benchmarks doc

---

## API Design

### Clean Separation Principle

**Good (Separate parameters):**
```python
validate_latex_table(
    table_image_base64="iVBORw0KGgo...",  # Image data
    latex_table="\\begin{tabular}...",     # Text data
    prompt=VALIDATION_PROMPT_TEMPLATE      # Instructions
)
```

**Bad (Mixed):**
```python
prompt = f"Compare LaTeX with image: data:image/png;base64,{huge_base64_string}..."
```

### OpenRouter API Call Structure

```python
payload = {
    "model": "google/gemini-2.0-flash",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt.format(latex_table=latex_table)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{table_image_base64}"
                    }
                }
            ]
        }
    ],
    "temperature": 0
}
```

---

## Error Handling

### Validator Errors
- Gemini API timeout → Return `{"valid": True, "errors": []}`
- Invalid JSON response → Retry once, then skip validation
- Image too large → Resize and retry

### Corrector Errors
- Correction fails → Keep original LaTeX
- Low confidence → Skip correction, log for review
- Structural changes fail → Keep original LaTeX

---

## Cost & Performance Analysis

### Baseline (No Validation)
- QwenVL: ~$0 (free tier)
- Claude: ~$0.027 per sample
- Total: ~$0.407 for 15 samples
- Time: ~45s per sample

### With Validation (Estimated)
- QwenVL: ~$0
- Gemini validator: ~$0.001-0.003 per table
- Corrector (if needed): ~$0.001-0.005 per table
- Claude: ~$0.027 per sample
- Total: ~$0.45-0.50 for 15 samples (+10-20%)
- Time: ~50-60s per sample (+10-30%)

**Tradeoff:**
- Cost: +10-20%
- Time: +10-30%
- Accuracy: +5-8% (expected)
- **ROI: Worth it if accuracy improvement materializes**

---

## Success Criteria

### Must Have
- ✅ Validator detects errors in worst samples (ID=4, ID=20)
- ✅ Corrector improves accuracy on those samples
- ✅ No breaking changes to existing API
- ✅ Production-ready code (no benchmark dependencies in src/)

### Should Have
- ✅ Overall accuracy improves from 84.8% baseline
- ✅ Validation adds <15s per sample
- ✅ Cost increase <25%
- ✅ Configurable validation strategies

### Nice to Have
- ✅ Validation report exposed to users (optional)
- ✅ Correction transparency (log what was changed)
- ✅ Metrics dashboard (errors by type)
- ✅ Auto-fallback on validation failure

---

## Future Enhancements

### Short Term
- Add validation confidence threshold configuration
- Support for partial table validation (specific cells)
- Validation caching (don't re-validate identical tables)

### Long Term
- Multi-model consensus (QwenVL + Gemini + GPT-5)
- Active learning (learn from corrections)
- Human-in-the-loop for low confidence cases
- Validation for other element types (charts, signatures)

---

## References

- OpenRouter Gemini 2.0 Flash: https://openrouter.ai/google/gemini-2.0-flash
- QwenVL Documentation: (internal)
- Benchmark Methodology: benchmark/evaluator.py
- Current Baseline Results: 84.8% (14/15 samples successful)
