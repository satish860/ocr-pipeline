# Table Validation & Correction Bugs

**Status:** Critical - Validation decreases accuracy instead of improving it
**Discovered:** 2025-11-16
**Affects:** table_validator.py, table_corrector.py
**Test Results:** Sample ID=4 accuracy dropped from 89.8% → 2.7% (-87.1%)

---

## Executive Summary

The table validation and correction pipeline has **critical bugs** that cause it to:
1. **Destroy table structure** during LaTeX parsing/rebuilding
2. **Generate false positive errors** where values are identical
3. **Apply harmful corrections** that remove important data
4. **Decrease overall accuracy** instead of improving it

**Impact:** Validation should be **disabled in production** until these issues are fixed.

---

## Bug #1: LaTeX Parsing Destroys Table Structure

### Severity: CRITICAL

### Description
The LaTeX table parsing in `table_corrector.py` removes `\hline` markers and fails to restore them correctly, resulting in complete destruction of the table structure.

### Evidence

**Sample ID=4 - Before Correction:**
```latex
\begin{tabular}{llrrrrr}
\hline
AO & Total & Studies & 25\% & 50\% & 75\% & Max \\
\hline
Data type & Dichotomous & 77237 & 50 & 102 & 243 & 1242071 \\
 & Continuous & 29902 & 33 & 62 & 162 & 18851 \\
 & Mixed & 5461 & 21 & 86 & 251 & 36511 \\
```

**After Correction (BROKEN):**
```latex
\begin{tabular}{llrrrrr}
& Continuous & 29902 & 33 & 62 & 162 & 18851 \\
& Mixed & 5461 & 21 & 86 & 251 & 36511 \\
& Cardiovascular & 77237 & 44 & 102 & 243 & 67880 \\
```

**Issues:**
- ❌ Header row (`AO & Total & Studies & 25\% & 50\% & 75\% & Max`) is **COMPLETELY REMOVED**
- ❌ `\hline` separators are missing
- ❌ First data row (`Data type & Dichotomous & ...`) is missing
- ❌ Table structure is completely mangled

### Root Cause

**File:** `src/ocr_pipeline/table_corrector.py`
**Function:** `parse_latex_rows()` (lines 19-57)

```python
def parse_latex_rows(latex: str) -> Tuple[str, List[str], str]:
    # ...
    # Split by \\ to get rows, filter out \hline and empty rows
    raw_rows = content.split(r'\\')
    rows = []
    for row in raw_rows:
        row = row.strip()
        # Skip \hline and empty rows  <-- BUG: Removes \hline permanently
        if row and not row.startswith(r'\hline'):
            rows.append(row)

    return header, rows, footer
```

**Problems:**
1. **Removes `\hline` without storing them** - can't restore separator positions
2. **No tracking of header row position** - loses the distinction between header and data rows
3. **Simple `split('\\\\')` doesn't handle escaped backslashes correctly**
4. **String replacement in `set_cell_value()` can match wrong rows** if values appear multiple times

### Impact on Accuracy

**Sample ID=4:**
- Baseline: 89.8% accuracy (19/186 errors)
- With Validation: **2.7% accuracy** (181/186 errors)
- **Accuracy drop: -87.1%**

The destroyed table structure makes it impossible for Claude to extract correct JSON, causing massive accuracy regression.

---

## Bug #2: Validator Reports False Positive Errors

### Severity: HIGH

### Description
The Gemini validator reports "errors" where the LaTeX value and image value are **identical**, leading to unnecessary and harmful corrections.

### Evidence

**Sample ID=4 - False Positive Errors:**

```
[1] Type: digit_mismatch
    Location: Row 2, Col 2
    LaTeX value: '77237'
    Image value: '77237'    ← IDENTICAL!
    Confidence: 0.90

[2] Type: digit_mismatch
    Location: Row 2, Col 5
    LaTeX value: '243'
    Image value: '243'      ← IDENTICAL!
    Confidence: 0.90

[10] Type: digit_mismatch
    Location: Row 23, Col 5
    LaTeX value: '242'
    Image value: '242'      ← IDENTICAL!
    Confidence: 0.90

[12] Type: digit_mismatch
    Location: Row 28, Col 6
    LaTeX value: '259627'
    Image value: '259627'   ← IDENTICAL!
    Confidence: 0.90
```

**Out of 15 errors detected, at least 4 were false positives (26.7% false positive rate).**

### Corrections Applied on False Positives

```
Applied corrections:
  [1] digit_mismatch at row 2, col 2
      '77237' -> '77237'           ← No actual change!
      Method: direct_replacement

  [2] digit_mismatch at row 2, col 5
      '243' -> '243'               ← No actual change!
      Method: direct_replacement
```

These "corrections" still trigger the LaTeX table rebuild, which destroys the structure.

### Root Cause

**File:** `src/ocr_pipeline/table_validator.py`
**Function:** `validate_table()` via Gemini 2.5 Flash API

**Possible causes:**
1. **Gemini vision model hallucination** - seeing differences that don't exist
2. **Prompt ambiguity** - model not understanding the task correctly
3. **OCR confidence misinterpretation** - reporting low confidence as errors
4. **Image quality issues** - model uncertain about blurry digits

### Missing Safeguard

**File:** `src/ocr_pipeline/table_corrector.py`
**Function:** `correct_table()` (lines 505-570)

**Current code does NOT filter false positives:**
```python
for error in errors:
    # No check if latex_value == image_value!
    correction_func = CORRECTION_FUNCTIONS.get(error_type)
    corrected_latex, log = correction_func(...)  # Applies correction blindly
```

**Should add:**
```python
for error in errors:
    # Filter false positives
    if error['latex_value'] == error['image_value']:
        corrections_skipped.append({
            "error_type": error_type,
            "reason": "false_positive_identical_values"
        })
        continue
    # ... rest of correction logic
```

---

## Bug #3: Validator Suggests Harmful Corrections

### Severity: HIGH

### Description
The validator suggests corrections that **remove important data** or change correct values to incorrect ones.

### Evidence

**Sample ID=20 - Harmful Header Corrections:**

```
[1] Type: spelling
    Location: Row 0, Col 1
    LaTeX value: 'IND/LOR VOLUME'
    Image value: 'VOLUME'           ← Removes "IND/LOR" prefix!
    Confidence: 1.00

[2] Type: spelling
    Location: Row 0, Col 2
    LaTeX value: 'NO. OF STORES'
    Image value: 'STORES'            ← Removes "NO. OF" prefix!
    Confidence: 1.00
```

**Applied corrections:**
```
[1] spelling at row 0, col 1
    'IND/LOR VOLUME' -> 'VOLUME'    ← LOSES DATA!

[2] spelling at row 0, col 2
    'NO. OF STORES' -> 'STORES'     ← LOSES DATA!
```

### Impact

These corrections **remove semantic information** from the table:
- "IND/LOR VOLUME" is more descriptive than just "VOLUME"
- "NO. OF STORES" is more clear than just "STORES"

The validator is interpreting **abbreviated headers** as errors when they are actually **more complete than what's in the image**.

### Root Cause

The validation prompt tells Gemini to compare "LaTeX value with image value character-by-character", but doesn't account for:
1. **OCR extracting full text** while image shows abbreviated text
2. **Headers that wrap across lines** in the image but are single-line in LaTeX
3. **Abbreviations being expanded** by the OCR system

**File:** `src/ocr_pipeline/table_validator.py` (lines 108-149)

The prompt should be updated to:
- Prefer LaTeX values if they contain more information
- Only flag as errors if LaTeX is clearly wrong, not just different
- Understand that "IND/LOR VOLUME" is better than "VOLUME"

---

## Bug #4: Wrong Corrections Applied

### Severity: MEDIUM

### Description
Some corrections change correct values to incorrect ones based on validator hallucinations.

### Evidence

**Sample ID=4:**

```
[11] Type: digit_mismatch
     Location: Row 26, Col 6
     LaTeX value: '133271'
     Image value: '131271'          ← Changed 133271 → 131271
     Confidence: 0.90

[14] Type: digit_mismatch
     Location: Row 31, Col 6
     LaTeX value: '133271'
     Image value: '131271'          ← Same "correction" applied
     Confidence: 0.90
```

**Question:** Is the LaTeX actually wrong? Or is the validator hallucinating?

Without manual verification against the ground truth, we can't confirm if these are:
- ✅ Correct fixes (OCR misread 131271 as 133271)
- ❌ Wrong corrections (LaTeX was right, validator hallucinated)

### Root Cause

**No validation of validator results** - we blindly trust Gemini's output without:
1. Checking if corrections improve or worsen accuracy
2. Comparing against ground truth (when available)
3. Requiring minimum confidence thresholds
4. Cross-validating with multiple models

---

## Bug #5: Extra Row Corrections Fail

### Severity: MEDIUM

### Description
The corrector tries to remove "extra rows" but fails due to invalid indices.

### Evidence

**Sample ID=20 - Table 1:**

```
[6] Type: extra_row
    Location: Row 7, Col 0
    Confidence: 1.00

[7] Type: extra_row
    Location: Row 8, Col 0
    Confidence: 1.00
```

**Skipped corrections:**
```
[1] extra_row - skipped_invalid_index
[2] extra_row - skipped_invalid_index
```

### Root Cause

**File:** `src/ocr_pipeline/table_corrector.py`
**Function:** `fix_extra_row()` (lines 202-230)

```python
def fix_extra_row(latex: str, error: Dict, image_base64: str) -> Tuple[str, Dict]:
    header, rows, footer = parse_latex_rows(latex)
    row_idx = error['location']['row']

    if 0 <= row_idx < len(rows):  # Index out of range check
        removed_row = rows.pop(row_idx)
        # ...
    else:
        # Invalid row index, return unchanged  ← FAILS HERE
        log = {..., "method": "skipped_invalid_index"}
        return latex, log
```

**Why indices are invalid:**
1. Validator reports row indices from the **full table** (including header)
2. `parse_latex_rows()` filters out `\hline` and header rows
3. Row indices no longer match after filtering
4. Result: Can't locate the rows to delete

---

## Test Results Summary

### Sample ID=4 (Complex Table)

| Metric | Baseline | With Validation | Change |
|--------|----------|-----------------|--------|
| Accuracy | **89.8%** | 2.7% | **-87.1%** ⚠️ |
| Errors | 19/186 | 181/186 | +162 errors |
| Status | Good | **CATASTROPHIC FAILURE** |

**Conclusion:** Validation completely destroys this sample.

---

### Sample ID=20 (Missing Rows Problem)

| Metric | Baseline | With Validation | Change |
|--------|----------|-----------------|--------|
| Accuracy | 21.1% | 20.3% | **-0.8%** ⚠️ |
| Errors | 45/57 | 55/69 | +10 errors |
| Status | Poor | **Slightly worse** |

**Conclusion:** Validation makes a bad situation worse.

---

## Root Causes Analysis

### 1. LaTeX Parsing Architecture Issue

**Current approach:**
```
LaTeX string → parse_latex_rows() → List[str] → modify → rebuild_latex_table() → LaTeX string
```

**Problems:**
- Loses structure information (header row, `\hline` positions)
- String-based replacement is fragile
- No AST or proper parsing

**Better approach:**
```
LaTeX string → Parse to AST → Modify AST → Serialize to LaTeX
```

Or use existing LaTeX parsing libraries like `TexSoup` or `pylatexenc`.

---

### 2. Validator Unreliability

**Issues:**
- Gemini 2.5 Flash vision model has hallucinations
- 26.7% false positive rate on Sample ID=4
- No confidence calibration
- Prompt may not be specific enough

**Mitigations needed:**
- Filter false positives (old_value == new_value)
- Require higher confidence thresholds (>0.95 for conservative)
- Multi-model consensus (use 2-3 models, take majority vote)
- Ground truth validation when available

---

### 3. No Quality Assurance on Corrections

**Current workflow:**
```
Validate → Detect errors → Apply corrections → Hope for the best
```

**Missing checks:**
- ❌ No verification that corrections improved accuracy
- ❌ No rollback if corrections make things worse
- ❌ No comparison with ground truth
- ❌ No confidence scoring of corrections

**Should be:**
```
Validate → Detect errors → Apply corrections → Verify accuracy → Rollback if worse
```

---

## Recommended Fixes

### Priority 1: Stop Using in Production

**Immediate action:**
- Set `validate_tables=False` as default ✅ (Already done)
- Document known issues in README
- Add warning in docstring

### Priority 2: Fix LaTeX Parsing

**File:** `src/ocr_pipeline/table_corrector.py`

**Options:**

**Option A: Use existing LaTeX parser**
```python
from TexSoup import TexSoup

def parse_latex_table(latex: str):
    soup = TexSoup(latex)
    table = soup.find('tabular')
    # Preserves all structure, \hline, etc.
    return table

def modify_cell(table, row, col, new_value):
    table.rows[row].cells[col].value = new_value
    return str(table)  # Serializes back to LaTeX with all structure preserved
```

**Option B: Improve current parser**
```python
def parse_latex_rows(latex: str) -> Tuple[str, List[str], List[int], str]:
    # ... existing code ...

    # NEW: Track \hline positions
    hline_positions = []
    for i, row in enumerate(raw_rows):
        if row.strip().startswith(r'\hline'):
            hline_positions.append(i)

    # NEW: Track header row
    header_row_index = 0  # First non-hline row

    return header, rows, hline_positions, footer

def rebuild_latex_table(header, rows, hline_positions, footer):
    # Restore \hline at original positions
    result = header + "\n"
    for i, row in enumerate(rows):
        if i in hline_positions:
            result += r"\hline" + "\n"
        result += row + r" \\" + "\n"
    result += footer
    return result
```

### Priority 3: Add False Positive Filter

**File:** `src/ocr_pipeline/table_corrector.py`
**Function:** `correct_table()` (line ~520)

```python
for error in errors:
    # NEW: Filter false positives
    if error.get('latex_value') == error.get('image_value'):
        corrections_skipped.append({
            "error_type": error_type,
            "location": error.get('location', {}),
            "reason": "false_positive_identical_values"
        })
        continue

    # Existing correction logic...
```

### Priority 4: Improve Validator Prompt

**File:** `src/ocr_pipeline/table_validator.py` (lines 108-149)

**Add to prompt:**
```
CRITICAL RULES (UPDATED):
- Only report errors where you are CERTAIN the LaTeX is wrong
- If LaTeX value contains MORE information than image (e.g., "IND/LOR VOLUME" vs "VOLUME"),
  DO NOT report as error - the LaTeX is better
- If values are identical, DO NOT report as error
- Require 95% confidence minimum for digit_mismatch errors
- Double-check all digit comparisons before reporting
```

### Priority 5: Add Quality Gates

**New function in `table_corrector.py`:**

```python
def verify_correction_quality(
    original_latex: str,
    corrected_latex: str,
    ground_truth: Optional[Dict] = None
) -> Dict:
    """
    Verify that corrections improved quality.

    Returns:
        {
            "improved": bool,
            "original_score": float,
            "corrected_score": float,
            "recommendation": "apply" | "rollback"
        }
    """
    # If ground truth available, compare both versions
    if ground_truth:
        original_accuracy = evaluate_against_ground_truth(original_latex, ground_truth)
        corrected_accuracy = evaluate_against_ground_truth(corrected_latex, ground_truth)

        return {
            "improved": corrected_accuracy > original_accuracy,
            "original_score": original_accuracy,
            "corrected_score": corrected_accuracy,
            "recommendation": "apply" if corrected_accuracy > original_accuracy else "rollback"
        }

    # Without ground truth, use heuristics
    # ...
```

---

## Testing Recommendations

### Test Coverage Needed

1. **Unit tests for LaTeX parsing:**
   ```python
   def test_parse_latex_preserves_structure():
       latex = r"\begin{tabular}{ll} \hline A & B \\ \hline C & D \\ \end{tabular}"
       header, rows, footer = parse_latex_rows(latex)
       reconstructed = rebuild_latex_table(header, rows, footer)
       assert reconstructed == latex  # Should be identical
   ```

2. **Integration test with ground truth:**
   ```python
   def test_validation_improves_accuracy():
       for sample in test_samples:
           baseline_acc = evaluate(extract_without_validation(sample))
           validated_acc = evaluate(extract_with_validation(sample))
           assert validated_acc >= baseline_acc  # Should never decrease
   ```

3. **False positive detection:**
   ```python
   def test_no_false_positives():
       result = validate_table(image, correct_latex)
       for error in result['errors']:
           assert error['latex_value'] != error['image_value']
   ```

---

## Estimated Fix Effort

| Priority | Task | Estimated Effort | Risk |
|----------|------|------------------|------|
| P1 | Disable in production | 1 hour | Low |
| P2 | Fix LaTeX parsing (Option A) | 8 hours | Medium |
| P2 | Fix LaTeX parsing (Option B) | 16 hours | High |
| P3 | Add false positive filter | 2 hours | Low |
| P4 | Improve validator prompt | 4 hours | Medium |
| P5 | Add quality gates | 8 hours | Medium |
| - | **Total (Option A)** | **23 hours** (~3 days) | - |
| - | **Total (Option B)** | **31 hours** (~4 days) | - |

**Recommendation:** Use Option A (existing parser) for faster, lower-risk fix.

---

## Alternative Approach: Abandon Current Implementation

Given the severity of issues, consider **alternative architectures**:

### Option 1: Post-Extraction Validation Only
```
QwenVL → Extract LaTeX → Claude JSON extraction → Validate JSON → Fix JSON
```
- Skip LaTeX correction entirely
- Validate final JSON output instead
- Lower risk of breaking working tables

### Option 2: Multi-Model Consensus
```
QwenVL → LaTeX Table A
Gemini → LaTeX Table B
GPT-4V → LaTeX Table C
→ Compare & merge best version
```
- No correction phase
- Pick best extraction from multiple models
- Higher cost but more reliable

### Option 3: Confidence-Based Routing
```
QwenVL → Extract with confidence scores
If confidence < 0.8:
    → Re-extract with different model
Else:
    → Use as-is
```
- Only validate low-confidence extractions
- Avoid breaking high-quality tables

---

## Conclusion

The table validation and correction pipeline has **critical bugs** that make it **unsuitable for production use**. The most severe issue is the LaTeX parsing destroying table structure, causing **87% accuracy regression** on Sample ID=4.

**Immediate action required:**
1. ✅ Keep validation disabled by default (already done)
2. Add warning documentation
3. Evaluate alternative approaches before investing in fixes

**Long-term options:**
1. Fix the bugs (23-31 hours estimated)
2. Pivot to alternative architecture
3. Shelve validation feature until better solution found

The commit has been pushed with a note about these issues. Further work should not proceed until architectural decision is made.
