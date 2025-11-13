# Phase 1 vs Baseline Comparison

## Summary

**Baseline (Gemini Flash 2.5, Original Prompt)**
- Average Accuracy: **84.41%**
- Total Samples: 6
- Model: google/gemini-2.5-flash (OCR) + anthropic/claude-haiku-4.5 (JSON)

**Phase 1 (Enhanced JSON Extraction Prompt)**
- Average Accuracy: **78.48%** ⚠️ DOWN by 5.93%
- Total Samples: 6
- Model: Same as baseline
- Change: Enhanced JSON extraction prompt with cross-section analysis instructions

## Detailed Comparison

| Sample | Baseline | Phase 1 | Change | Status |
|--------|----------|---------|--------|--------|
| sample_1_TABLE | 100% | 100% | 0% | ✅ No change |
| sample_2_CHART | 74.00% | 71.23% | -2.77% | ⚠️ Worse |
| sample_3_CHART | 77.78% | 70.37% | -7.41% | ⚠️ Worse |
| sample_4_TABLE | 80.46% | 57.47% | -22.99% | ❌ Significantly worse |
| sample_5_TABLE | 89.60% | 90.35% | +0.75% | ✅ Slightly better |
| sample_6_TABLE | 84.29% | 81.43% | -2.86% | ⚠️ Worse |

## Key Findings

### ❌ Phase 1 Failed
1. **Overall regression**: Average accuracy dropped from 84.41% to 78.48%
2. **sample_4_TABLE**: Major regression (-23%), suggesting enhanced prompt confused the model on complex tables
3. **sample_2_CHART**: Still missing parent segments (71.23% vs 74% baseline)
4. **Most samples got worse**: 4 out of 6 samples regressed

### 🔍 Root Cause Analysis
The enhanced prompt with "cross-section analysis" instructions appears to have:
- **Confused the model** on complex tables (sample_4_TABLE)
- **Not solved the core problem**: Parent segments still missing in sample_2_CHART
- **Added complexity** without providing actionable spatial data

### 💡 Conclusion
**Prompt engineering alone is insufficient.** The JSON extractor needs actual spatial relationship data, not just better instructions. The model cannot "look for adjacent sections" when it only receives flat markdown text.

## Next Steps

### Phase 2: Pre-merge Related Regions (Required)
Instead of asking the JSON extractor to analyze spatial relationships, we should:

1. **Before JSON extraction**: Use `spatial_analyzer.analyze_relationships()` to identify related regions
2. **Merge regions**: Physically combine chart regions with nearby paragraph regions (within ~100px)
3. **Pass enhanced markdown**: JSON extractor receives pre-merged content, no spatial analysis needed

### Implementation Location
- File: `test_batch_samples.py` lines 93-153
- After OCR markdown generation, before JSON extraction
- Use existing `spatial_analyzer` to find nearby regions
- Merge related regions into single markdown sections

### Expected Improvement
- Parent segments will be included in same section as chart table
- JSON extractor sees combined content, no analysis needed
- Should restore 84.41% baseline and potentially improve sample_2_CHART to ~90%+

## Test Results Detail

### Baseline Results (TEST_RESULTS_SUMMARY.md)
```
Total samples: 6
Successful: 6
Failed: 0
Average accuracy: 84.41%
Total fields: 796
Total differences: 124

Chart Samples:
- sample_2_CHART: 74.00%
- sample_3_CHART: 77.78%

Table Samples:
- sample_1_TABLE: 100%
- sample_4_TABLE: 80.46%
- sample_5_TABLE: 89.60%
- sample_6_TABLE: 84.29%
```

### Phase 1 Results (batch_test_results.json)
```
Total samples: 6
Successful: 6
Failed: 0
Average accuracy: 78.48%
Total fields: 797
Total differences: 128

Results by sample:
- sample_1_TABLE: 100% (0 differences)
- sample_2_CHART: 71.23% (21 differences)
- sample_3_CHART: 70.37% (8 differences)
- sample_4_TABLE: 57.47% (37 differences) ⚠️
- sample_5_TABLE: 90.35% (36 differences)
- sample_6_TABLE: 81.43% (26 differences)
```

## Decision

**Revert Phase 1 changes or keep?**
- ❌ **Revert**: Baseline was better overall
- ✅ **Keep for now**: Document attempt, proceed to Phase 2 which should fix the core issue

**Recommended**: Keep Phase 1 changes documented in git history, but proceed immediately to Phase 2 which addresses the root cause.
