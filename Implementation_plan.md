# Implementation Plan: Agentic Claude Refinement Loop

## Overview
Implement an iterative refinement loop where Claude Sonnet 4.5 progressively refines OCR output by reviewing its previous corrections, using a **Markdown-first approach** with **full document replacement**.

## Architecture Choice

### Format Strategy: **Option 1 - Markdown + HTML Tables**
- Keep QwenVL output in Markdown format
- Convert only LaTeX tables → HTML tables
- Preserve Markdown for all other content
- Output: Clean Markdown with embedded HTML tables

### Refinement Strategy: **Full Document Replacement (Approach 3 - Hybrid)**
- Claude returns complete refined Markdown document each iteration
- Claude provides structured change metadata for transparency
- No complex patching logic - simple replacement
- Change metadata used for logging, convergence detection, user transparency

---

## Implementation Phases

### Phase 1: Core Agentic Loop (Foundation)
**Goal:** Implement basic iterative refinement loop

**Tasks:**
1. Create `RefinementHistory` class to track iterations
2. Implement `refine_with_agentic_loop()` function
3. Add iteration prompt builder with correction history
4. Implement convergence detection (no changes = stop)
5. Add max iterations safety limit (default: 3)

**Files to modify:**
- `src/ocr_pipeline/claude_refiner.py` - Add agentic loop logic
- Keep existing `refine_with_claude()` for single-pass (backward compatibility)

**Success Criteria:**
- ✅ Loop executes up to max_iterations
- ✅ Each iteration receives previous refinement + changes
- ✅ Stops early when converged (no changes between iterations)
- ✅ Returns refinement history with all iterations tracked

---

### Phase 2: Markdown-First Format Handling
**Goal:** Ensure Markdown format is preserved through refinement

**Tasks:**
1. Update Claude system prompt to specify Markdown output
2. Modify `refine_with_claude()` to NOT convert Markdown→HTML
3. Keep TableConverter for LaTeX→HTML table conversion only
4. Add format validation (ensure output is valid Markdown)

**Files to modify:**
- `src/ocr_pipeline/claude_refiner.py` - Update prompts for Markdown
- `src/ocr_pipeline/table_converter.py` - Ensure it only touches tables
- `src/ocr_pipeline/ocr_pipeline.py` - Update orchestration flow

**Success Criteria:**
- ✅ QwenVL Markdown preserved (headers, paragraphs, bold, alignment divs)
- ✅ Only tables converted to HTML
- ✅ Claude outputs Markdown (not full HTML)
- ✅ Final output is valid Markdown with embedded HTML tables

---

### Phase 3: Structured Change Tracking
**Goal:** Track what Claude changes in each iteration

**Tasks:**
1. Design change metadata JSON schema
2. Update Claude prompt to return structured changes
3. Parse and validate change metadata from Claude response
4. Store changes in RefinementHistory
5. Display changes to user in readable format

**Change Metadata Schema:**
```json
{
  "refined_markdown": "...",
  "changes": [
    {
      "iteration": 2,
      "element_type": "table|paragraph|header|signature|handwritten",
      "location": "table row 3, cell 2",
      "change_type": "correction|addition|deletion|restructure",
      "old_value": "12",
      "new_value": "123",
      "confidence": 0.90,
      "reason": "Missing digit visible in image"
    }
  ],
  "overall_confidence": 0.92,
  "note": "Optional human-readable summary"
}
```

**Files to modify:**
- `src/ocr_pipeline/claude_refiner.py` - Add change parsing logic

**Success Criteria:**
- ✅ Claude returns structured JSON with changes
- ✅ Changes parsed and validated correctly
- ✅ Change history tracked across iterations
- ✅ User can see what was changed and why

---

### Phase 4: Smart Stopping Criteria
**Goal:** Stop iterations intelligently (don't waste API calls)

**Tasks:**
1. Implement convergence detection (string comparison)
2. Add confidence threshold stopping (e.g., stop at 95%)
3. Add no-changes detection from metadata
4. Implement safety limits (max iterations)

**Stopping Logic:**
```python
# Stop if:
1. refined_markdown == previous_markdown (converged)
2. len(changes) == 0 (no changes)
3. confidence >= confidence_threshold (high confidence)
4. iteration >= max_iterations (safety limit)
```

**Files to modify:**
- `src/ocr_pipeline/claude_refiner.py` - Add stopping logic

**Success Criteria:**
- ✅ Stops when no changes detected
- ✅ Stops when confidence threshold reached
- ✅ Respects max_iterations limit
- ✅ Logs reason for stopping

---

### Phase 5: Integration with OCRPipeline
**Goal:** Add agentic refinement as optional feature

**Tasks:**
1. Add parameters to OCRPipeline:
   - `agentic_refine: bool = False` (opt-in)
   - `max_refinement_iterations: int = 3`
   - `confidence_threshold: float = 90.0`
2. Update orchestration to use agentic loop when enabled
3. Keep backward compatibility (refine=True uses single-pass)
4. Update CLI to support agentic refinement flag

**Files to modify:**
- `src/ocr_pipeline/ocr_pipeline.py` - Add agentic_refine parameter
- `src/ocr_pipeline/cli.py` - Add --agentic-refine flag

**Success Criteria:**
- ✅ `OCRPipeline(refine=True)` uses single-pass (backward compatible)
- ✅ `OCRPipeline(agentic_refine=True)` uses iterative loop
- ✅ CLI supports both modes
- ✅ Parameters configurable by user

---

### Phase 6: Cost & Latency Monitoring
**Goal:** Track cost and performance of agentic refinement

**Tasks:**
1. Add usage tracking per iteration
2. Calculate total cost (sum of all iterations)
3. Track total latency (time for all iterations)
4. Add warnings for high cost scenarios
5. Log cost/latency metrics in result

**Metrics to Track:**
- Iterations executed
- Total tokens used (input + output)
- Total cost ($)
- Total time (seconds)
- Cost per iteration
- Average iteration time

**Files to modify:**
- `src/ocr_pipeline/claude_refiner.py` - Add cost/latency tracking

**Success Criteria:**
- ✅ Usage data collected per iteration
- ✅ Total cost calculated and returned
- ✅ Latency measured and logged
- ✅ User sees cost breakdown in result

---

## Technical Design

### RefinementHistory Class
```python
class RefinementHistory:
    """Tracks refinement iterations and changes."""

    def __init__(self, original_markdown: str):
        self.original_markdown = original_markdown
        self.iterations: List[RefinementIteration] = []

    def add_iteration(
        self,
        refined_markdown: str,
        changes: List[Dict],
        confidence: float,
        usage: Dict
    ):
        """Add a refinement iteration."""
        self.iterations.append({
            'iteration': len(self.iterations) + 1,
            'refined_markdown': refined_markdown,
            'changes': changes,
            'confidence': confidence,
            'usage': usage,
            'timestamp': datetime.now()
        })

    def get_latest_refinement(self) -> str:
        """Get most recent refined markdown."""
        return self.iterations[-1]['refined_markdown'] if self.iterations else self.original_markdown

    def has_converged(self) -> bool:
        """Check if last two iterations are identical."""
        if len(self.iterations) < 2:
            return False
        last = self.iterations[-1]['refined_markdown']
        prev = self.iterations[-2]['refined_markdown']
        return last.strip() == prev.strip()

    def total_cost(self) -> float:
        """Calculate total API cost across iterations."""
        return sum(it['usage'].get('total_cost', 0) for it in self.iterations)

    def to_dict(self) -> Dict:
        """Export history as dictionary."""
        return {
            'total_iterations': len(self.iterations),
            'converged': self.has_converged(),
            'final_confidence': self.iterations[-1]['confidence'] if self.iterations else 0,
            'total_cost': self.total_cost(),
            'iterations': self.iterations
        }
```

### Main Refinement Loop
```python
def refine_with_agentic_loop(
    image_input,
    qwen_result: Dict,
    max_iterations: int = 3,
    confidence_threshold: float = 90.0,
    include_usage: bool = False
) -> Dict:
    """Iterative refinement loop with correction history."""

    # Initialize history
    history = RefinementHistory(qwen_result['markdown'])

    # Iteration loop
    for iteration in range(1, max_iterations + 1):
        print(f"Refinement iteration {iteration}/{max_iterations}...")

        # Build prompt with full history
        prompt = build_iteration_prompt(
            iteration=iteration,
            original=history.original_markdown,
            previous_iterations=history.iterations,
            image=image_input
        )

        # Call Claude with history
        response = call_claude_refinement_api(
            image_input,
            prompt,
            include_usage=include_usage
        )

        # Parse response
        refined_markdown = response['refined_markdown']
        changes = response['changes']
        confidence = response['confidence']
        usage = response.get('usage', {})

        # Track iteration
        history.add_iteration(refined_markdown, changes, confidence, usage)

        # Check stopping criteria
        if history.has_converged():
            print(f"✓ Converged at iteration {iteration} (no changes)")
            break

        if len(changes) == 0:
            print(f"✓ No changes needed at iteration {iteration}")
            break

        if confidence >= confidence_threshold:
            print(f"✓ High confidence reached: {confidence}%")
            break

        print(f"  Made {len(changes)} changes, confidence: {confidence}%")

    # Return final result
    return {
        'success': True,
        'markdown': history.get_latest_refinement(),
        'images': qwen_result.get('images', []),
        'elements': qwen_result.get('elements', []),
        'refinement': history.to_dict()
    }
```

### Prompt Builder
```python
def build_iteration_prompt(
    iteration: int,
    original: str,
    previous_iterations: List[Dict],
    image
) -> str:
    """Build prompt with correction history."""

    if iteration == 1:
        # First iteration: simple refinement
        return f"""Refine this OCR-extracted Markdown document.

**Original Markdown:**
{original}

**Instructions:**
- Fix clear OCR errors (misread text, wrong table values, missing content)
- Keep output in MARKDOWN format (headers as #, bold as **, etc.)
- Tables should be HTML: <table>...</table>
- Preserve alignment: <div align="right">...</div>
- Preserve coordinate annotations: <!-- Type (x,y,w,h) -->

**Return JSON:**
{{
  "refined_markdown": "...",
  "changes": [
    {{
      "element_type": "table",
      "location": "table row 3, cell 2",
      "change_type": "correction",
      "old_value": "12",
      "new_value": "123",
      "confidence": 0.90,
      "reason": "Missing digit visible in image"
    }}
  ],
  "overall_confidence": 0.90
}}
"""

    else:
        # Subsequent iterations: show history
        prev = previous_iterations[-1]

        changes_summary = "\n".join([
            f"{i+1}. {c['element_type']} at {c['location']}: '{c['old_value']}' → '{c['new_value']}' ({c['reason']})"
            for i, c in enumerate(prev['changes'])
        ])

        return f"""This is refinement iteration {iteration}.

**Original Markdown:**
{original}

**Your Previous Refinement (Iteration {prev['iteration']}):**
{prev['refined_markdown']}

**Changes You Made:**
{changes_summary}

**Task:**
Review your previous refinement against the image.
- Verify your changes were correct
- Check if you missed anything
- Look for new errors

**Return JSON:**
{{
  "refined_markdown": "...",  // Updated markdown (or same if no changes)
  "changes": [...],  // Only NEW changes (not previous ones)
  "overall_confidence": 0.95
}}

If no changes needed, return:
{{
  "refined_markdown": "<same as previous>",
  "changes": [],
  "overall_confidence": 0.95,
  "note": "No changes needed"
}}
"""
```

---

## Success Criteria

### Functional Requirements
- ✅ **Iterative refinement works**: Multiple iterations execute successfully
- ✅ **History passed correctly**: Each iteration sees previous refinements
- ✅ **Convergence detection**: Stops when no changes needed
- ✅ **Markdown preserved**: Output format matches input format
- ✅ **Changes tracked**: All modifications logged with metadata
- ✅ **Cost tracked**: Total cost calculated and returned

### Quality Requirements
- ✅ **Accuracy improves**: Each iteration should increase confidence
- ✅ **No degradation**: Later iterations don't break earlier fixes
- ✅ **Conservative**: Only changes clear errors, doesn't over-correct
- ✅ **Comprehensive**: Checks all content types (text, tables, signatures, etc.)

### Performance Requirements
- ✅ **Reasonable latency**: 3 iterations complete in < 60 seconds
- ✅ **Cost-effective**: Average cost increase ≤ 3x single-pass refinement
- ✅ **Efficient**: Stops early when possible (convergence, high confidence)

### User Experience Requirements
- ✅ **Transparent**: User sees what changed and why
- ✅ **Configurable**: User controls max iterations and thresholds
- ✅ **Backward compatible**: Existing refine=True still works
- ✅ **Clear output**: Refinement history included in result

---

## Testing Strategy

### Unit Tests
1. `RefinementHistory` class methods
2. Convergence detection logic
3. Change metadata parsing
4. Stopping criteria evaluation

### Integration Tests
1. Single iteration refinement
2. Multi-iteration refinement (2-3 iterations)
3. Early stopping (convergence)
4. Early stopping (high confidence)
5. Max iterations safety limit

### End-to-End Tests
1. Simple document (converges in 1 iteration)
2. Complex document (requires 2-3 iterations)
3. Perfect extraction (no changes needed)
4. Document with mixed errors (text + table errors)

### Test Images
- Simple invoice (1 table, few paragraphs)
- Complex form (multiple tables, signatures, handwritten)
- Clean scan (high quality, minimal errors)
- Poor scan (low quality, many errors)

---

## Rollout Plan

### Phase 1: Development (Week 1)
- Implement core agentic loop
- Add Markdown-first handling
- Basic change tracking

### Phase 2: Testing (Week 2)
- Unit tests
- Integration tests
- Cost/latency benchmarking

### Phase 3: Refinement (Week 3)
- Optimize prompts based on test results
- Tune stopping criteria
- Performance improvements

### Phase 4: Documentation & Release (Week 4)
- Update README with agentic refinement docs
- Add usage examples
- Update CLAUDE.md with implementation details
- Release with opt-in flag

---

## Risk Mitigation

### Risk 1: High Cost
- **Mitigation**: Default to single-pass, make agentic opt-in
- **Monitoring**: Log cost per iteration, warn if > $0.50
- **Fallback**: Auto-disable if cost exceeds threshold

### Risk 2: Poor Convergence
- **Mitigation**: Max iterations safety limit (default: 3)
- **Monitoring**: Track convergence rate in tests
- **Fallback**: Return best iteration if oscillation detected

### Risk 3: Quality Degradation
- **Mitigation**: Validate each iteration, rollback if worse
- **Monitoring**: Compare confidence scores across iterations
- **Fallback**: Keep previous iteration if confidence drops

### Risk 4: Latency Explosion
- **Mitigation**: Parallel processing where possible
- **Monitoring**: Timeout per iteration (60s max)
- **Fallback**: Return partial result if timeout

---

## Success Metrics

### Accuracy Metrics
- **Target**: 5-10% accuracy improvement over single-pass
- **Measure**: Compare against ground truth on benchmark dataset
- **Track**: Per-iteration accuracy gain

### Cost Metrics
- **Target**: Average cost ≤ 3x single-pass refinement
- **Measure**: Total API cost across iterations
- **Track**: Cost per document type (simple vs complex)

### Performance Metrics
- **Target**: 95% of documents complete in < 60 seconds
- **Measure**: Total processing time (all iterations)
- **Track**: Latency distribution (p50, p95, p99)

### User Metrics
- **Target**: 80% of iterations converge before max_iterations
- **Measure**: Convergence rate on production data
- **Track**: Average iterations needed per document type

---

## Implementation Priority

**CRITICAL (Must Have):**
1. Core agentic loop with iteration history
2. Markdown-first format preservation
3. Basic convergence detection
4. Max iterations safety limit

**HIGH (Should Have):**
5. Structured change tracking
6. Confidence-based stopping
7. Cost/latency monitoring
8. OCRPipeline integration

**MEDIUM (Nice to Have):**
9. Advanced stopping criteria (quality regression detection)
10. Detailed change metadata
11. Per-iteration validation

**LOW (Future):**
12. Parallel iteration processing
13. Adaptive iteration limits
14. ML-based convergence prediction

---

This implementation plan provides a clear roadmap from basic iteration to production-ready agentic refinement with proper monitoring, safety limits, and user control.
