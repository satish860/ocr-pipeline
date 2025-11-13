# Plan: OCR Pipeline Enhancement - Document Detection & Specialized Handling

## Overview
Add document type detection, rotation fixing, and specialized handling for forms and checks while keeping general documents (including invoices) in markdown format.

---

## Phase 1: Combined Detection & Classification (PRIORITY)

### What We're Building
Single API call to Qwen-8B that detects both:
1. **Document Type**: form, check, or general
2. **Rotation**: 0°, 90°, 180°, or 270°

### Implementation

**Step 1.1: Create Document Classifier**
- File: `src/ocr_pipeline/document_classifier.py`
- Model: `qwen/qwen3-vl-8b-instruct` (fast, cheap)
- Method: `classify_and_detect_rotation(image) -> Dict`
- Returns:
  ```json
  {
    "document_type": "form|check|general",
    "rotation_degrees": 0|90|180|270,
    "confidence": 0.95
  }
  ```

**Step 1.2: Enhance Image Preprocessor**
- File: `src/ocr_pipeline/image_preprocessor.py`
- Add: Vision-based orientation detection using classifier
- Flow:
  1. EXIF auto-rotation (instant, free)
  2. Call Qwen-8B for rotation detection (if still sideways)
  3. Apply rotation
  4. Fine deskew (existing code)

**Step 1.3: Integrate into API**
- File: `src/api/main.py`
- Add classification step after preprocessing
- Store document_type in pipeline context
- Add to response: `document_type`, `rotation_degrees`, `confidence`

**Step 1.4: Test Detection**
- Test `input/rotated.png` → should detect "form" + 90° rotation
- Test `input/Cheque.jpg` → should detect "check" + 0° rotation
- Test `input/image-1.jpg` → should detect "general" + 0° rotation

**Estimated Time:** 4-5 hours

---

## Phase 2: Form-Specific Handling

### Step 2.1: Form Structure Extraction
- File: `src/ocr_pipeline/form_extractor.py`
- Extract form as structured JSON:
  ```json
  {
    "form_type": "generic_form",
    "sections": [
      {
        "title": "Personal Information",
        "fields": [
          {"label": "Name:", "value": "John Doe", "type": "text", "bbox": [...]}
        ]
      }
    ]
  }
  ```

### Step 2.2: Form HTML Generator
- File: `src/ocr_pipeline/html_generator.py`
- Generate visual-only HTML (read-only, not editable)
- Include bounding box overlay (colored boxes like annotated image)
- CSS styling for professional form display

### Step 2.3: API Integration
- Route forms to HTML generation
- Add `form_structure` and `html` to response

**Estimated Time:** 5-6 hours

---

## Phase 3: Check-Specific Handling

### Step 3.1: Check Field Extraction
- File: `src/ocr_pipeline/check_extractor.py`
- Extract structured check data:
  ```json
  {
    "check_number": "1234",
    "date": "2025-01-15",
    "payee": "John Smith",
    "amount_numeric": "$1,500.00",
    "amount_written": "One thousand five hundred dollars",
    "memo": "Invoice #123",
    "signature_present": true
  }
  ```

### Step 3.2: Check HTML Generator
- Extend `html_generator.py`
- Visual representation of check with fields highlighted

**Estimated Time:** 3-4 hours

---

## Phase 4: Response Routing & API Finalization

### Step 4.1: Conditional Pipeline Routing
- Based on document_type, route to:
  - **Forms** → Form extraction → HTML output
  - **Checks** → Check extraction → Structured JSON + HTML
  - **General** (including invoices) → Existing markdown pipeline

### Step 4.2: Enhanced API Response
```json
{
  "success": true,
  "document_type": "form|check|general",
  "rotation_degrees": 90,
  "confidence": 0.95,
  "output_format": "html|markdown",

  "markdown": "...",
  "html": "<html>...</html>",
  "form_structure": {...},
  "check_data": {...},

  "regions": [...],
  "annotated_image": "base64..."
}
```

**Estimated Time:** 2-3 hours

---

## Architecture Overview

```
Image Input
    ↓
┌───────────────────────────┐
│ Preprocessing             │
│ 1. EXIF rotation         │
│ 2. Qwen-8B Detection     │ ← New: Combined detection
│    - Document type       │
│    - Rotation degrees    │
│ 3. Apply rotation        │
│ 4. Fine deskew           │
└───────────┬───────────────┘
            ↓
┌───────────────────────────┐
│ Layout Detection          │
│ Qwen-30B (detailed)      │ ← Existing
└───────────┬───────────────┘
            ↓
     Document Type?
            ↓
    ┌───────┴──────┬──────────┐
    ↓              ↓          ↓
  FORM          CHECK      GENERAL
    ↓              ↓          ↓
Form Extract   Check      Markdown
    ↓          Extract     Pipeline
Form HTML      JSON +         ↓
    ↓          HTML       Markdown
    ↓              ↓          ↓
    └──────────────┴──────────┘
               ↓
        JSON Response
```

---

## Model Usage Strategy

| Task | Model | Why |
|------|-------|-----|
| Document Classification | `qwen/qwen3-vl-8b-instruct` | Fast, cheap, good for simple classification |
| Rotation Detection | `qwen/qwen3-vl-8b-instruct` | Same call as classification |
| Layout Detection | `qwen/qwen3-vl-30b-a3b-thinking` | More accurate for complex layouts |
| OCR Extraction | `google/gemini-2.5-flash` | High quality text extraction |

---

## Document Type Definitions

| Type | Description | Output Format | Examples |
|------|-------------|---------------|----------|
| **form** | Has form_labels, form_fields, checkboxes | HTML with overlay | Tax forms, applications, surveys |
| **check** | Bank check/cheque with MICR, payee, amount | Structured JSON + HTML | Personal checks, business checks |
| **general** | Everything else: invoices, letters, articles, tables | Markdown | Invoices, receipts, documents, tables |

---

## Files to Create

1. `src/ocr_pipeline/document_classifier.py` - Classification + rotation detection
2. `src/ocr_pipeline/form_extractor.py` - Form structure extraction
3. `src/ocr_pipeline/check_extractor.py` - Check field extraction
4. `src/ocr_pipeline/html_generator.py` - HTML generation for forms/checks
5. `tests/test_document_classifier.py` - Tests for classification
6. `tests/test_form_extractor.py` - Tests for form extraction
7. `tests/test_check_extractor.py` - Tests for check extraction

## Files to Modify

1. `src/ocr_pipeline/image_preprocessor.py` - Add vision-based rotation detection
2. `src/api/main.py` - Add classification, routing, enhanced response
3. `pyproject.toml` - Add markdown library dependency

---

## Total Effort Estimate
- **Phase 1**: 4-5 hours (Priority - do first)
- **Phase 2**: 5-6 hours
- **Phase 3**: 3-4 hours
- **Phase 4**: 2-3 hours
- **Total**: 14-18 hours

---

## Success Criteria
- [ ] Rotation detection works on `rotated.png` (90° → corrected)
- [ ] Form detection works on `rotated.png` → outputs HTML
- [ ] Check detection works on `Cheque.jpg` → outputs structured JSON
- [ ] General docs (tables) still output markdown
- [ ] API backward compatible (markdown always included)
- [ ] Bounding box overlay visible in HTML forms

---

## Implementation Notes

### Phase 1 Priority Items
1. Start with document classification using Qwen-8B
2. Combine rotation detection in same API call (cost-effective)
3. Test on all three document types before proceeding
4. Ensure EXIF rotation works first (free, instant)

### Key Design Decisions
- **Invoices = General**: No separate handling needed, markdown works well
- **Combined Detection**: One Qwen-8B call for type + rotation
- **Model Optimization**: 8B for classification, 30B for layout detection
- **HTML Output**: Visual-only (read-only), not editable forms
- **Bounding Boxes**: Overlay on image in HTML output

---

## Next Steps
1. Implement **Phase 1** only (detection & classification)
2. Test thoroughly on sample images
3. Review results before proceeding to Phase 2
4. Iterate one phase at a time
