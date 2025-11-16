"""
Unit test to verify float/int normalization fix works correctly.

Tests the normalize_numeric_comparison() function to ensure it handles
mixed numeric/string types properly.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.normalizer import normalize_numeric_comparison
from benchmark.evaluator import compare_json


def test_normalize_numeric_comparison():
    """Test the normalize_numeric_comparison function directly."""
    print("=" * 70)
    print("TESTING: normalize_numeric_comparison()")
    print("=" * 70)

    test_cases = [
        # (value1, value2, expected_match, description)
        (17.0, "17", True, "Float 17.0 vs String '17'"),
        (17, "17", True, "Int 17 vs String '17'"),
        (1234.56, "1234.56", True, "Float 1234.56 vs String '1234.56'"),
        ("$1,234", 1234, True, "String '$1,234' vs Int 1234"),
        (17.0, "17.0", True, "Float 17.0 vs String '17.0'"),
        (17.5, "17", False, "Float 17.5 vs String '17' (different)"),
        (17, "18", False, "Int 17 vs String '18' (different)"),
        ("hello", 17, False, "String 'hello' vs Int 17 (different types)"),
    ]

    passed = 0
    failed = 0

    for value1, value2, should_match, description in test_cases:
        norm1, norm2 = normalize_numeric_comparison(value1, value2)
        matches = (norm1 == norm2)

        status = "PASS" if matches == should_match else "FAIL"

        if matches == should_match:
            passed += 1
            print(f"[{status}] {description}")
            print(f"       {value1} ({type(value1).__name__}) vs {value2} ({type(value2).__name__})")
            print(f"       Normalized: '{norm1}' vs '{norm2}' -> Match: {matches}")
        else:
            failed += 1
            print(f"[{status}] {description}")
            print(f"       {value1} ({type(value1).__name__}) vs {value2} ({type(value2).__name__})")
            print(f"       Normalized: '{norm1}' vs '{norm2}' -> Match: {matches}")
            print(f"       Expected: {should_match}, Got: {matches}")
        print()

    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_evaluator_integration():
    """Test the evaluator with float/int comparisons."""
    print("\n" + "=" * 70)
    print("TESTING: Evaluator Integration")
    print("=" * 70)

    # Test case 1: Float vs String number
    predicted = {
        "amount": 17.0,
        "quantity": 5.0,
        "total": 1234.56
    }

    ground_truth = {
        "amount": "17",
        "quantity": "5",
        "total": "1234.56"
    }

    print("\nTest 1: Float vs String Numbers")
    print(f"Predicted: {predicted}")
    print(f"Ground Truth: {ground_truth}")

    result = compare_json(predicted, ground_truth)
    accuracy = result['accuracy'] * 100

    print(f"\nResult:")
    print(f"  Total fields: {result['total_fields']}")
    print(f"  Different fields: {result['different_fields']}")
    print(f"  Accuracy: {accuracy:.1f}%")

    if accuracy == 100.0:
        print("  [PASS] All float/string numbers matched correctly!")
    else:
        print(f"  [FAIL] Expected 100% accuracy, got {accuracy:.1f}%")
        print(f"  Differences: {result['differences']}")

    # Test case 2: Mixed with actual errors
    predicted2 = {
        "amount": 17.0,  # Should match "17"
        "quantity": 6.0,  # Should NOT match "5" (real error)
        "total": 1234.56  # Should match "1234.56"
    }

    ground_truth2 = {
        "amount": "17",
        "quantity": "5",
        "total": "1234.56"
    }

    print("\n\nTest 2: Mixed (2 matches, 1 real error)")
    print(f"Predicted: {predicted2}")
    print(f"Ground Truth: {ground_truth2}")

    result2 = compare_json(predicted2, ground_truth2)
    accuracy2 = result2['accuracy'] * 100

    print(f"\nResult:")
    print(f"  Total fields: {result2['total_fields']}")
    print(f"  Different fields: {result2['different_fields']}")
    print(f"  Accuracy: {accuracy2:.1f}%")

    expected_accuracy = 66.7  # 2 out of 3 correct
    if abs(accuracy2 - expected_accuracy) < 1.0:
        print(f"  [PASS] Correctly identified 1 error (accuracy {accuracy2:.1f}%)")
    else:
        print(f"  [FAIL] Expected ~{expected_accuracy}% accuracy, got {accuracy2:.1f}%")
        print(f"  Differences: {result2['differences']}")

    return accuracy == 100.0 and abs(accuracy2 - expected_accuracy) < 1.0


def main():
    print("\n" + "=" * 70)
    print("FLOAT/INT NORMALIZATION FIX VALIDATION")
    print("=" * 70)
    print("\nThis test validates that the normalization fix correctly handles")
    print("cases where Claude extracts numbers as floats (e.g., 17.0) but")
    print("ground truth stores them as strings (e.g., '17').\n")

    # Run tests
    test1_passed = test_normalize_numeric_comparison()
    test2_passed = test_evaluator_integration()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if test1_passed and test2_passed:
        print("[SUCCESS] All tests passed!")
        print("\nThe normalization fix is working correctly:")
        print("- Float/int values are converted to strings before comparison")
        print("- Numeric equivalence is properly detected (17.0 == '17')")
        print("- Real errors are still caught (6.0 != '5')")
        print("\nExpected improvement: +2-4% accuracy on TABLE samples")
        print("(when network connectivity to openrouter.ai is restored)")
    else:
        print("[FAILURE] Some tests failed!")
        if not test1_passed:
            print("- normalize_numeric_comparison() function has issues")
        if not test2_passed:
            print("- Evaluator integration has issues")

    print("=" * 70)


if __name__ == "__main__":
    main()
