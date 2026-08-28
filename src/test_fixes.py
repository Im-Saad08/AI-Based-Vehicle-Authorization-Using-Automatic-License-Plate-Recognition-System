"""Test script to verify fix_token_characters and plate_format_score fixes."""
from normalize_plate import fix_token_characters, normalize_plate_text
from recognize_plate import plate_format_score


def test_fix_token_characters():
    print("\n=== Testing fix_token_characters ===\n")
    test_cases = [
        ('ACZ853', 'ACZ853'),
        ('FZ886', 'FZ886'),
        ('LE151051', 'LE151051'),
        ('4OOB', '4008'),
        ('MLE4008', 'MLE4008'),
        ('MLE', 'MLE'),
        ('1234', '1234'),
        ('ABC123', 'ABC123'),
        ('MNA08', 'MNA08'),
    ]

    all_pass = True
    for inp, expected in test_cases:
        result = fix_token_characters(inp)
        status = 'OK' if result == expected else 'FAIL'
        if result != expected:
            all_pass = False
        print(f'  {status} fix_token_characters("{inp}") = "{result}" (expected: "{expected}")')

    return all_pass


def test_plate_format_score():
    print("\n=== Testing plate_format_score length scoring ===\n")
    # Test that complete two-line reads get higher scores than truncated
    test_cases = [
        # (plate_text, expected_length_bucket, expected_score_range)
        ('ACZ853', 6, 15),      # Single line - should get +15
        ('FZ886', 5, 15),       # Single line - should get +15
        ('LE151051', 8, 25),    # Complete two-line - should get +25 (HIGHER than single!)
        ('ACZ853ICT', 9, 25),   # With region still attached - should get +25
        ('ABC1234', 7, 25),     # Complete single - should get +25
        ('AB12', 4, 15),        # Short - should get +15
        ('ABC12345', 8, 25),    # Complete - should get +25
    ]

    all_pass = True
    for text, length, expected_base in test_cases:
        score = plate_format_score(text)
        status = 'OK' if (score >= expected_base) else 'FAIL'
        if score < expected_base:
            all_pass = False
        print(f'  {status} plate_format_score("{text}") = {score} (length={length}, base={expected_base})')

    # Verify two-line complete reads beat single-line
    print("\n--- Verifying twoline > singleline preference ---")
    single_score = plate_format_score('ACZ853')  # 6 chars, +15
    twoline_score = plate_format_score('LE151051')  # 8 chars, +25

    if twoline_score > single_score:
        print(f'  OK  Twoline read score ({twoline_score}) > single-line score ({single_score})')
    else:
        print(f'  FAIL Twoline read score ({twoline_score}) <= single-line score ({single_score})')
        all_pass = False

    return all_pass


def test_normalize_with_region_filtering():
    print("\n=== Testing normalize_plate_text with region filtering ===\n")
    test_cases = [
        ('ACZ 853', 'ACZ853'),
        ('LE 15 1051', 'LE151051'),
        ('FZ 886', 'FZ886'),
        ('ICT ISLAMABAD ACZ853', 'ACZ853'),  # Region words filtered out
        ('PUNJAB GOVT FZ886', 'FZ886'),
    ]

    all_pass = True
    for inp, expected in test_cases:
        result = normalize_plate_text(inp)
        plate = result['plate_number']
        status = 'OK' if plate == expected else 'FAIL'
        if plate != expected:
            all_pass = False
        print(f'  {status} normalize_plate_text("{inp}") = "{plate}" (expected: "{expected}")')

    return all_pass


if __name__ == '__main__':
    print("=" * 60)
    print("TESTING FIXES: position-aware correction + length scoring")
    print("=" * 60)

    t1 = test_fix_token_characters()
    t2 = test_plate_format_score()
    t3 = test_normalize_with_region_filtering()

    print("\n" + "=" * 60)
    if t1 and t2 and t3:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)
