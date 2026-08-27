import re

REGION_WORDS = {
    "PUNJAB", "ISLAMABAD", "ICT", "SINDH", "KPK", "BALOCHISTAN",
    "FEDERAL", "GOVERNMENT", "PAKISTAN", "GOVT", "ISL", "KHYBER",
    "PAKHTUNKHWA", "CCT", "ICTISLAMABAD"
}

CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8"
}

DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B"
}


def fix_token_characters(token):
    """Correct character confusion: if token starts with a digit or contains digits (e.g. 4OOB), convert letters to digits."""
    if not token:
        return ""

    digit_count = sum(1 for c in token if c.isdigit())
    starts_with_digit = token[0].isdigit()

    # If starts with a digit or contains numbers (e.g. 4OOB -> 4008)
    if starts_with_digit or digit_count >= 1:
        fixed = []
        for c in token:
            if c in CHAR_TO_DIGIT:
                fixed.append(CHAR_TO_DIGIT[c])
            else:
                fixed.append(c)
        return "".join(fixed)
    else:
        # Alpha token (e.g. MLE, MNA)
        fixed = []
        for c in token:
            if c in DIGIT_TO_CHAR:
                fixed.append(DIGIT_TO_CHAR[c])
            else:
                fixed.append(c)
        return "".join(fixed)


def normalize_plate_text(text):
    if text is None:
        text = ""

    raw_text = str(text).upper().strip()

    # Extract alphanumeric tokens
    raw_tokens = re.findall(r"[A-Z0-9]+", raw_text)

    # Filter out region words
    tokens = [t for t in raw_tokens if t not in REGION_WORDS]

    # Fix character confusion in tokens
    corrected_tokens = [fix_token_characters(t) for t in tokens]

    # Identify numeric registration year candidates (2 digits)
    year_candidates = [
        t for t in corrected_tokens
        if t.isdigit() and len(t) == 2
    ]

    # Identify main plate numbers (3+ digits)
    main_number_candidates = [
        t for t in corrected_tokens
        if t.isdigit() and len(t) >= 3
    ]

    registration_year = ""
    plate_tokens = corrected_tokens.copy()

    # Extract 2-digit registration year if present alongside main number
    if len(year_candidates) == 1 and len(main_number_candidates) >= 1:
        registration_year = year_candidates[0]
        plate_tokens.remove(registration_year)

    # Join into a single-line canonical plate string without spaces or dashes
    plate_number = "".join(plate_tokens)

    return {
        "raw_text": raw_text,
        "plate_number": plate_number,
        "registration_year": registration_year
    }


def format_plate_display(plate_number):
    """Format plate number for display as ABC-123 (letters-numbers with dash)."""
    if not plate_number:
        return ""

    # Find the boundary between letters and numbers
    # Pakistani plates typically: 2-3 letters followed by 3-4 digits
    match = re.match(r'^([A-Z]+)(\d+)$', plate_number)
    if match:
        letters = match.group(1)
        numbers = match.group(2)
        return f"{letters}-{numbers}"

    # If no clear letter-number split, return as-is
    return plate_number