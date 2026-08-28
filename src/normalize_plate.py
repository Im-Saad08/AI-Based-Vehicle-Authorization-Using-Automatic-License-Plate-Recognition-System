import re

# Region/city/province words that appear on plates but are NOT part of the plate number
# We filter out ANY token that CONTAINS these as substrings (not just exact match)
# to catch joined OCR tokens like "ICTISLAMABAD", "ISLAHABAD", etc.
REGION_WORDS = {
    # Provinces/territories
    "PUNJAB", "SINDH", "KPK", "KHYBER", "PAKHTUNKHWA", "BALOCHISTAN", "BALUCHISTAN",
    "ISLAMABAD", "ICT", "ISL", "CCT", "FEDERAL", "CAPITAL", "TERRITORY",
    # Government plates
    "GOVERNMENT", "GOVT", "GOVT OF", "PAKISTAN", "PAK", "FED",
    # Cities/regions commonly printed on plates
    "LAHORE", "KARACHI", "RAWALPINDI", "PESHAWAR", "QUETTA", "MULTAN", "FAISALABAD",
    "GUJRANWALA", "HYDERABAD", "SARGODHA", "SUKKUR", "BAHAWALPUR", "JHELUM",
    "GILGIT", "SKARDU", "MIRPUR", "MUZAFFARABAD",
    # Common compound region labels seen on plates
    "ICTISLAMABAD", "ISLAHABAD", "ISLAMABADICT", "PUNJABGOVT", "SINDHGOVT",
    "KPKGOVT", "FEDERALGOVT", "KHYBERPAKHTUNKHWA", "KHYBERPAXHTUNIKHWA"
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
    """Correct character confusion with position-aware logic.

    For Pakistani plates (typically LETTERS + NUMBERS format like ACZ853, LE151051, FZ886):
    - Find the boundary between letters and digits
    - Only convert letters to digits in the DIGIT segment (right side)
    - Keep letters as letters in the LETTER segment (left side)
    """
    if not token:
        return ""

    # Check if token contains both letters and digits (mixed alphanumeric)
    has_letters = any(c.isalpha() for c in token)
    has_digits = any(c.isdigit() for c in token)

    # Pure alpha or pure digit token - use original logic
    if not (has_letters and has_digits):
        digit_count = sum(1 for c in token if c.isdigit())
        starts_with_digit = token[0].isdigit()

        if starts_with_digit or digit_count >= 1:
            # Pure digit or starts with digit -> convert letters to digits
            fixed = []
            for c in token:
                if c in CHAR_TO_DIGIT:
                    fixed.append(CHAR_TO_DIGIT[c])
                else:
                    fixed.append(c)
            return "".join(fixed)
        else:
            # Pure alpha -> convert digits to letters
            fixed = []
            for c in token:
                if c in DIGIT_TO_CHAR:
                    fixed.append(DIGIT_TO_CHAR[c])
                else:
                    fixed.append(c)
            return "".join(fixed)

    # Mixed alphanumeric token - position-aware correction
    # Find the letter-to-digit transition boundary
    # Pakistani plates: LETTERS followed by DIGITS
    boundary_idx = None
    for i in range(1, len(token)):
        if token[i-1].isalpha() and token[i].isdigit():
            boundary_idx = i
            break

    # If no clear boundary found, fall back to original logic
    if boundary_idx is None:
        # Try: starts with digit -> all to digits
        if token[0].isdigit():
            fixed = []
            for c in token:
                if c in CHAR_TO_DIGIT:
                    fixed.append(CHAR_TO_DIGIT[c])
                else:
                    fixed.append(c)
            return "".join(fixed)
        else:
            # Starts with letter -> keep as letters
            fixed = []
            for c in token:
                if c in DIGIT_TO_CHAR:
                    fixed.append(DIGIT_TO_CHAR[c])
                else:
                    fixed.append(c)
            return "".join(fixed)

    # Apply position-aware correction
    fixed = []
    for i, c in enumerate(token):
        if i < boundary_idx:
            # LETTER segment: keep letters, convert any misread digits to letters
            if c in DIGIT_TO_CHAR:
                fixed.append(DIGIT_TO_CHAR[c])
            else:
                fixed.append(c)
        else:
            # DIGIT segment: keep digits, convert any misread letters to digits
            if c in CHAR_TO_DIGIT:
                fixed.append(CHAR_TO_DIGIT[c])
            else:
                fixed.append(c)

    return "".join(fixed)


def normalize_plate_text(text):
    if text is None:
        text = ""

    raw_text = str(text).upper().strip()

    # Extract alphanumeric tokens
    raw_tokens = re.findall(r"[A-Z0-9]+", raw_text)

    # Filter out region words — match tokens that CONTAIN any region word as substring
    # (not just exact match), to catch OCR-joined tokens like "ICTISLAMABAD"
    def is_region_token(token):
        for region_word in REGION_WORDS:
            if region_word in token:
                return True
        return False

    tokens = [t for t in raw_tokens if not is_region_token(t)]

    # Fix character confusion in tokens
    corrected_tokens = [fix_token_characters(t) for t in tokens]

    # Join into a single-line canonical plate string without spaces or dashes
    # All corrected tokens are merged in reading order (top-to-bottom, left-to-right)
    # No special-casing for any 2-digit segment — they are part of the plate identity
    plate_number = "".join(corrected_tokens)

    return {
        "raw_text": raw_text,
        "plate_number": plate_number
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