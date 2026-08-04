import re


# ----------------------------------------
# Normalize OCR plate text
# ----------------------------------------
def normalize_plate_text(text):

    # Handle empty input
    if text is None:
        text = ""

    # Convert to string and uppercase
    raw_text = str(text).upper().strip()

    # Extract alphanumeric tokens
    #
    # Examples:
    # "MNA 08 5445"   -> ["MNA", "08", "5445"]
    # "MNA-08-5445"   -> ["MNA", "08", "5445"]
    # "mna 08 5445"   -> ["MNA", "08", "5445"]
    #
    tokens = re.findall(
        r"[A-Z0-9]+",
        raw_text
    )

    # ----------------------------------------
    # Identify numeric tokens
    # ----------------------------------------

    # Two-digit numeric tokens
    year_candidates = [
        token
        for token in tokens
        if token.isdigit()
        and len(token) == 2
    ]

    # Longer numeric tokens
    main_number_candidates = [
        token
        for token in tokens
        if token.isdigit()
        and len(token) >= 3
    ]

    # ----------------------------------------
    # Default registration year
    # ----------------------------------------
    registration_year = ""

    # Make a copy of tokens
    plate_tokens = tokens.copy()

    # ----------------------------------------
    # Identify registration year
    #
    # We only classify a 2-digit number
    # as a registration year when:
    #
    # 1. Exactly one 2-digit number exists
    # 2. At least one longer numeric token exists
    #
    # Example:
    #
    # MNA 08 5445
    #
    # 08   -> Registration Year
    # 5445 -> Main plate number
    # ----------------------------------------
    if (
        len(year_candidates) == 1
        and len(main_number_candidates) >= 1
    ):

        registration_year = year_candidates[0]

        # Remove the year from plate tokens
        plate_tokens.remove(
            registration_year
        )

    # ----------------------------------------
    # Build canonical plate number
    # ----------------------------------------
    plate_number = "".join(
        plate_tokens
    )

    # ----------------------------------------
    # Return all information
    # ----------------------------------------
    return {
        "raw_text": raw_text,
        "plate_number": plate_number,
        "registration_year": registration_year
    }