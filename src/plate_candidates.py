from itertools import product


# ----------------------------------------
# Common OCR Confusions
# ----------------------------------------

OCR_CONFUSIONS = {

    "0": ["0", "O"],
    "O": ["O", "0"],

    "1": ["1", "I"],
    "I": ["I", "1"],

    "2": ["2", "Z"],
    "Z": ["Z", "2"],

    "5": ["5", "S"],
    "S": ["S", "5"],

    "8": ["8", "B"],
    "B": ["B", "8"]

}


# ----------------------------------------
# Generate candidate plates
# ----------------------------------------

def generate_candidates(plate_text):

    characters = []

    for char in plate_text:

        if char in OCR_CONFUSIONS:

            characters.append(
                OCR_CONFUSIONS[char]
            )

        else:

            characters.append(
                [char]
            )

    candidates = []

    for candidate in product(*characters):

        candidates.append(
            "".join(candidate)
        )

    return list(
        dict.fromkeys(candidates)
    )