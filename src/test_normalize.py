from normalize_plate import normalize_plate_text


test_cases = [
    "MNA 5445",
    "MNA-5445",
    "mna 5445",
    "MNA 08 5445",
    "MNA-08-5445",
    "MNA08",
    "ABC123",
    "ICT-5678"
]


for text in test_cases:

    result = normalize_plate_text(
        text
    )

    print("\nOriginal:", text)

    print(
        "Plate Number:",
        result["plate_number"]
    )