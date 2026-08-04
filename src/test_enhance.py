import cv2
import os

from enhance_plate import enhance_plate


# ----------------------------------------
# Input plate image
# ----------------------------------------
image_path = "img/output/cropped_plates/DSC_1049_plate_1.png"


# ----------------------------------------
# Read plate image
# ----------------------------------------
plate_image = cv2.imread(
    image_path
)

if plate_image is None:

    print(
        "Error: Unable to load plate image."
    )

    exit()


# ----------------------------------------
# Generate enhanced versions
# ----------------------------------------
enhanced_versions = enhance_plate(
    plate_image
)


# ----------------------------------------
# Create output folder
# ----------------------------------------
output_folder = (
    "img/output/enhanced_plates"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ----------------------------------------
# Save every enhanced version
# ----------------------------------------
for name, image in enhanced_versions.items():

    output_path = os.path.join(
        output_folder,
        f"{name}.png"
    )

    cv2.imwrite(
        output_path,
        image
    )

    print(
        f"Saved: {output_path}"
    )


# ----------------------------------------
# Final message
# ----------------------------------------
print(
    "\nEnhancement completed successfully."
)