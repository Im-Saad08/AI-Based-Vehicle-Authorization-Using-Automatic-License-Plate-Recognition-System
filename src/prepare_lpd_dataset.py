import os
import shutil
import random


# ==================================================
# SETTINGS
# ==================================================

SOURCE_DATASET = r"E:/Vehicle Authorization System/dataset/Diverse-LPD"
OUTPUT_DATASET = r"E:/Vehicle Authorization System/dataset/LPD_YOLO"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ==================================================
# CHECK RATIOS
# ==================================================

if abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) > 0.001:

    print(
        "Error: Train/Val/Test ratios must add up to 1."
    )

    exit()


# ==================================================
# SOURCE FOLDERS
# ==================================================

SOURCE_IMAGES = os.path.join(
    SOURCE_DATASET,
    "images"
)

SOURCE_LABELS = os.path.join(
    SOURCE_DATASET,
    "labels"
)


# ==================================================
# CHECK SOURCE DATASET
# ==================================================

if not os.path.isdir(SOURCE_IMAGES):

    print(
        f"Error: Image folder not found:\n"
        f"{SOURCE_IMAGES}"
    )

    exit()


if not os.path.isdir(SOURCE_LABELS):

    print(
        f"Error: Label folder not found:\n"
        f"{SOURCE_LABELS}"
    )

    exit()


# ==================================================
# GET IMAGES
# ==================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


image_files = [

    file

    for file in os.listdir(SOURCE_IMAGES)

    if file.lower().endswith(image_extensions)

]


# ==================================================
# CHECK IMAGES
# ==================================================

if not image_files:

    print("Error: No images found.")

    exit()


print("\n" + "=" * 60)
print("DIVERSE-LPD DATASET PREPARATION")
print("=" * 60)

print(
    f"\nImages found: {len(image_files)}"
)


# ==================================================
# CHECK CORRESPONDING LABELS
# ==================================================

valid_images = []
missing_labels = []

for image_file in image_files:

    image_name = os.path.splitext(
        image_file
    )[0]

    label_file = image_name + ".txt"

    label_path = os.path.join(
        SOURCE_LABELS,
        label_file
    )

    if os.path.isfile(label_path):

        valid_images.append(
            image_file
        )

    else:

        missing_labels.append(
            image_file
        )


print(
    f"Images with labels: {len(valid_images)}"
)

print(
    f"Images without labels: {len(missing_labels)}"
)


# ==================================================
# USE ONLY IMAGES WITH LABELS
# ==================================================

image_files = valid_images


if not image_files:

    print(
        "\nError: No images with matching labels found."
    )

    exit()


# ==================================================
# SHUFFLE DATASET
# ==================================================

random.seed(
    RANDOM_SEED
)

random.shuffle(
    image_files
)


# ==================================================
# CALCULATE SPLIT
# ==================================================

total = len(image_files)

train_count = int(
    total * TRAIN_RATIO
)

val_count = int(
    total * VAL_RATIO
)

test_count = (
    total
    - train_count
    - val_count
)


train_images = image_files[
    :train_count
]

val_images = image_files[
    train_count:
    train_count + val_count
]

test_images = image_files[
    train_count + val_count:
]


# ==================================================
# CREATE OUTPUT FOLDERS
# ==================================================

splits = {

    "train": train_images,
    "val": val_images,
    "test": test_images

}


for split in splits:

    os.makedirs(
        os.path.join(
            OUTPUT_DATASET,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            OUTPUT_DATASET,
            split,
            "labels"
        ),
        exist_ok=True
    )


# ==================================================
# COPY IMAGES + LABELS
# ==================================================

print("\nCopying dataset files...")


for split, files in splits.items():

    print(
        f"\n{split.upper()}: {len(files)} images"
    )

    for image_file in files:

        # ------------------------------------------
        # Image
        # ------------------------------------------

        source_image = os.path.join(
            SOURCE_IMAGES,
            image_file
        )

        destination_image = os.path.join(
            OUTPUT_DATASET,
            split,
            "images",
            image_file
        )

        shutil.copy2(
            source_image,
            destination_image
        )


        # ------------------------------------------
        # Label
        # ------------------------------------------

        image_name = os.path.splitext(
            image_file
        )[0]

        label_file = image_name + ".txt"

        source_label = os.path.join(
            SOURCE_LABELS,
            label_file
        )

        destination_label = os.path.join(
            OUTPUT_DATASET,
            split,
            "labels",
            label_file
        )

        shutil.copy2(
            source_label,
            destination_label
        )


# ==================================================
# CREATE DATA.YAML
# ==================================================

yaml_path = os.path.join(
    OUTPUT_DATASET,
    "data.yaml"
)


yaml_content = f"""path: {os.path.abspath(OUTPUT_DATASET).replace(chr(92), "/")}

train: train/images
val: val/images
test: test/images

nc: 1

names:
  0: license_plate
"""


with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        yaml_content
    )


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print(
    f"\nTraining images   : {len(train_images)}"
)

print(
    f"Validation images : {len(val_images)}"
)

print(
    f"Testing images    : {len(test_images)}"
)

print(
    f"\nDataset created at:"
)

print(
    os.path.abspath(
        OUTPUT_DATASET
    )
)

print(
    f"\nYAML file:"
)

print(
    os.path.abspath(
        yaml_path
    )
)

print("\n" + "=" * 60)