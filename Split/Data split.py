import os
import random
import shutil

def split_dataset(images_dir, labels_dir, output_base, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    """
    Splits an image dataset and its corresponding label files into training, 
    validation, and test sets based on specified ratios, and copies them to 
    a new directory structure.

    Parameters:
    - images_dir (str): Folder containing the images.
    - labels_dir (str): Folder containing the corresponding label files.
    - output_base (str): Base folder where the split datasets will be stored.
    - train_ratio (float): Ratio for the training set (default: 0.8).
    - val_ratio (float): Ratio for the validation set (default: 0.2).
    - test_ratio (float): Ratio for the test set (default: 0.0).
    """

    # -----------------------------
    # Dataset split ratios
    # -----------------------------
    splits = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio
    }

    # -----------------------------
    # Gather all image files
    # -----------------------------
    # Only include files with common image extensions
    images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)  # Shuffle the list to ensure random distribution
    total = len(images)     # Total number of images

    # -----------------------------
    # Calculate number of files per split
    # -----------------------------
    counts = {
        "train": int(total * train_ratio),
        "val": int(total * val_ratio)
    }
    counts["test"] = total - counts["train"] - counts["val"]  # Assign remaining files to test (if any)

    # If a split ratio is 0, force its file count to 0
    for split, ratio in splits.items():
        if ratio == 0:
            counts[split] = 0

    # -----------------------------
    # Assign images to each split
    # -----------------------------
    train_files = images[:counts["train"]]
    val_files = images[counts["train"]:counts["train"] + counts["val"]]
    test_files = images[counts["train"] + counts["val"]:]

    files_map = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # -----------------------------
    # Function to copy images and labels to the target folder
    # -----------------------------
    def copy_files(file_list, split_name):
        for img_name in file_list:
            img_path = os.path.join(images_dir, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"  # Corresponding label file
            label_path = os.path.join(labels_dir, label_name)

            # Copy the image
            shutil.copy(img_path, f"{output_base}/{split_name}/images/{img_name}")

            # Copy the label if it exists
            if os.path.exists(label_path):
                shutil.copy(label_path, f"{output_base}/{split_name}/labels/{label_name}")

    # -----------------------------
    # Create folders and copy files for splits with ratio > 0
    # -----------------------------
    for split, ratio in splits.items():
        if ratio == 0:
            print(f"📌 Skipping {split} split because its ratio = 0")
            continue

        os.makedirs(f"{output_base}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_base}/{split}/labels", exist_ok=True)

        copy_files(files_map[split], split)

    print("Dataset splitting completed successfully 🎉")