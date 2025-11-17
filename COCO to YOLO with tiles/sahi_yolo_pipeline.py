import os
from sahi.slicing import slice_coco
from sahi.utils.coco import Coco
from ultralytics.data.converter import convert_coco

def process_and_convert_dataset(
    coco_json_path,
    coco_images_dir,
    output_dir,
    yolo_labels_dir,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.0,
    overlap_width_ratio=0.0,
    min_area_ratio=0.1,
    image_format="png",
    remove_empty_annotations=True,
    verbose=True
):
    """
    Slices a COCO dataset using SAHI, cleans the output, and converts the 
    sliced dataset annotations to YOLO format.

    Parameters:
    - coco_json_path (str): Original COCO annotation file path.
    - coco_images_dir (str): Directory of original images.
    - output_dir (str): Output directory for the sliced COCO dataset (images/JSON).
    - yolo_labels_dir (str): Output directory for YOLO format labels.
    - slice_height (int): Tile height in pixels (default: 640).
    - slice_width (int): Tile width in pixels (default: 640).
    - overlap_height_ratio (float): Vertical overlap ratio (default: 0.0).
    - overlap_width_ratio (float): Horizontal overlap ratio (default: 0.0).
    - min_area_ratio (float): Minimum area ratio for annotations (default: 0.1).
    - image_format (str): Output image format (default: "png").
    - remove_empty_annotations (bool): Skip tiles with no annotations (default: True).
    - verbose (bool): Verbosity flag (default: True).
    """
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    # ===============================
    # 2) Run SAHI Slicing
    # ===============================
    slice_coco(
        coco_annotation=coco_json_path,
        images_dir=coco_images_dir,
        output_dir=output_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        min_area_ratio=min_area_ratio,
        image_format=image_format,
        remove_empty_annotations=remove_empty_annotations,
        verbose=verbose
    )
    print("[INFO] Slicing finished. Output saved to:", output_dir)

    # ===============================
    # 3) Load and Print COCO Stats
    # ===============================
    out_coco = os.path.join(output_dir, "coco.json")  # SAHI default output

    try:
        coco = Coco.from_coco_dict_or_path(out_coco)  # Load resulting COCO file
        print("[INFO] COCO Stats:")
        print(coco.stats)
    except FileNotFoundError:
        print(f"[ERROR] Sliced COCO file not found at: {out_coco}")
        return # Exit if file not found

    # ===============================
    # 4) Clean Folder (Remove Unused Files)
    # ===============================
    folder_path = output_dir
    file_names = set([img.filename for img in coco.images])  # Valid image names

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Delete files not in COCO images list and not JSON
        if filename not in file_names and not filename.endswith(".json"):
            os.remove(file_path)
            print(f"Removed: {filename}")

    # ===============================
    # 5) Convert COCO → YOLO Format
    # ===============================
    convert_coco(
        labels_dir=output_dir,    # Directory containing sliced JSON
        save_dir=yolo_labels_dir, # Output directory for YOLO labels
        use_segments=True,        # Enable segmentation conversion
        use_keypoints=False,
        cls91to80=False           # Set False for custom datasets
    )

    print("[INFO] Conversion to YOLO format completed.")
