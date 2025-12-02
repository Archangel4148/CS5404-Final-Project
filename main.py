import json
import os
from pathlib import Path

import torch
from loading_things import load_dataset_relations, load_ply_pointcloud
from download_drive_images import download_files
from paths import fix_path
from pipeline import process_one_object
from restructure_files import move_images_and_build_full_relations

DESIRED_FILE_COUNT = 3  # How many item files (tar.gzip) to download
OBJECTS_PER_GROUP = 3  # How many objects from each group to process
IMAGES_PER_OBJECT = 2  # How many images per object to process

# Constructing paths
ROOT_DIR = Path(os.getcwd())
BASE_DB_PATH = fix_path(ROOT_DIR / "datasets" / "omniobject3d")
DRIVE_LINK_LIST_PATH = fix_path(ROOT_DIR / "drive_file_ids.txt")
RAW_IMAGES_PATH = BASE_DB_PATH / "images_raw"
IMAGES_PATH = BASE_DB_PATH / "images"
POINTCLOUD_ROOT = BASE_DB_PATH / "ply_16384" / "extracted" / "16384"
FINAL_RELATION_FILE_PATH = BASE_DB_PATH / "object_relations.json"
OUTPUT_ROOT = BASE_DB_PATH / "spar3d_outputs"

# Test code for PyTorch/GPU
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

############# DOWNLOADING DATABASE FILES #############
SKIP_DOWNLOAD = False

if not SKIP_DOWNLOAD:
    # Read desired file IDs
    urls = open(DRIVE_LINK_LIST_PATH, "r").readlines()

    # Download selected ones
    wanted_files = urls[:DESIRED_FILE_COUNT]
    download_files(wanted_files, RAW_IMAGES_PATH, unzip=True)

    # Move files and build relations
    move_images_and_build_full_relations(
        move_source_path=RAW_IMAGES_PATH,
        move_dest_path=IMAGES_PATH,
        pointcloud_root=POINTCLOUD_ROOT,
        relation_output_path=FINAL_RELATION_FILE_PATH
    )

############# LOADING/TESTING #############

# Testing parameters
distortion_levels = [{"blur": 1.0, "noise": 0.0, "exposure": 1.0}]
taus = [0.1, 0.2, 0.5]

# Load relations
grouped_data = load_dataset_relations(json_path=FINAL_RELATION_FILE_PATH)
print("Loaded:", len(grouped_data.keys()), "objects")

# Identify all images that will actually be processed
images_to_keep = set()
for group_objs in grouped_data.values():
    for obj_id, fields in list(group_objs.items())[:OBJECTS_PER_GROUP]:
        img_files = sorted(list(Path(fields["images"]).glob("*.png")))[:IMAGES_PER_OBJECT]
        for img_file in img_files:
            images_to_keep.add(img_file.resolve())

# Delete unneeded images in the main image folder
for img_file in Path(IMAGES_PATH).rglob("*.png"):
    if img_file.resolve() not in images_to_keep:
        img_file.unlink()

print(f"[INFO] Kept {len(images_to_keep)} images, deleted unused images from {IMAGES_PATH}")

results = {}
for group_name, group_objs in grouped_data.items():
    results[group_name] = {}
    # Run the pipeline on each object
    for obj_id, fields in list(group_objs.items())[:OBJECTS_PER_GROUP]:
        print(f"\n=== Processing object: {obj_id} ===")

        result = process_one_object(
            object_id=obj_id,
            img_dir=fields["images"],
            gt_pointcloud_path=fields["point_cloud"],
            distortion_levels=distortion_levels,
            taus=taus,
            images_per_object=IMAGES_PER_OBJECT,
            keep_distorted=False,
            output_root=Path(OUTPUT_ROOT)
        )

        # Save result for this object
        results[group_name][obj_id] = result

# Save results to a JSON file
RESULTS_PATH = BASE_DB_PATH / "spar3d_outputs" / "pipeline_results.json"
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nAll results saved → {RESULTS_PATH}")