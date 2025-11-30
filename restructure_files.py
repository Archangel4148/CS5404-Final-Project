import os
import shutil
import json
from pathlib import Path

def move_images_and_build_image_relations(source_root:str , dest_root:str):
    """
    Moves Google Drive images into:
         dest_root/category/object_id/*.png
    Builds { object_id: image_folder } mapping.

    After processing each "image_X/" folder:
        - Deletes that folder entirely
        - Keeps any .tar.gz files
    """
    os.makedirs(dest_root, exist_ok=True)
    relations = {}

    source_root = Path(source_root)

    for img_folder in source_root.iterdir():
        if not img_folder.is_dir():
            continue

        # e.g., "image_1/", "image_2/", ...
        print(f"[PROCESS] Handling folder: {img_folder.name}")

        for object_folder in img_folder.iterdir():
            if not object_folder.is_dir():
                continue

            object_id = object_folder.name              # e.g., anise_001
            category = object_id.split("_")[0]          # e.g., anise

            final_dir = Path(dest_root) / category / object_id
            final_dir.mkdir(parents=True, exist_ok=True)

            pngs = list(object_folder.glob("*.png"))
            if not pngs:
                print(f"[WARN] No PNGs inside {object_folder}")
                continue

            for png in pngs:
                shutil.move(str(png), final_dir / png.name)

            relations[object_id] = str(final_dir)
            print(f"[OK] Moved images for {object_id} → {final_dir}")

        # ---------------------------------------------------------------------
        # CLEANUP: delete extracted folder, keep only original .tar.gz files
        # ---------------------------------------------------------------------
        print(f"[CLEANUP] Deleting extracted folder: {img_folder.name}")
        shutil.rmtree(img_folder, ignore_errors=True)

    return relations


def merge_with_pointclouds(image_map, pointcloud_root, output_path):
    pointcloud_root = Path(pointcloud_root)
    final_map = {}

    for category_dir in pointcloud_root.iterdir():
        if not category_dir.is_dir():
            continue

        for object_dir in category_dir.iterdir():
            if not object_dir.is_dir():
                continue

            object_id = object_dir.name
            ply_files = list(object_dir.glob("*.ply"))

            if not ply_files:
                print(f"[WARN] No .ply for {object_id}")
                continue

            pc_path = str(ply_files[0])
            img_path = image_map.get(object_id)

            if img_path is None:
                print(f"[WARN] No images for {object_id}")
                continue

            final_map[object_id] = {
                "point_cloud": pc_path,
                "images": img_path
            }

            print(f"[OK] Linked {object_id}")

    with open(output_path, "w") as f:
        json.dump(final_map, f, indent=4)

    print(f"\n[SAVED] Final unified dataset map → {output_path}")

def move_images_and_build_full_relations(move_source_path: str, move_dest_path: str, pointcloud_root: str, relation_output_path: str):
    image_map = move_images_and_build_image_relations(move_source_path, move_dest_path)
    merge_with_pointclouds(
        image_map=image_map,
        pointcloud_root=pointcloud_root,
        output_path=relation_output_path,
    )
