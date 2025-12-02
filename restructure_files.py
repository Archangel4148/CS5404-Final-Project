import os
import shutil
import json
from collections import defaultdict
from pathlib import Path


def move_images_and_build_image_relations(source_root: Path, dest_root: Path):
    """
    Moves images into dest_root/category/object_id/*.png
    Handles both standard and nested layouts.
    Builds { object_id: image_folder } mapping.
    """
    os.makedirs(dest_root, exist_ok=True)
    relations = {}

    for img_folder in source_root.iterdir():
        if not img_folder.is_dir():
            continue

        print(f"[PROCESS] Handling folder: {img_folder.name}")

        for object_folder in img_folder.iterdir():
            if not object_folder.is_dir():
                continue

            object_id = object_folder.name
            category = object_id.split("_")[0]
            final_dir = dest_root / category / object_id
            final_dir.mkdir(parents=True, exist_ok=True)

            # Detect the nested "images/" layout
            nested_images_dir = object_folder / "render" / "images"
            if nested_images_dir.is_dir():
                pngs = list(nested_images_dir.glob("*.png"))
            else:
                pngs = list(object_folder.glob("*.png"))

            if not pngs:
                print(f"[WARN] No PNGs found for {object_id} in either layout")
                continue

            for png in pngs:
                shutil.move(str(png), final_dir / png.name)

            relations[object_id] = str(final_dir)
            print(f"[OK] Moved images for {object_id} → {final_dir}")

        # Cleanup: delete extracted folder but keep .tar.gz
        shutil.rmtree(img_folder, ignore_errors=True)
        print(f"[CLEANUP] Done with folder: {img_folder.name}")

    return relations



def merge_with_pointclouds(image_map, pointcloud_root, output_path):
    pointcloud_root = Path(pointcloud_root)
    final_map = defaultdict(dict)

    for object_id, img_path in image_map.items():
        # Derive category from object_id, e.g., "anise_001" -> "anise"
        category = "_".join(object_id.split("_")[:-1])
        category_dir = pointcloud_root / category
        object_dir = category_dir / object_id

        if not object_dir.is_dir():
            print(f"[WARN] No point cloud directory for {object_id}")
            continue

        ply_files = list(object_dir.glob("*.ply"))
        if not ply_files:
            print(f"[WARN] No .ply files for {object_id}")
            continue

        final_map[category][object_id] = {
            "point_cloud": str(ply_files[0]),
            "images": img_path
        }
        print(f"[OK] Linked {object_id}")

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(final_map, f, indent=4)

    print(f"\n[SAVED] Final unified dataset map → {output_path}")


def move_images_and_build_full_relations(move_source_path: Path, move_dest_path: Path, pointcloud_root: Path,
                                         relation_output_path: Path):
    image_map = move_images_and_build_image_relations(move_source_path, move_dest_path)
    merge_with_pointclouds(
        image_map=image_map,
        pointcloud_root=pointcloud_root,
        output_path=relation_output_path,
    )
