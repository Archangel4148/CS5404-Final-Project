import os
import shutil
import json
from collections import defaultdict
from pathlib import Path


def move_images_and_build_image_relations(source_root: Path, dest_root: Path) -> dict[str, str]:
    """Moves images from source_root to dest_root and organizes/names them for processing."""
    os.makedirs(dest_root, exist_ok=True)
    relations = {}

    for img_folder in source_root.iterdir():
        # Don't try to process the zipped folders
        if not img_folder.is_dir():
            continue

        print(f"Handling folder: {img_folder.name}")

        for object_folder in img_folder.iterdir():
            # Don't try to process the alignment files
            if not object_folder.is_dir():
                continue

            # Create the new object folder
            object_id = object_folder.name
            category = object_id.split("_")[0]
            final_dir = dest_root / category / object_id
            final_dir.mkdir(parents=True, exist_ok=True)

            # Handle the inconsistent database file structure
            nested_images_dir = object_folder / "render" / "images"
            if nested_images_dir.is_dir():
                pngs = list(nested_images_dir.glob("*.png"))
            else:
                pngs = list(object_folder.glob("*.png"))

            if not pngs:
                print(f"WARNING: No PNGs found for {object_id}")
                continue

            # Move the images and update the relation
            for png in pngs:
                shutil.move(str(png), final_dir / png.name)

            relations[object_id] = str(final_dir)
            print(f"Moved images for {object_id} → {final_dir}")

        # Delete extracted folder (we keep the .tar.gz files)
        shutil.rmtree(img_folder, ignore_errors=True)
        print(f"Cleaning up folder: {img_folder.name}")

    return relations


def merge_with_pointclouds(image_map, pointcloud_root, output_path):
    """Build the relations between image_id, point_cloud_path, and image_path"""
    pointcloud_root = Path(pointcloud_root)
    final_map = defaultdict(dict)
    for object_id, img_path in image_map.items():
        # Get the category and paths
        category = "_".join(object_id.split("_")[:-1])
        category_dir = pointcloud_root / category
        object_dir = category_dir / object_id

        if not object_dir.is_dir():
            print(f"WARNING: No point cloud directory for {object_id}")
            continue

        ply_files = list(object_dir.glob("*.ply"))
        if not ply_files:
            print(f"WARNING: No .ply files for {object_id}")
            continue

        final_map[category][object_id] = {
            "point_cloud": str(ply_files[0]),
            "images": img_path
        }
        print(f"Linked {object_id}")

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(final_map, f, indent=4)
    print(f"\nFinal dataset map saved to {output_path}")


def move_images_and_build_full_relations(move_source_path: Path, move_dest_path: Path, pointcloud_root: Path,
                                         relation_output_path: Path):
    """Get the image map and build the full mapping"""
    image_map = move_images_and_build_image_relations(move_source_path, move_dest_path)
    merge_with_pointclouds(
        image_map=image_map,
        pointcloud_root=pointcloud_root,
        output_path=relation_output_path,
    )
