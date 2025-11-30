import json
from pathlib import Path
from plyfile import PlyData
import numpy as np


RELATION_FILE = r"D:\Programming\CS5404-Final-Project\datasets\omniobject3d\object_relations.json"
def load_dataset_relations(json_path=RELATION_FILE):
    """
    Loads your dataset relation file and returns:
        - object_ids: list[str]
        - ply_paths: list[Path]
        - image_dirs: list[Path]
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    object_ids = []
    ply_paths = []
    image_dirs = []

    for obj_id, fields in data.items():
        pc_path = fields.get("point_cloud")
        img_dir = fields.get("images")

        if pc_path is None or img_dir is None:
            print(f"[WARN] Missing fields for {obj_id}, skipping")
            continue

        object_ids.append(obj_id)
        ply_paths.append(Path(pc_path))
        image_dirs.append(Path(img_dir))

    return object_ids, ply_paths, image_dirs


def load_ply_pointcloud(path):
    ply = PlyData.read(path)
    data = ply['vertex']
    points = np.vstack([data['x'], data['y'], data['z']]).T
    return points

def load_item_ids_from_drive_list(list_path: str):
    """Get a list of item IDs from the provided file (expects a list of Google Drive share links)"""
    with open(list_path, "r") as f:
        raw_paths = [line.strip() for line in f.readlines() if line.strip()]
        wanted_files = []
        for path in raw_paths[:3]:
            id = path.replace(r"https://drive.google.com/file/d/", "")
            id = id.replace(r"/view?usp=drive_link", "")
            print("Parsed File ID:", id)
            wanted_files.append(id)
    return wanted_files