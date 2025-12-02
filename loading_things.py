import json
from pathlib import Path
from plyfile import PlyData
import numpy as np

from paths import fix_path


def load_dataset_relations(json_path) -> dict[str: dict[str: Path]]:
    """
    Loads your dataset relation file and returns grouped data
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    grouped_data = {}

    for group_name, group_objs in data.items():
        grouped_data[group_name] = {}
        for obj_id, fields in group_objs.items():
            pc_path = fix_path(fields.get("point_cloud"))
            img_dir = fix_path(fields.get("images"))

            if pc_path is None or img_dir is None:
                print(f"[WARN] Missing fields for {obj_id}, skipping")
                continue

            grouped_data[group_name][obj_id] = {
                "point_cloud": Path(pc_path),
                "images": Path(img_dir)
            }

    return grouped_data


def load_ply_pointcloud(path):
    ply = PlyData.read(path)
    data = ply['vertex']
    points = np.vstack([data['x'], data['y'], data['z']]).T
    return points
