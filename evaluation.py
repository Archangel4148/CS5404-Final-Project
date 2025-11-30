import os
import random
import requests
import trimesh
import numpy as np


# ===============================
# CONFIG
# ===============================

OMNI_BASE = "https://omniobject3d.stanford.edu/download"
LOCAL_BASE = "datasets/omniobject3d"

# List of all OmniObject3D object categories
# (Complete official list, stable across releases)
OMNI_CLASSES = [
    "airplane", "apple", "banana", "basket", "bathtub", "bed", "bench",
    "bicycle", "bottle", "bowl", "bus", "cabinet", "camera", "can", "cap",
    "car", "chair", "clock", "cup", "desk", "door", "faucet", "guitar",
    "helmet", "jar", "keyboard", "knife", "laptop", "microwave", "motorbike",
    "mug", "pencil", "piano", "plate", "pot", "remote", "scissors",
    "stapler", "stove", "table", "teapot", "toaster", "toilet", "train",
    "vase", "washing_machine"
]

# Each class has models numbered 0001–00XX
MAX_MODELS_PER_CLASS = 50   # safe upper bound


# ===============================
# DOWNLOAD FUNCTION
# ===============================

def download_omni_object(category: str, model_id: int, out_dir=LOCAL_BASE):
    """
    Download a single OmniObject3D mesh.
    Returns path to model.obj or None.
    """

    model_name = f"{category}_{model_id:04d}"
    dest_dir = os.path.join(out_dir, model_name)
    mesh_path = os.path.join(dest_dir, "model.obj")

    # Skip if already cached
    if os.path.exists(mesh_path):
        return mesh_path

    os.makedirs(dest_dir, exist_ok=True)

    # File URL
    url = f"{OMNI_BASE}/{category}/{model_name}.obj"

    print(f"Downloading: {model_name} ...")

    r = requests.get(url)
    if r.status_code != 200:
        print(f" ❌ Failed: {model_name}")
        return None

    with open(mesh_path, "wb") as f:
        f.write(r.content)

    print(f" ✔ Saved: {mesh_path}")
    return mesh_path


# ===============================
# LOADING + SAMPLING
# ===============================

def load_mesh_as_points(path, sample_points=100000):
    """Loads a mesh file and uniformly samples its surface."""
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [mesh.geometry[g] for g in mesh.geometry]
        )

    pts, _ = trimesh.sample.sample_surface(mesh, sample_points)
    return pts.astype(np.float32)


# ===============================
# MASTER PIPELINE
# ===============================

def get_random_omni_samples(n=5, points_per_mesh=50000):
    """
    Downloads n random OmniObject3D models and samples points from them.
    Returns list of numpy arrays.
    """
    pointclouds = []

    for _ in range(n):
        # Pick random category
        cat = random.choice(OMNI_CLASSES)

        # Pick random model ID (many classes have <50, but safe)
        model_id = random.randint(1, MAX_MODELS_PER_CLASS)

        mesh_path = download_omni_object(cat, model_id)

        if mesh_path is None:
            print("Skipping failed download.")
            continue

        try:
            pts = load_mesh_as_points(mesh_path, sample_points=points_per_mesh)
            pointclouds.append(pts)
        except Exception as e:
            print("Error loading mesh:", e)

    return pointclouds


pcs = get_random_omni_samples(n=10, points_per_mesh=20000)
print(len(pcs), "point clouds loaded.")
print("Shape of first:", pcs[0].shape)
