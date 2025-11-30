import torch
from loading_things import load_dataset_relations, load_ply_pointcloud, load_item_ids_from_drive_list
from download_drive_images import download_files
from restructure_files import move_images_and_build_full_relations

# Required inputs
BASE_DB_PATH = r"D:\Programming\CS5404-Final-Project\datasets\omniobject3d"
DRIVE_LINK_LIST_PATH = f"D:\Programming\CS5404-Final-Project\drive_file_ids.txt"
DESIRED_FILE_COUNT = 3  # How many item files (tar.gzip) to download

# Other constructed paths
RAW_IMAGES_PATH = BASE_DB_PATH + r"\images_raw"
IMAGES_PATH = BASE_DB_PATH + r"\images"
POINTCLOUD_ROOT = BASE_DB_PATH + r"\ply_16384\extracted\16384"
FINAL_RELATION_FILE_PATH = BASE_DB_PATH + r"\object_relations.json"

# Test code for PyTorch/GPU
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# Read desired file IDs
urls = open(DRIVE_LINK_LIST_PATH, "r").readlines()

# Download selected ones
# wanted_files = item_ids[:DESIRED_FILE_COUNT]
wanted_files = [urls[5]]
download_files(wanted_files, RAW_IMAGES_PATH, unzip=True)

# Move files and build relations
move_images_and_build_full_relations(
    move_source_path=RAW_IMAGES_PATH,
    move_dest_path=IMAGES_PATH,
    pointcloud_root=POINTCLOUD_ROOT,
    relation_output_path=FINAL_RELATION_FILE_PATH
)

# Load relations
ids, plys, imgs = load_dataset_relations()
print("Loaded:", len(ids), "objects")

# Load the first point cloud file
pts = load_ply_pointcloud(plys[0])
print(pts.shape)  # Should be (16384, 3)