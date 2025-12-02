import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from distortion import distort_image
from evaluation import evaluate_pointcloud
from loading_things import load_ply_pointcloud
from paths import fix_path

SPAR3D_DIR = fix_path(Path("/mnt/c/Users/joshu/PycharmProjects/CS5404-Final-Project/stable-point-aware-3d"))


def run_spar3d_reconstruction(image_path: Path, output_file_path: Path) -> np.ndarray:
    """Run SPAR3D on the provided image, and output the resulting point cloud"""

    # Create a temporary folder for SPAR3D output
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cmd = [
            "python", str((SPAR3D_DIR / "run.py").resolve()), str(image_path),
            "--output-dir", str(tmpdir_path)
        ]
        print(f"[SPAR3D] Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # SPAR3D creates a folder (named '0') in the output directory
        output_subfolder = tmpdir_path / "0"
        ply_files = list(output_subfolder.glob("*.ply"))
        if not ply_files:
            raise RuntimeError(f"No .ply file generated for {image_path}")

        # Move the .ply file to the desired location
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ply_files[0]), str(output_file_path))
        print(f"[SPAR3D] Saved .ply → {output_file_path}")

    # Load the point cloud as numpy array
    pts = load_ply_pointcloud(output_file_path)
    return pts


def process_one_object(object_id: str, img_dir: Path, gt_pointcloud_path: Path, distortion_levels: list[dict],
                       taus: list[float], images_per_object: int = 1, keep_distorted: bool = False,
                       output_root: Path | None = None) -> dict:
    """
    Runs the full testing pipeline:
        1. distort images for each distortion level
        2. run SPAR3D
        3. evaluate results
    """

    results = dict(object_id=object_id, images=[])

    # Load ground truth
    gt_pts = load_ply_pointcloud(gt_pointcloud_path)

    # Load the first {images_per_object} database images
    images = sorted(list(Path(img_dir).glob("*.png")))[:images_per_object]

    # Add the original image (no distortion)
    base_level = dict(blur=0.0, noise=0.0, exposure=1.0)
    all_levels = [base_level] + distortion_levels

    if output_root is None:
        output_root = Path("spar3d_outputs")

    # Ensure the output folder exists
    output_root = Path(output_root)
    (output_root / object_id).mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images):

        img_results = dict(
            image_idx=i,
            original_image=str(img_path),
            distortions=[],
        )

        for distortion in all_levels:
            blur, noise, exposure = distortion["blur"], distortion["noise"], distortion["exposure"]
            print(f"\n[INFO] Image {i}, Distortion: blur={blur}, noise={noise}, exposure={exposure}")

            # Make a temporary folder for this distortion
            distort_dir = (output_root / object_id / f"blur{blur}_noise{noise}_exp{exposure}")
            distort_dir.mkdir(parents=True, exist_ok=True)

            # Distort the image
            img = Image.open(img_path).convert("RGB")
            img = distort_image(img, blur=blur, noise=noise, exposure=exposure)

            # Save the distorted image to the temp directory
            dist_path = distort_dir / f"img_{i}.png"
            img.save(dist_path)

            # Run SPAR3D on the distorted image
            out_ply = output_root / object_id / f"pts_img{i}_blur{blur}_noise{noise}_exp{exposure}.ply"
            pred_pts = run_spar3d_reconstruction(dist_path, output_file_path=out_ply)

            img_results["distortions"].append(dict(
                distorted_image=str(dist_path),
                distortion=dict(blur=blur, noise=noise, exposure=exposure),
                evaluations=[],
            ))

            # Evaluate and save results for each tau
            for tau in taus:
                metrics = evaluate_pointcloud(pred_pts, gt_pts, tau=tau)
                img_results["distortions"][-1]["evaluations"].append(dict(
                    tau=tau,
                    metrics=metrics,
                ))

            # Clean up the distorted image folder
            if not keep_distorted:
                print(f"[CLEANUP] Removing folder: {distort_dir}")
                shutil.rmtree(distort_dir, ignore_errors=True)

        results["images"].append(img_results)

    return results


def save_results_json(results: dict, path: Path):
    """Save the pipeline results to a json"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
