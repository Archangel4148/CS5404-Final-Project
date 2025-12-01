import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from distortion import distort_image
from evaluation import evaluate_pointcloud
from loading_things import load_ply_pointcloud


def run_spar3d_reconstruction(image_path: Path) -> np.ndarray:
    """Run SPAR3D on the provided image, and output the resulting point cloud"""
    # TODO: Actually do the SPAR3D pipeline.

    print(f"[PLACEHOLDER] Running SPAR3D on {image_path}")
    fake_pts = np.random.rand(512, 3) - 0.5
    return fake_pts


def process_one_object(object_id: str, img_dir: Path, gt_pointcloud_path: Path, distortion_levels: list[dict],
                       taus: list[float], images_per_object: int = 1, keep_distorted: bool = False,
                       output_root: Path | None = None) -> dict:
    """
    Runs the full testing pipeline:
        1. distort images for each distortion level
        2. run SPAR3D
        3. evaluate results
    """

    results = dict(object_id=object_id, evaluations=[])

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

    for distortion in all_levels:
        blur, noise, exposure = distortion["blur"], distortion["noise"], distortion["exposure"]

        print(f"\n[INFO] Distortion: blur={blur}, noise={noise}, exposure={exposure}")

        # Make a temporary folder for this distortion
        distort_dir = (output_root / object_id / f"blur{blur}_noise{noise}_exp{exposure}")
        distort_dir.mkdir(parents=True, exist_ok=True)

        distorted_paths = []
        for i, img_path in enumerate(images):
            # Distort the image
            img = Image.open(img_path).convert("RGB")
            img = distort_image(img, blur=blur, noise=noise, exposure=exposure)

            # Save the distorted image to the temp directory
            out_path = distort_dir / f"img_{i}.png"
            img.save(out_path)

            distorted_paths.append((img_path, out_path))

        print(f"[INFO] Saved {len(distorted_paths)} distorted images → {distort_dir}")

        # Run SPAR3D on each distorted image
        for idx, (orig_path, dist_path) in enumerate(distorted_paths):
            pred_pts = run_spar3d_reconstruction(dist_path)

            # Evaluate and save results for each tau
            for tau in taus:
                metrics = evaluate_pointcloud(pred_pts, gt_pts, tau=tau)
                results["evaluations"].append(dict(
                    image_idx=idx,
                    original_image=str(orig_path),
                    distorted_image=str(dist_path),
                    distortion=dict(blur=blur, noise=noise, exposure=exposure),
                    tau=tau,
                    metrics=metrics,
                ))

        # Clean up the distorted image folder (unless keeping it)
        if keep_distorted:
            print(f"[KEEP] Keeping distorted images: {distort_dir}")
        else:
            print(f"[CLEANUP] Removing folder: {distort_dir}")
            shutil.rmtree(distort_dir, ignore_errors=True)

    return results


def save_results_json(results: dict, path: Path):
    """Save the pipeline results to a json"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
