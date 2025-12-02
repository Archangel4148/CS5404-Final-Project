import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from distortion import distort_image
from evaluation import evaluate_pointcloud
from loading_things import load_dataset_relations, load_ply_pointcloud
from parse_results import parse_results
from paths import fix_path

SPAR3D_DIR = fix_path(Path("/mnt/c/Users/joshu/PycharmProjects/CS5404-Final-Project/stable-point-aware-3d"))


def run_spar3d_reconstruction(image_path: Path, output_file_path: Path) -> np.ndarray:
    """Run SPAR3D on the provided image, and output the resulting point cloud"""
    # Create a temporary folder for SPAR3D output
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Run the SPAR3D command
        cmd = [
            "python", str((SPAR3D_DIR / "run.py").resolve()), str(image_path),
            "--output-dir", str(tmpdir_path)
        ]
        print(f"SPAR3D: Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # (SPAR3D creates a folder named "0/" in the output directory)
        output_subfolder = tmpdir_path / "0"
        ply_files = list(output_subfolder.glob("*.ply"))
        if not ply_files:
            raise RuntimeError(f"No .ply file generated for {image_path}")

        # Move the .ply file to the output folder (it persists after the test)
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(ply_files[0]), str(output_file_path))
        print(f"SPAR3D: Saved point cloud to {output_file_path}")

    # Load the point cloud as numpy array
    pts = load_ply_pointcloud(output_file_path)
    return pts


def process_one_object(object_id: str, img_dir: Path, gt_pointcloud_path: Path, distortion_levels: list[dict],
                       taus: list[float], images_per_object: int = 1, keep_distorted: bool = False,
                       output_root: Path | None = None) -> dict:
    """
    Runs the full testing pipeline:
        1. distort images for each distortion level
        2. run SPAR3D on each distortion
        3. evaluate results at each tolerance (tau)
        4. collect and return results
    """
    results = dict(object_id=object_id, images=[])

    # Load ground truth
    gt_pts = load_ply_pointcloud(gt_pointcloud_path)

    # Load the database images to be processed
    images = sorted(list(Path(img_dir).glob("*.png")))[:images_per_object]

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

        for distortion in distortion_levels:
            blur, noise, exposure = distortion["blur"], distortion["noise"], distortion["exposure"]
            print(f"\nImage {i}, Distortion: blur={blur}, noise={noise}, exposure={exposure}")

            # Make a temporary folder for this distortion
            distort_dir = (output_root / object_id / f"blur{blur}_noise{noise}_exp{exposure}")
            distort_dir.mkdir(parents=True, exist_ok=True)

            # Distort the image
            img = Image.open(img_path).convert("RGB")
            img = distort_image(img, blur=blur, noise=noise, exposure=exposure)

            # Save the distorted image to the directory
            dist_path = distort_dir / f"img_{i}.png"
            img.save(dist_path)

            # Run SPAR3D on the distorted image
            out_ply = output_root / object_id / f"pts_img{i}_blur{blur}_noise{noise}_exp{exposure}.ply"
            start_time = time.perf_counter()
            pred_pts = run_spar3d_reconstruction(dist_path, output_file_path=out_ply)
            spar3d_time = time.perf_counter() - start_time
            print(f"SPAR3D: Finished in {spar3d_time:.2f} seconds")

            # Update results for this distortion
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
                print(f"Removing folder: {distort_dir}")
                shutil.rmtree(distort_dir, ignore_errors=True)

        results["images"].append(img_results)

    return results


def save_results_json(results: dict, path: Path):
    """Save the pipeline results to a json"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def reevaluate_results_with_folder(
        old_json_path: Path,
        new_json_path: Path,
        new_taus: list[float],
        spar3d_outputs_root: Path,
        object_relations_path: Path,
):
    """
    Re-run evaluations for new tau values using already-generated point clouds in spar3d_outputs.
    - Uses the previous JSON to know which objects/images/distortions exist
    - Finds PLY files in spar3d_outputs/<object_id>/
    """

    # Load old results JSON
    with open(old_json_path, "r") as f:
        data = json.load(f)

    for category_name, category_objs in data.items():
        for object_id, obj_data in category_objs.items():
            # Find the object folder in the spar3d_outputs folder
            object_folder = spar3d_outputs_root / object_id
            if not object_folder.exists():
                continue

            # Find the ground truth point cloud from the relations file
            grouped_data = load_dataset_relations(object_relations_path)
            gt_path = grouped_data[category_name][object_id]["point_cloud"]
            gt_points = load_ply_pointcloud(gt_path)

            for img_data in obj_data["images"]:
                for dist_data in img_data["distortions"]:
                    # Reconstruct the point cloud file name
                    pts_file_name = f"pts_img{img_data['image_idx']}_blur{dist_data['distortion']['blur']}_noise{dist_data['distortion']['noise']}_exp{dist_data['distortion']['exposure']}.ply"
                    points = load_ply_pointcloud(object_folder / pts_file_name)

                    # Clear old evaluations
                    dist_data["evaluations"] = []

                    # Recompute for each new tau
                    for tau in new_taus:
                        metrics = evaluate_pointcloud(points, gt_points, tau=tau)
                        dist_data["evaluations"].append({
                            "tau": tau,
                            "metrics": metrics
                        })

    # Save updated JSON
    new_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved updated results → {new_json_path}")


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    new_taus = [0.1, 0.2, 0.5]

    reevaluate_results_with_folder(
        old_json_path=Path(
            r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\datasets\omniobject3d\spar3d_outputs\pipeline_results_20251201_231844.json"),
        new_json_path=Path("pipeline_results_re-evaluated.json"),
        spar3d_outputs_root=Path(
            r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\datasets\omniobject3d\spar3d_outputs"),
        object_relations_path=Path(
            r"C:\Users\joshu\PycharmProjects\CS5404-Final-Project\datasets\omniobject3d\object_relations.json"),
        new_taus=new_taus
    )

    parse_results(
        json_path=Path("pipeline_results_re-evaluated.json"),
        csv_path=Path("parsed_results-re-evaluated.csv"),
    )
