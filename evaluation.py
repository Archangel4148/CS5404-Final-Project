import os

import torch
import numpy as np
from loading_things import load_ply_pointcloud


def to_tensor(x, device=None):
    """Converts numpy → torch tensor safely."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.float()
    if device is not None:
        x = x.to(device)

    return x


def chamfer_distance(pcl_a, pcl_b, device="cuda"):
    """
    Chamfer Distance between two point clouds.

    pcl_a: [N, 3]
    pcl_b: [M, 3]
    device: "cuda" or "cpu"
    """

    # Move to tensor + device
    pcl_a = to_tensor(pcl_a, device)
    pcl_b = to_tensor(pcl_b, device)

    # Compute pairwise distances
    # Efficient batched squared L2 norms
    diff = pcl_a.unsqueeze(1) - pcl_b.unsqueeze(0)
    dist = torch.sum(diff ** 2, dim=2)

    # CD = mean(min(dist(a→b))) + mean(min(dist(b→a)))
    cd_ab = torch.mean(torch.min(dist, dim=1)[0])
    cd_ba = torch.mean(torch.min(dist, dim=0)[0])

    return cd_ab + cd_ba


def fscore(pcl_a, pcl_b, tau=0.01, device="cuda"):
    """
    Computes the F-score between two point clouds:
    Precision, Recall, and F-score.
    """

    pcl_a = to_tensor(pcl_a, device)
    pcl_b = to_tensor(pcl_b, device)

    diff = pcl_a.unsqueeze(1) - pcl_b.unsqueeze(0)
    dist = torch.sqrt(torch.sum(diff ** 2, dim=2))

    # Precision: fraction of predicted points close to GT
    precision = (torch.min(dist, dim=1)[0] < tau).float().mean()

    # Recall: fraction of GT points close to predicted
    recall = (torch.min(dist, dim=0)[0] < tau).float().mean()

    if precision + recall == 0:
        f = torch.tensor(0.0, device=device)
    else:
        f = 2 * precision * recall / (precision + recall)

    return precision.item(), recall.item(), f.item()


def normalize_points(pts):
    pts = pts - pts.mean(axis=0)   # center at origin
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / scale
    return pts

def evaluate_pointcloud(pred_pts, gt_pts, tau=0.01):
    """
    Master evaluation function with GPU→CPU fallback.
    pred_pts: numpy array, shape [N, 3]
    gt_pts:   numpy array, shape [M, 3]
    """

    # Try CUDA first
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize point clouds (match coordinate systems/alignment)
    pred_pts, gt_pts = normalize_points(pred_pts), normalize_points(gt_pts)

    try:
        cd = chamfer_distance(pred_pts, gt_pts, device=device)
        prec, rec, f = fscore(pred_pts, gt_pts, tau=tau, device=device)
        return {
            "chamfer_distance": float(cd),
            "precision": prec,
            "recall": rec,
            "fscore": f,
            "device_used": device,
        }

    except RuntimeError:
        # Fallback to CPU if CUDA OOM or unavailable
        print("[WARN] CUDA failed — falling back to CPU")

        device = "cpu"
        cd = chamfer_distance(pred_pts, gt_pts, device=device)
        prec, rec, f = fscore(pred_pts, gt_pts, tau=tau, device=device)

        return {
            "chamfer_distance": float(cd),
            "precision": prec,
            "recall": rec,
            "fscore": f,
            "device_used": device,
        }



if __name__ == "__main__":
    TESTING_FOLDER = os.path.join(os.getcwd(), "examples_for_testing\\")
    GROUND_TRUTH_PATH = TESTING_FOLDER + r"db_ball_013_points.ply"
    SPAR3D_PATH = TESTING_FOLDER + r"spar3d_ball_013_points.ply"
    
    # Load ground truth (~16k points)
    gt = load_ply_pointcloud(GROUND_TRUTH_PATH)

    # Load SPAR3D output (512 pts)
    pred = load_ply_pointcloud(SPAR3D_PATH)

    # Evaluate the reconstruction at each threshold
    taus_to_test = [0.1, 0.2, 0.5]
    for tau in taus_to_test:
        results = evaluate_pointcloud(pred, gt, tau)
        print(f"Results (tau = {tau}): {results}")

    