import os

import torch
import numpy as np
from loading_things import load_ply_pointcloud


def to_tensor(x: np.ndarray, device=None):
    """Converts numpy array to a torch tensor."""
    x = torch.from_numpy(x)
    x = x.float()
    # If there's a device, convert to that device's format
    if device is not None:
        x = x.to(device)
    return x


def chamfer_distance(pcl_a: np.ndarray, pcl_b: np.ndarray, device) -> float:
    """Calculate Chamfer Distance between two point clouds using the provided device."""
    # Convert to tensors
    pcl_a = to_tensor(pcl_a, device)
    pcl_b = to_tensor(pcl_b, device)

    # Compute pairwise distances
    diff = pcl_a.unsqueeze(1) - pcl_b.unsqueeze(0)
    dist = torch.sum(diff ** 2, dim=2)

    # CD = mean(min(dist(a→b))) + mean(min(dist(b→a)))
    cd_ab = torch.mean(torch.min(dist, dim=1)[0])
    cd_ba = torch.mean(torch.min(dist, dim=0)[0])

    return float(cd_ab + cd_ba)


def fscore(pcl_a, pcl_b, tau=0.01, device="cuda") -> tuple[float, float, float]:
    """Computes precision, recall, and F-score between two point clouds."""
    # Convert to tensors
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


def normalize_points(pts: np.ndarray) -> np.ndarray:
    """Normalize points from a point cloud (for consistent positioning)."""
    pts = pts - pts.mean(axis=0)   # center at origin
    scale = np.max(np.linalg.norm(pts, axis=1))
    pts = pts / scale
    return pts

def evaluate_pointcloud(pred_pts: np.ndarray, gt_pts: np.ndarray, tau=0.01) -> dict:
    """Main evaluation function; calculates metrics between pred_pts and gt_pts."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize point clouds to match the coordinates
    pred_pts, gt_pts = normalize_points(pred_pts), normalize_points(gt_pts)

    try:
        # Calculate metrics
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
        # Fallback to CPU if something goes wrong
        print("WARNING: Evaluation failed, falling back to CPU")

        # Re-calculate metrics
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
    GROUND_TRUTH_PATH = TESTING_FOLDER + r"db_ball_013.ply"
    SPAR3D_PATH = TESTING_FOLDER + r"spar3d_ball_013.ply"
    
    # Load ground truth
    gt = load_ply_pointcloud(GROUND_TRUTH_PATH)

    # Load SPAR3D output
    pred = load_ply_pointcloud(SPAR3D_PATH)

    # Evaluate the reconstruction at each threshold
    taus_to_test = [0.01, 0.05, 0.075, 0.1, 0.2, 0.5]
    for tau in taus_to_test:
        results = evaluate_pointcloud(pred, gt, tau)
        print(f"Results (tau = {tau}): {results}")

    