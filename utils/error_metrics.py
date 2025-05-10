import numpy as np
from pyquaternion import Quaternion

def quaternion_absolute_distance(estimate, ground_truth , rotate_frame=False):
    """
    Computes the absolute distance between corresponding quaternions in two arrays
    using pyquaternion's absolute_distance method.

    Parameters:
        estimate (np.ndarray): n×4 array of estimated quaternions.
        ground_truth (np.ndarray): n×4 array of ground truth quaternions.

    Returns:
        np.ndarray: 1×n array of distances (in radians). NaN if input is invalid or contains NaNs.
    """
    estimate = np.asarray(estimate)
    ground_truth = np.asarray(ground_truth)

    if estimate.shape != ground_truth.shape or estimate.shape[1] != 4:
        raise ValueError("Both input arrays must be of shape (n, 4)")

    def compute_distance(pair):
        est_q, gt_q = pair[:4], pair[4:]
        if np.any(np.isnan(est_q)) or np.any(np.isnan(gt_q)):
            return 0.0
        q1 = Quaternion(est_q)
        q2 = Quaternion(gt_q)
        
        if rotate_frame:
            q1 = Quaternion(1/np.sqrt(2), 0, 0, 1/np.sqrt(2)) * q1 
        return np.rad2deg(Quaternion.absolute_distance(q1, q2))

    combined = np.hstack((estimate, ground_truth))
    distances = np.apply_along_axis(compute_distance, 1, combined)

    return distances
