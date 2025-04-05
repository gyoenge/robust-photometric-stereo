import numpy as np
from tqdm import tqdm
import cv2

def solve_least_squares(I, L, rows, cols, mask):
    """
    Least squares solution for normal estimation and albedo recovery.
    
    Parameters:
        imgn (list of str): List of image file paths.
        L (numpy.ndarray): Light directions (11, 3).
        rows (numpy.ndarray): Row indices of the object pixels.
        cols (numpy.ndarray): Column indices of the object pixels.

    Returns:
        normal (numpy.ndarray): Estimated surface normals.
        albedo (numpy.ndarray): Estimated albedo.
    """
    h, w = mask.shape
    normal = np.zeros((h, w, 3))
    albedo = np.zeros((h, w))
    # TODO: Fill this functions
    # Note: normal.shape (3872, 5808, 3), albedo.shape (3872, 5808)
    # Note: I.shape (1840977, 11)=(m,n), L.shape (11, 3)=(n,3)
    # G = I @ np.linalg.inv(L @ L.T) @ L 
    G = np.linalg.inv(L.T @ L) @ L.T @ I.T  # (3,m)
    # G = np.linalg.pinv(L) @ I.T
    G = G.T 
    gnorm = np.linalg.norm(G, ord=2, axis=1, keepdims=True)

    for i in tqdm(range(len(rows)), desc='| Solving least squares'):
        # TODO: Fill this functions
        # Important Notes: opencv uses BGR, not RGB   
        albedo[rows[i],cols[i]] = gnorm[i]
        normal[rows[i],cols[i]] = G[i] / gnorm[i]

    return normal, albedo

