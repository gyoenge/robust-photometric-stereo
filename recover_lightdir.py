import numpy as np
from tqdm import tqdm
import cv2

def recover_light_direction(chromeballs, chromeball_mask):
    """
    Recovers the light direction from images of chromeballs by detecting the brightest spot on each ball.
    
    Parameters:
        chromeballs (list of str): List of file paths to chromeball images.
        chromeball_mask (np.ndarray): Binary mask indicating the region of the chromeball in the image.

    Returns:
        L (np.ndarray): Array of recovered light directions (N x 3), where N is the number of chromeball images.
    """
    rows, cols = np.where(chromeball_mask == 1)
    # TODO: Fill this functions
    # find radius 
    radius_x = (cols.max()-rows.min())/2
    radius_y = (rows.max()-cols.min())/2 
    radius = (radius_x+radius_y)/2
    # find center 
    center_x = np.mean(cols)
    center_y = np.mean(rows)
    center = [center_x, center_y]

    L = []
    for i, img_path in enumerate(tqdm(chromeballs, desc='Recover light direction')):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
        # TODO: Fill this functions
        # Important Notes: You can use any opencv functions to find the brightest point!
        img = img * chromeball_mask

        # find brightest point         
        brightest = np.max(img)
        brightest_points = np.where(img == brightest)
        brightest_point = [np.mean(brightest_points[1]), np.mean(brightest_points[0])]

        # find N
        Nx = brightest_point[0] - center[0]
        Ny = center[1] - brightest_point[1]
        Nz = np.sqrt(radius**2 - Nx**2 - Ny**2)
        Nx, Ny, Nz = Nx/radius, Ny/radius, Nz/radius
        N = np.array([Nx, Ny, Nz])
        # set R 
        R = np.array([0, 0, 1])
        # calculate L 
        Lxyz = 2 * (N @ R) * N - R
        Lxyz = Lxyz / np.linalg.norm(Lxyz)

        L.append(Lxyz)

    L = np.array(L)
    return L
