import numpy as np
import time
from tqdm import tqdm
import cv2

def shrink(X, tau):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)
     

def ialm(D, image_mask,threshold=1e-7): # 1e-3 
    """
    Performs low-rank and sparse decomposition of the input matrix D using
    the Inexact Augmented Lagrange Multiplier (IALM) method.
    
    The algorithm decomposes D into:
        D = A + E
    where A is a low-rank matrix and E is a sparse error matrix.
    
    Parameters:
        D (np.ndarray): The input data matrix of shape (m, n).
        threshold (float): Convergence threshold based on the residual change.
        
    Returns:
        A (np.ndarray): The recovered low-rank component.
        E (np.ndarray): The recovered sparse error component.
        iterations (int): The number of iterations executed.
    """
    m, n = D.shape
    iterations = 0
    Y = np.zeros((m, n))
    A = np.zeros((m, n))
    E = np.zeros((m, n))

    # TODO: Fill this functions
    # Important Notes: You can use any functions in numpy to implement svd!
    # Recommend you to use frobenius norm in numpy

    # hyperparameters 
    mu = 1.25 / np.linalg.norm(D, ord=2) # (weak) 1.0 
    mu_bar = mu * 1e7 
    rho = 1.6 # (weak) 1.1
    lam = float(1 / np.sqrt(m))
    print(f"| rpca) mu_0: {mu}, lambda: {lam}")

    mu_flag = True
    
    Y = D / max(np.linalg.norm(D, ord=2), (1/lam)*np.linalg.norm(D, ord=np.inf))

    for iter in tqdm(range(100), desc='| Solving rpca..'):
        iterations += 1 

        # update A
        U, S, Vh = np.linalg.svd(D - E + (1/mu)*Y, full_matrices=False) 
        S = shrink(S, 1/mu)
        A = U @ np.diag(S) @ Vh

        E_prev = E.copy()
        # update E
        E = shrink((D - A) + (1/mu)*Y, lam/mu)

        # update lagrange multiplier 
        Y = Y + mu * ((D-A-E))

        #############################################################################
        # visualize A / E / Y / D
        # rows, cols = np.where(image_mask == 1)
        # h, w = image_mask.shape
        # A_vis = np.zeros((h, w ))
        # E_vis = np.zeros((h, w ))
        # Y_vis = np.zeros((h, w ))
        # D_vis = np.zeros((h, w ))
        # for i in range(len(rows)):
        #     A_vis[rows[i], cols[i]] = max(A[i,0], 0.0)
        #     E_vis[rows[i], cols[i]] = E[i,0] # max(E[i,0], 0.0)
        #     Y_vis[rows[i], cols[i]] = Y[i,0] # max(Y[i,0], 0.0)
        #     D_vis[rows[i], cols[i]] = max(D[i,0], 0.0)
        # target_size = (int(300 * w / h), 300)
        # A_vis = cv2.resize((A_vis*255*image_mask).astype(np.uint8), target_size)
        # E_vis = cv2.resize((E_vis*255*image_mask).astype(np.uint8), target_size)
        # Y_vis = cv2.resize((Y_vis*255*image_mask).astype(np.uint8), target_size)
        # D_vis = cv2.resize((D_vis*255*image_mask).astype(np.uint8), target_size)
        # stacked_vis = np.vstack([A_vis, E_vis, Y_vis, D_vis])
        # cv2.imshow("RPCA - A / E / Y / D", stacked_vis)
        # if cv2.waitKey(1) == 27: # ESC
        #     break
        #############################################################################

        # update mu
        mu = min(mu * rho, mu_bar)
        if mu_flag and mu==mu_bar:
            print(f"mu fixed : {mu_bar}")
            mu_flag = False

        # check convergence 
        error1 = np.linalg.norm(D-A-E, ord='fro') / np.linalg.norm(D, ord='fro')
        error2 = np.linalg.norm(min(mu, np.sqrt(mu))*(E-E_prev), ord='fro') / np.linalg.norm(D, ord='fro')
        if iter>=10 and error1 < threshold and error2 < 1e-5:
            break 

    print(f"| rpca) A: {np.min(A)}~{np.max(A)}, E: {np.min(E)}~{np.max(E)}")

    return A, E, iterations

