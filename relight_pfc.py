import numpy as np
import cv2

def relight_pfc_object(A, I, relit_image, mask, thresholds=[0.1, 0.15]):
    """
    Generate relit image using estimated normal map and albedo.
    
    Parameters:
        normal_map (numpy.ndarray): Estimated surface normal map (HxWx3)
        albedo_map (numpy.ndarray): Estimated albedo map (HxW)
        light_dir (numpy.ndarray): New light direction (3,)
        mask (numpy.ndarray): Binary mask of valid pixels (HxW)
    
    Returns:
        relit_img (numpy.ndarray): Image relit under the new light direction (HxW)
    """
    h, w = mask.shape
    rows, cols = np.where(mask == 1)

    # pfc (photometric factor classification)
    threshold1, threshold2 = thresholds

    PFC_C = np.zeros((h, w, 3))
    PFC_A = np.zeros((h, w, 3))
    PFC_D = np.zeros((h, w, 3))
    PFC_S = np.zeros((h, w, 3))
    PFC_C_mask = np.zeros((h, w))
    PFC_A_mask = np.zeros((h, w))
    PFC_D_mask = np.zeros((h, w))
    PFC_S_mask = np.zeros((h, w))
    print(f"| PFC(Photometric Factor Classification)..")
    for i in range(len(rows)):
        if A[i]>0 and I[i]<=threshold1 : 
            PFC_C[rows[i], cols[i]] = [1, 1, 0] # yellow
            PFC_C_mask[rows[i], cols[i]] = 1
        if A[i]<=0 and I[i]<=threshold1 : 
            PFC_A[rows[i], cols[i]] = [0, 1, 0] # green
            PFC_A_mask[rows[i], cols[i]] = 1
        if abs(A[i]-I[i])<threshold2*I[i] and I[i]>threshold1 : 
            PFC_D[rows[i], cols[i]] = [0, 0, 1] # blue
            PFC_D_mask[rows[i], cols[i]] = 1
        if I[i]-A[i]>threshold2*I[i] and I[i]>threshold1 : 
            PFC_S[rows[i], cols[i]] = [1, 1, 1] # white
            PFC_S_mask[rows[i], cols[i]] = 1
    
    PFC = np.clip((PFC_C+PFC_A+PFC_D+PFC_S), 0, 1)
    img_mask_color = np.stack([mask]*3, axis=-1)
    PFC = cv2.cvtColor((PFC * img_mask_color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # relight 
    print(f"| PFC Relighting..")
    for i in range(len(rows)):
        if PFC_C_mask[rows[i],cols[i]] == 1: 
            # deep shadow 
            relit_image[rows[i],cols[i]] *= 0.1
        if PFC_A_mask[rows[i],cols[i]] == 1: 
            # shadow 
            relit_image[rows[i],cols[i]] *= 0.2
        if PFC_D_mask[rows[i],cols[i]] == 1: 
            # ordinary 
            relit_image[rows[i],cols[i]] *= 1.1
        if PFC_S_mask[rows[i],cols[i]] == 1: 
            # highlight 
            relit_image[rows[i],cols[i]] *= 1.3 
    
    relit_image *= mask 
    relit_image = np.clip(relit_image, 0, 1)

    return relit_image, PFC, PFC_C_mask, PFC_A_mask, PFC_D_mask, PFC_S_mask
