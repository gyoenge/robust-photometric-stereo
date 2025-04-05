import numpy as np
import cv2

def relight_object(normal_map, albedo_map, light_dir, mask):
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
    # TODO: Fill this functions
    # Important Notes: Consider Channels 

    # Note : albedo_map.shape (3872, 5808), normal_map.shape (3872, 5808, 3), light_dir.shape (3,) 

    print("| Relighting.. ")
    relit_img = albedo_map * np.clip((normal_map @ light_dir) , 0, 1)
    
    relit_img *= mask 

    return relit_img
