import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from recover_lightdir import recover_light_direction
from least_squares import solve_least_squares
from rpca import ialm
from relight import relight_object
from relight_pfc import relight_pfc_object

def args_parser():
    parser = argparse.ArgumentParser(description='PA1: Photometric stereo')
    parser.add_argument('-d', '--dataset_root', type=str, default='./input')
    parser.add_argument('-o', '--object', type=str, default='all')
    parser.add_argument('-i', '--image_cnt', type=int, default=12)

    args = parser.parse_args()
    return args

def compute_mse(original, estimated):
    return np.mean((original - estimated) ** 2)

def main():
    #############################################################################
    ############################## Step0: Settings ##############################
    #############################################################################
    # Parse command-line arguments
    args = args_parser()
    if args.object=='all':
        objects = ['choonsik', 'toothless', 'nike', 'moai']

    else:
        objects = [args.object]  # fixed 
        
    # Make output dirs
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    #############################################################################
    #################### Step1: Recovery for Light Direction ####################
    #############################################################################
    if os.path.isfile(f"{output_dir}/light_dirs.npy"):
        L = np.load("./output/light_dirs.npy")
    else:    
        chromeballs = [f"{args.dataset_root}/chromeball/{i}.jpg" for i in range(1, args.image_cnt)]
        chromeball_mask = cv2.imread(f"{args.dataset_root}/chromeball/mask.bmp", cv2.IMREAD_GRAYSCALE) / 255.0
            
        # Recover Light Directions
        L = recover_light_direction(chromeballs, chromeball_mask)
        np.save(f"{output_dir}/light_dirs.npy", L)
    
    for obj in objects:
        print(f"### processing {obj} " + '#'*(50-len(obj)))
        obj_dir = os.path.join(output_dir, obj)
        os.makedirs(obj_dir, exist_ok=True)
        
        input_dir = os.path.join(args.dataset_root, obj)
        
        # Load Data
        images = [f"{input_dir}/{i}.jpg" for i in range(1, args.image_cnt)]
        image_mask = cv2.imread(f"{input_dir}/mask.bmp", cv2.IMREAD_GRAYSCALE) / 255.0
        
        #############################################################################
        ########################### Step2: Least Squares ############################
        #############################################################################
        print("LS")
    
        rows, cols = np.where(image_mask == 1)

        I = np.zeros((len(rows), args.image_cnt - 1))

        for i, img_path in enumerate(tqdm(images, desc='| Load images')):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) / 255.0
            for j in range(len(rows)):
                I[j, i] = img[rows[j], cols[j]]

        normal, albedo = solve_least_squares(I, L, rows, cols, image_mask)

        os.makedirs(f"{output_dir}/{obj}/ls", exist_ok=True)
        # cv2.imwrite(f"{output_dir}/{obj}/ls/normal_map.png", (normal * 255).astype(np.uint8))  
        # fixed for visualization 
        img_mask_color = np.stack([image_mask]*3, axis=-1)
        vis_normal = cv2.cvtColor(((normal+1)/2 * img_mask_color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/{obj}/ls/normal_map.png", vis_normal)  
        cv2.imwrite(f"{output_dir}/{obj}/ls/albedo_map.png", (np.clip(albedo, 0, 1) * 255 * image_mask).astype(np.uint8))

        # save normal, albedo
        np.save(f"{output_dir}/{obj}/ls/normal.npy", normal)
        np.save(f"{output_dir}/{obj}/ls/albedo.npy", albedo)

        #### relight mse save seperately #######################################################
        ground_truth = cv2.imread(f"{input_dir}/unknown.jpg", cv2.IMREAD_GRAYSCALE) / 255.0
        unknown_light_dir = np.load(f"{args.dataset_root}/unknown_light_dir.npy")
        relit_image = relight_object(normal, albedo, unknown_light_dir, image_mask)
        cv2.imwrite(f"{output_dir}/{obj}/relit_image_ls.png", (relit_image * 255).astype(np.uint8))
        mse = compute_mse(ground_truth, relit_image)
        with open(f"{obj_dir}/mse_ls.txt", "w") as f:
            f.write(f"mse: {mse}\n")
        print(f"{obj}: mse (ls)= {mse}")

        #############################################################################
        ########################### Step3: RPCA via IALM ############################
        #############################################################################
        print("RPCA")

        A_hat, E_hat, iter = ialm(I, image_mask )

        #### [check] visualize I & A_hat & E_hat ###############################################
        h, w = image_mask.shape
        for j in range(0, args.image_cnt-1): 
            I_img = np.zeros((h, w))
            A_hat_img = np.zeros((h, w)) 
            E_hat_img = np.zeros((h, w))
            IminusA_img = np.zeros((h, w))
            I_img[rows, cols] = I[:, 0]
            A_hat_img[rows, cols] = A_hat[:, 0]
            E_hat_img[rows, cols] = E_hat[:, 0]
            IminusA_img[rows, cols] = I[:, 0] - A_hat[:, 0]
            A_hat_img_cliped = np.clip(A_hat_img, 0, 1)
            IminusA_img = np.clip(IminusA_img, 0, 1) # (IminusA_img - np.min(IminusA_img)) 
            IminusA_img_magnified = (IminusA_img - np.min(IminusA_img)) / (np.max(IminusA_img) - np.min(IminusA_img) + 1e-8)
            os.makedirs(f"{output_dir}/{obj}/rpcaiae/{j+1}", exist_ok=True)
            cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/{j+1}/I.png", (I_img * 255 * image_mask).astype(np.uint8))
            cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/{j+1}/A_hat.png", (A_hat_img * 255 * image_mask).astype(np.uint8))
            cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/{j+1}/A_hat_clipped.png", (A_hat_img_cliped * 255 * image_mask).astype(np.uint8))
            cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/{j+1}/E_hat.png", (E_hat_img * 255 * image_mask).astype(np.uint8))
            cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/{j+1}/IminusA.png", (IminusA_img * 255 * image_mask).astype(np.uint8))
            cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/{j+1}/IminusA_magnified.png", (IminusA_img_magnified * 255 * image_mask).astype(np.uint8))
            # save I, A, E numpy array
            np.save(f"{output_dir}/{obj}/rpcaiae/{j+1}/I.npy", I[:,j])
            np.save(f"{output_dir}/{obj}/rpcaiae/{j+1}/A.npy", A_hat[:,j])
            np.save(f"{output_dir}/{obj}/rpcaiae/{j+1}/E.npy", E_hat[:,j])
        ######################################################################################### 

        # A_hat = np.clip(A_hat, 0, 1)
        normal, albedo = solve_least_squares(A_hat, L, rows, cols, image_mask)

        os.makedirs(f"{output_dir}/{obj}/rpca", exist_ok=True)
        # cv2.imwrite(f"{output_dir}/{obj}/rpca/normal_map.png", (normal * 255).astype(np.uint8))
        # fixed for visualization 
        img_mask_color = np.stack([image_mask]*3, axis=-1)
        vis_normal = cv2.cvtColor(((normal+1)/2 * img_mask_color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/{obj}/rpca/normal_map.png", vis_normal)  
        cv2.imwrite(f"{output_dir}/{obj}/rpca/albedo_map.png", (np.clip(albedo, 0, 1) * 255 * image_mask).astype(np.uint8))
        
        # save normal, albedo
        np.save(f"{output_dir}/{obj}/rpca/normal.npy", normal)
        np.save(f"{output_dir}/{obj}/rpca/albedo.npy", albedo)

        #############################################################################
        ############################# Step4: Relighting #############################
        #############################################################################

        ground_truth = cv2.imread(f"{input_dir}/unknown.jpg", cv2.IMREAD_GRAYSCALE) / 255.0
        cv2.imwrite(f"{output_dir}/{obj}/ground_truth.png", (ground_truth * image_mask * 255).astype(np.uint8))

        unknown_light_dir = np.load(f"{args.dataset_root}/unknown_light_dir.npy")

        relit_image = relight_object(normal, albedo, unknown_light_dir, image_mask)
        cv2.imwrite(f"{output_dir}/{obj}/relit_image.png", (relit_image * 255).astype(np.uint8))
        
        # Compute MSE
        mse = compute_mse(ground_truth, relit_image)
        with open(f"{obj_dir}/mse.txt", "w") as f:
            f.write(f"mse: {mse}\n")
        print(f"{obj}: mse = {mse}")

        #############################################################################
        ############################# (+): PFC Relighting ###########################
        #############################################################################
        print("PFC")

        # unknown rpca 
        I_unknown = np.zeros((len(rows), 1))
        for j in range(len(rows)):
            I_unknown[j, 0] = relit_image[rows[j], cols[j]]
        
        A_hat_unknown, E_hat_unknown, _ = ialm(I_unknown, image_mask)

        # save unknown rpca vis result 
        I_img[rows, cols] = I_unknown[:, 0]
        A_hat_img[rows, cols] = A_hat_unknown[:, 0]
        E_hat_img[rows, cols] = E_hat_unknown[:, 0]
        IminusA_img[rows, cols] = I_unknown[:, 0] - A_hat_unknown[:, 0]
        A_hat_img_cliped = np.clip(A_hat_img, 0, 1)
        IminusA_img = np.clip(IminusA_img, 0, 1) # (IminusA_img - np.min(IminusA_img)) 
        IminusA_img_magnified = (IminusA_img - np.min(IminusA_img)) / (np.max(IminusA_img) - np.min(IminusA_img) + 1e-8)
        os.makedirs(f"{output_dir}/{obj}/rpcaiae/unknown", exist_ok=True)
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/I.png", (I_img * 255 * image_mask).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/A_hat.png", (A_hat_img * 255 * image_mask).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/A_hat_clipped.png", (A_hat_img_cliped * 255 * image_mask).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/E_hat.png", (E_hat_img * 255 * image_mask).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/IminusA.png", (IminusA_img * 255 * image_mask).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/IminusA_magnified.png", (IminusA_img_magnified * 255 * image_mask).astype(np.uint8))
        # save I, A, E numpy array
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/I.npy", I[:,0])
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/A.npy", A_hat[:,0])
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/E.npy", E_hat[:,0])

        # pfc and relighting 
        if obj == 'toothless':
            thresholds = [0.05, 0.3]
        else: 
            thresholds = [0.1, 0.15]
        relit_pfc_image, PFC, PFC_C_mask, PFC_A_mask, PFC_D_mask, PFC_S_mask = relight_pfc_object(
            A_hat_unknown, I_unknown, relit_image, image_mask, thresholds)
        cv2.imwrite(f"{output_dir}/{obj}/relit_image_pfc.png", (relit_pfc_image * 255).astype(np.uint8))

        # save PFC vis result
        cv2.imwrite(f"{output_dir}/{obj}/rpcaiae/unknown/PFC.png", PFC) 
        # save PFC npy 
        os.makedirs(f"{output_dir}/{obj}/rpcaiae/unknown/PFC", exist_ok=True)
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/PFC/PFC_C_mask.npy", PFC_C_mask)
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/PFC/PFC_A_mask.npy", PFC_A_mask)
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/PFC/PFC_D_mask.npy", PFC_D_mask)
        np.save(f"{output_dir}/{obj}/rpcaiae/unknown/PFC/PFC_S_mask.npy", PFC_S_mask)
        
        # Compute MSE
        mse = compute_mse(ground_truth, relit_pfc_image)
        with open(f"{obj_dir}/mse_pfc.txt", "w") as f:
            f.write(f"mse: {mse}\n")
        print(f"{obj}: mse (pfc) = {mse}")


        print("#"*(16+50))

if __name__ == "__main__":
    main()