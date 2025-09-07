# Robust Photometric Stereo
## Computer Vision Course Coding Assignment 1 

- Course: GIST Computer Vision (EC4216)
- Project Type: Robust Photometric Stereo Implementation Individual Coding Assignment

### Overview 

<p align="justify">
In this project, we implemented a <b>Robust Photometric Stereo pipeline</b> to precisely estimate the 3D surface information of objects from multiple images. First, assuming that objects follow <b>Lambertian reflectance properties</b>, we implemented the traditional Photometric Stereo method based on <b>Least Squares</b>. Next, to effectively handle <b>non-Lambertian outliers</b> such as shadows and highlights that may occur in real-world environments, we applied <b>Robust Principal Component Analysis (RPCA)</b> to build a Robust Photometric Stereo approach.
</p>

<p align="justify">
Experiments were conducted on four objects (moai, nike, choonsik, toothless), photographed under <b>11 different lighting conditions with a fixed camera position</b>. Using the albedo maps and surface normals estimated by each method, we performed <b>relighting under novel lighting conditions</b> and quantitatively and qualitatively validated the results.
</p>

<p align="justify">
Additionally, to further improve relighting performance, we introduced a method that performs <b>Photometric Factor Classification</b> on the generated images and adjusts brightness by applying different weights to each factor. Compared to the traditional Least Squares-based approach, the Robust Photometric Stereo produced <b>smoother surface normals and reduced outliers in the albedo map</b>, and the photometric factor-based correction effectively reduced relighting errors.
</p>

<p align="center">
<img width="80%" alt="image" src="https://github.com/user-attachments/assets/76d739d2-6151-45bf-80b4-8f76d02c6d10" />
</p>

---

## Description

The provided code follows these main steps:
1. **Recover Light Directions:**  
   Use images of a chromeball to compute light directions.  
   *Fill the #todo blank in:* `recover_light_direction`

2. **Compute Normal Map and Albedo (Least Squares):**  
   For each object, extract pixel intensities under different lighting conditions and solve a least squares problem to estimate the surface normal map and albedo.  
   *Fill the #todo blank in:* `solve_least_squares`

3. **Robust Principal Component Analysis (RPCA):**  
   Apply the Iterative Augmented Lagrangian Multiplier (IALM) method to decompose the intensity matrix into low-rank and sparse components for improved robustness.  
   *Fill the #todo blank in:* `ialm`

5. **Relighting:**  
   Generate a relit image of the object using the given light direction, the normal map, and the albedo.  
   *Fill the #todo blank in* `relight_object`

### Directory Structure
```
. 
├── input/                  # Input dataset directory 
│ ├── chromeball/           # 11 Chromeball images and 1 mask image
│ ├── choonsik/             # 11 Object images and 1 mask image + 1 unknown image
│ ├── toothless/            # 11 Object images and 1 mask image + 1 unknown image
│ ├── nike/                 # 11 Object images and 1 mask image + 1 unknown image
│ └── moai/                 # 11 Object images and 1 mask image + 1 unknown image
├── output/                 # Output directory (created automatically) to save results 
├── main.py                 # Main Python script (the provided code) 
├── recover_lightdir.py     # Module to implement recover_light_direction 
├── least_squares.py        # Module to implement solve_least_squares 
├── ialm.py                 # Module to implement ialm 
└── relight.py              # Module to implement relight_object 
```

### Requirements

- **Python Version:** 3.6 or above
- **Libraries:**  
  - OpenCV (`cv2`)
  - NumPy
  - argparse
  - tqdm

You can install the required libraries using pip:

```
pip install opencv-python numpy tqdm
```

### Usage
Run the main script from the command line. The script accepts the following

arguments:

- `-d` or `--dataset_root`: Path to the dataset directory (default: ./PA1_dataset)
- `-o` or `--object`: Object to process (default: all). If set to a specific object name (e.g., choonsik), only that object will be processed.
- `-i` or `--image_cnt`: Number of images to use (default: 11)


### Implementation Steps

1. Light Direction Recovery:
   - Loads chromeball images and the corresponding mask.
   - Calls recover_light_direction to compute the light directions.
   - Saves the computed light directions to output/light_dirs.npy.

2. Processing Each Object:
   - For each specified object (or all objects if all is selected), the script:
   - Loads the object’s images and mask.
   - Extracts pixel intensities from the images where the mask is active.
   - Calls solve_least_squares with the intensity matrix and recovered light directions to compute the normal map and albedo.
   - Saves the resulting normal map and albedo images under the object's output directory.

3. RPCA Using IALM:
   - Decomposes the intensity matrix using the Augmented Lagrangian Multiplier method.
   - Recomputes the normal map and albedo from the low-rank approximation.
   - Saves the RPCA results.

4. Unknown Lighting and Relighting:
   - Loads an unknown image with a following lighting direction (given).
   - Calls relight_object to produce a relit image of the object.
   - Computes the Mean Squared Error (MSE) between the unknown image and the relit image.
   - Saves the relit image and the MSE value.

### Output
The script generates the following outputs for each object:

- Light Direction: Saved as light_dir.npy.
- Normal Map: Saved as normal_map.png under both ls and rpca directories.
- Albedo Map: Saved as albedo_map.png under both ls and rpca directories.
- Unknown Image: Saved as unknown_image.png.
- Relit Image: Saved as relit_image.png.
- MSE Value: Saved as mse.txt.
  
All outputs are saved in the output/ directory.
