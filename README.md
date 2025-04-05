# PA1: Photometric Stereo
This repository contains the code for a programming assignment on photometric stereo. The goal of this assignment is to recover the surface normals and albedo of an object from multiple images captured under different lighting conditions, and then to use this information to relight the object under an unknown lighting condition.

- Due: 5th, April, 2025
- TA Session: 1st, April, 2025 and 3rd, April, 2025
- TA: Wooseok Jeon (jws5271a@gm.gist.ac.kr)

**NO PLAGIARISM, NO DELAY, DON'T USE AI SUPPORTER (If you do not comply, you will get F!)**

## Overview

### Before you start, make sure you understand the code flow by reading main.py

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

## Directory Structure
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

## Requirements

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

## Usage
Run the main script from the command line. The script accepts the following

arguments:

- `-d` or `--dataset_root`: Path to the dataset directory (default: ./PA1_dataset)
- `-o` or `--object`: Object to process (default: all). If set to a specific object name (e.g., choonsik), only that object will be processed.
- `-i` or `--image_cnt`: Number of images to use (default: 11)


## Implementation Steps

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

## Output
The script generates the following outputs for each object:

- Light Direction: Saved as light_dir.npy.
- Normal Map: Saved as normal_map.png under both ls and rpca directories.
- Albedo Map: Saved as albedo_map.png under both ls and rpca directories.
- Unknown Image: Saved as unknown_image.png.
- Relit Image: Saved as relit_image.png.
- MSE Value: Saved as mse.txt.
  
All outputs are saved in the output/ directory.
