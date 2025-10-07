import os
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt

from master.helpers import explore_3D_data
from master.peprocessing import Preprocessing

# ================================
# Configuration
# ================================
ROOT_DIR = "../dataset"
OUTPUT_FILE = "vesselness_output.nii.gz"  # change as needed

# ================================
# Find T2-weighted image
# ================================
t2_file = None
mask_file = None

for root, dirs, files in os.walk(ROOT_DIR):
    print("Searching in:", root)
    for f in files:
        if f.endswith("_T2w.nii.gz"):
            t2_file = os.path.join(root, f)
            break
    if t2_file:
        break

for f in os.listdir(os.path.dirname(t2_file)):
    if f.endswith("_T2w_brain.nii.gz"):
        mask_file = os.path.join(os.path.dirname(t2_file), f)
        break


if (t2_file or mask_file) is None:
    raise FileNotFoundError("T2W image or Mask not found in dataset directory.")



# ================================
# Read and orient image and mask
# ================================
raw_img_sitk = sitk.ReadImage(t2_file)
raw_img_arr = sitk.GetArrayFromImage(raw_img_sitk)

raw_mask_sitk = sitk.ReadImage(mask_file)
raw_mask_sitk = sitk.DICOMOrient(raw_mask_sitk)
raw_mask_arr = sitk.GetArrayFromImage(raw_mask_sitk)

pre_processing = Preprocessing(
    raw_img_arr,
    voxel_size=raw_img_sitk.GetSpacing(),
    s_min=0.5, s_max=3.0, num_scales=5,
    alpha=0.5, beta=0.5, gamma=1.0,
    t2w=True,
    t1=0.2, t2=0.5
)


# ================================
# N4 bias field correction
# ================================
n4_img = pre_processing.n4_bias_correction(raw_mask_arr)
n4_arr = sitk.GetArrayFromImage(n4_img)
sitk.WriteImage(sitk.GetImageFromArray(n4_arr), "n4.nii.gz")


# ================================
# Clahe3D histogram equalization
# ================================
vol, affine, header = pre_processing.load_nifti("n4.nii.gz")
mask, _, _ = pre_processing.load_nifti(mask_file)
mask = mask > 0.0

p99 = np.percentile(vol[mask], 99)
if p99 > 0:
    vol = vol / p99

vol_norm = np.zeros_like(vol, dtype=np.float32)
vol_norm[mask] = (vol[mask] - vol[mask].min()) / (vol[mask].max() - vol[mask].min())
out = pre_processing.clahe_3d(vol_norm, mask)
pre_processing.save_nifti(out, affine, header, "clahe3d", dtype=np.float32)



# ================================
# Frangi vesselness filtering
# ================================
img = sitk.ReadImage("clahe3d.nii.gz")
img_data = sitk.GetArrayFromImage(img)
spacing = img.GetSpacing()[::-1]
vesselness, binary = pre_processing.frangi_3d(img_data)




# ================================
# Save results
# ================================
vesselness_img = sitk.GetImageFromArray(vesselness.astype(np.float32))
vesselness_img.SetSpacing(raw_img_sitk.GetSpacing())
vesselness_img.SetOrigin(raw_img_sitk.GetOrigin())
vesselness_img.SetDirection(raw_img_sitk.GetDirection())
sitk.WriteImage(vesselness_img, OUTPUT_FILE)

# Binary segmentation (optional)
binary_output = OUTPUT_FILE.replace(".nii.gz", "_binary.nii.gz")
binary_img = sitk.GetImageFromArray(binary.astype(np.uint8))
binary_img.SetSpacing(raw_img_sitk.GetSpacing())
binary_img.SetOrigin(raw_img_sitk.GetOrigin())
binary_img.SetDirection(raw_img_sitk.GetDirection())
sitk.WriteImage(binary_img, binary_output)

print("Vesselness image saved to:", OUTPUT_FILE)
print("Binary mask saved to:", binary_output)
