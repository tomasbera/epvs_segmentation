import numpy as np
import nibabel as nib
from skimage import exposure
from matplotlib import pyplot as plt


def load_nifti(path):
    nii = nib.load(path)
    arr = nii.get_fdata(dtype=np.float32)
    return arr, nii.affine, nii.header


def save_nifti(arr, affine, header, outpath, dtype=np.float32):
    arr = np.asarray(arr, dtype=dtype)
    new_nii = nib.Nifti1Image(arr, affine, header=header)
    nib.save(new_nii, outpath)


def clahe_3d(image, mask, kernel_size=8, clip_limit=0.01, nbins=256):
    print("=== CLAHE Debug ===")
    print("Input image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Mask sum (number of voxels inside):", np.sum(mask))

    # Pick central slice for visualization
    cz = image.shape[2] // 2

    # Original image
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 4, 1)
    plt.imshow(image[:, :, cz], cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # Fill outside mask
    image_copy = image.copy()
    outside = ~mask
    if np.any(mask):
        fill = np.median(image[mask])
    else:
        fill = 0.0
    image_copy[outside] = fill

    plt.subplot(1, 4, 2)
    plt.imshow(image_copy[:, :, cz], cmap='gray')
    plt.title('Fill outside-mask')
    plt.axis('off')

    # Apply CLAHE
    out = exposure.equalize_adapthist(
        image_copy,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins
    )
    plt.subplot(1, 4, 3)
    plt.imshow(out[:, :, cz], cmap='gray')
    plt.title('After CLAHE')
    plt.axis('off')

    # Restore outside-mask
    out[outside] = image[outside]
    plt.subplot(1, 4, 4)
    plt.imshow(out[:, :, cz], cmap='gray')
    plt.title('Final output')
    plt.axis('off')

    plt.show()

    # Print min/max for debug
    print("Image min/max before CLAHE:", np.min(image_copy), np.max(image_copy))
    print("Output min/max after CLAHE:", np.min(out), np.max(out))

    return out.astype(np.float32)

if __name__ == "__main__":
    in_nii = "dataset/opennero/sub-001/anat/sub-001_T2w.nii.gz"
    mask_nii = "dataset/opennero/sub-001/anat/sub-001_T2w_brain.nii.gz"
    out_nii = "dataset/opennero/sub-001/anat/sub-001_T2w_brainhist.nii.gz"

    # load input and mask
    vol, affine, header = load_nifti(in_nii)
    mask, _, _ = load_nifti(mask_nii)
    mask = mask > 0.0   # ensure boolean mask
    cz = vol.shape[2] // 2
    plt.figure(figsize=(16, 4))

    # normalize intensities (optional, but often helpful)
    p99 = np.percentile(vol[mask], 99)
    if p99 > 0:
        vol = vol / p99

    # normalize inside mask to [0,1] for CLAHE
    vol_norm = np.zeros_like(vol, dtype=np.float32)
    vol_norm[mask] = (vol[mask] - vol[mask].min()) / (vol[mask].max() - vol[mask].min())

    out = clahe_3d(vol_norm, mask, kernel_size=8, clip_limit=0.02)
    plt.subplot(1, 4, 4)
    plt.imshow(out[:, :, cz], cmap='gray')
    plt.title('Final output')
    plt.axis('off')

    plt.show()

    # save as float32 NIfTI (keeps [0,1] range)
    save_nifti(out, affine, header, out_nii, dtype=np.float32)
    print("Saved:", out_nii)
