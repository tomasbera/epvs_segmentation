import numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage import exposure
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import binary_dilation, generate_binary_structure


class Preprocessing:
    def __init__(
        self,
        image: np.ndarray,
        voxel_size=(0.8, 0.8, 0.8),
        t2w=True,
        s_min=0.5,
        s_max=3.0,
        num_scales=5,
        alpha=0.5,
        beta=0.5,
        gamma=1.0,
        t1=0.2,
        t2=0.5
    ):
        self.image = image
        self.voxel_size = voxel_size
        self.t2w = t2w

        self.s_min = s_min
        self.s_max = s_max
        self.num_scales = num_scales
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.t1 = t1
        self.t2 = t2



    # ---------------- I/O ----------------
    def load_nifti(self, path):
        nii = nib.load(path)
        arr = nii.get_fdata(dtype=np.float32)
        return arr, nii.affine, nii.header

    def save_nifti(self, arr, affine, header, outpath, dtype=np.float32):
        arr = np.asarray(arr, dtype=dtype)
        nii = nib.Nifti1Image(arr, affine, header=header)
        nib.save(nii, outpath)



    # ---------------- N4 Bias Correction ----------------
    def n4_bias_correction(self, mask, shrink_factor=2):
        sitk_image = sitk.GetImageFromArray(self.image.astype(np.float32))
        mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        mask.CopyInformation(sitk_image)

        shrink_factors = [shrink_factor] * sitk_image.GetDimension()
        image_small = sitk.Shrink(sitk_image, shrink_factors)
        mask_small = sitk.Shrink(mask, shrink_factors)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.Execute(image_small, mask_small)

        log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)
        corrected = sitk_image / sitk.Exp(log_bias_field)

        return corrected



    # ---------------- CLAHE ----------------
    def clahe_3d(self, image, mask, kernel_size=8, clip_limit=0.01, nbins=256):
        image_copy = image.copy()
        outside = ~mask
        if np.any(mask):
            fill = np.median(image[mask])
        else:
            fill = 0.0
        image_copy[outside] = fill
        out = exposure.equalize_adapthist(
            image_copy,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins
        )

        out[outside] = image[outside]
        return out.astype(np.float32)



    # ---------------- Frangi Vesselness ----------------
    def frangi_thresholding(self, vesselness):
        strong = vesselness >= self.t2
        weak = (vesselness >= self.t1) & (vesselness < self.t2)

        struct = generate_binary_structure(3, 3)
        dilated = binary_dilation(strong, structure=struct, iterations=2)

        hysteresis = strong | (weak & dilated)
        return hysteresis.astype(np.uint8)


    def frangi_3d(self, image):
        vesselness = np.zeros_like(self.image)
        scales = np.linspace(self.s_min, self.s_max, self.num_scales)
        scales_vox = [sigma / 0.8 for sigma in scales]

        for sigma in scales_vox:
            H = hessian_matrix(image, sigma=sigma, order='rc', use_gaussian_derivatives=False)
            l1, l2, l3 = hessian_matrix_eigvals(H)

            L = np.stack((l1, l2, l3), axis=0)
            idx = np.argsort(np.abs(L), axis=0)
            L_sorted = np.take_along_axis(L, idx, axis=0)
            l1s, l2s, l3s = L_sorted[0], L_sorted[1], L_sorted[2]

            if self.t2w:
                mask = (l2s < 0) & (l3s < 0)
            else:
                mask = (l2s > 0) & (l3s > 0)

            eps = np.finfo(np.float64).eps
            Ra = np.abs(l2s) / (np.abs(l3s) + eps)
            Rb = np.abs(l1s) / (np.sqrt(np.abs(l2s * l3s)) + eps)
            S = np.sqrt(l1s**2 + l2s**2 + l3s**2)

            a2, b2, c2 = self.alpha**2, self.beta**2, self.gamma**2

            termA = 1 - np.exp(-(Ra**2) / (2 * a2))
            termB = np.exp(-(Rb**2) / (2 * b2))
            termC = 1 - np.exp(-(S**2) / (2 * c2))

            V = termA * termB * termC
            V[~mask] = 0.0
            vesselness = np.maximum(vesselness, V)

        vmax = vesselness.max()
        if vmax > 0:
            vesselness /= vmax

        if self.t1 is not None and self.t2 is not None:
            binary = self.frangi_thresholding(vesselness)
            return vesselness, binary

        return vesselness
