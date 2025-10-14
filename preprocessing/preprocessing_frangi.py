import os
import itertools
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import generate_binary_structure, binary_dilation
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
import pandas as pd


class Frangi3DCanonical:
    def __init__(self, image, spacing=(1.0, 1.0, 1.0), alpha=0.5, beta=0.5, gamma=1.0,
                 black_vessels=True, s_min=0.5, s_max=3.0, num_scales=5,
                 t1=None, t2=None):

        self.image = image.astype(np.float64)
        self.spacing = spacing
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_vessels = black_vessels
        self.s_min = s_min
        self.s_max = s_max
        self.num_scales = num_scales
        self.t1 = t1
        self.t2 = t2

    def compute(self):
        vesselness = np.zeros_like(self.image, dtype=np.float64)
        scales = np.linspace(self.s_min, self.s_max, self.num_scales)
        scales_vox = [sigma / min(self.spacing) for sigma in scales]

        for sigma in scales_vox:
            # Hessian & eigenvalues
            H = hessian_matrix(self.image, sigma=sigma, order='rc', use_gaussian_derivatives=False)
            l1, l2, l3 = hessian_matrix_eigvals(H)

            # Sort eigenvalues by absolute magnitude
            L = np.stack((l1, l2, l3), axis=0)
            idx = np.argsort(np.abs(L), axis=0)
            L_sorted = np.take_along_axis(L, idx, axis=0)
            l1s, l2s, l3s = L_sorted[0], L_sorted[1], L_sorted[2]

            # Sign condition
            if self.black_vessels:
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
            binary = self._hysteresis_thresholding(vesselness, self.t1, self.t2)
            return vesselness, binary

        return vesselness, None

    def _hysteresis_thresholding(self, vesselness, t1, t2):
        strong = vesselness >= t2
        weak = (vesselness >= t1) & (vesselness < t2)

        struct = generate_binary_structure(3, 3)
        dilated = binary_dilation(strong, structure=struct, iterations=2)

        hysteresis = strong | (weak & dilated)
        return hysteresis.astype(np.uint8)


def dice_score(pred, gt):
    """Compute Dice coefficient between prediction and ground truth (binary masks)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    return 2.0 * intersection / (pred.sum() + gt.sum() + 1e-8)


if __name__ == "__main__":
    # --- Load input image ---
    img = sitk.ReadImage("dataset/braindata/sub-001/anat/sub-001_T2w.nii.gz")
    img_data = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()[::-1]

    # Optional: load ground-truth mask for evaluation
    ground_truth_path = "../dataset/binary_epvs_groundtruth/mask/sub-001/sub-001_desc-mask_PVS.nii"
    gt_data = None
    if ground_truth_path and os.path.exists(ground_truth_path):
        gt_data = sitk.GetArrayFromImage(sitk.ReadImage(ground_truth_path))

    # --- Parameter ranges ---
    alphas = [0.5, 1.0]
    betas = [0.5, 1.0]
    gammas = [0.5, 1.0, 2.0]
    s_mins = [0.5, 1.0]
    s_maxs = [3.0, 4.0]
    thresholds = [(0.2, 0.5), (0.1, 0.4)]

    output_dir = "frangi_results"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    # --- Grid search ---
    for (alpha, beta, gamma, s_min, s_max, (t1, t2)) in itertools.product(
        alphas, betas, gammas, s_mins, s_maxs, thresholds
    ):
        print(f"Running α={alpha}, β={beta}, γ={gamma}, s=({s_min},{s_max}), t=({t1},{t2})")

        frangi = Frangi3DCanonical(
            img_data,
            spacing=spacing,
            s_min=s_min, s_max=s_max, num_scales=5,
            alpha=alpha, beta=beta, gamma=gamma,
            black_vessels=True,
            t1=t1, t2=t2
        )

        vesselness, binary = frangi.compute()

        # Save results
        tag = f"a{alpha}_b{beta}_g{gamma}_s{s_min}-{s_max}_t{t1}-{t2}"

        vesselness_img = sitk.GetImageFromArray(vesselness.astype("float32"))
        vesselness_img.CopyInformation(img)
        sitk.WriteImage(vesselness_img, os.path.join(output_dir, f"frangi_{tag}.nii.gz"))

        if binary is not None:
            binary_img = sitk.GetImageFromArray(binary.astype("uint8"))
            binary_img.CopyInformation(img)
            sitk.WriteImage(binary_img, os.path.join(output_dir, f"frangi_{tag}_binary.nii.gz"))

            # If GT available, compute Dice
            dice = None
            if gt_data is not None:
                dice = dice_score(binary, gt_data)
                print(f" -> Dice score: {dice:.4f}")

            results.append({
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "s_min": s_min,
                "s_max": s_max,
                "t1": t1,
                "t2": t2,
                "dice": dice
            })

    # Save evaluation table
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "frangi_results.csv"), index=False)
        print("Results saved to frangi_results.csv")
