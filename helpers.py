import numpy as np
import matplotlib.pyplot as plt
import torch
from ignite.utils import to_onehot
from ipywidgets import interact
import SimpleITK as sitk
import cv2
from monai.losses import DiceLoss
from monai.utils import first
from tqdm import tqdm


def explore_3D_data(arr, cmap: str = 'gray'):

    if isinstance(arr, sitk.Image):
        arr = sitk.GetArrayFromImage(arr)

    def fn(SLICE):
        plt.figure(figsize=(7,7))
        plt.imshow(arr[SLICE, :, :], cmap=cmap)
        plt.axis("off")
        plt.show()

    interact(fn, SLICE=(0, arr.shape[0]-1))





def explore_3D_data_comparison(arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = 'gray'):

    if isinstance(arr_before, sitk.Image) and isinstance(arr_after, sitk.Image):
        arr_before = sitk.GetArrayFromImage(arr_before)
        arr_after = sitk.GetArrayFromImage(arr_after)

    assert arr_after.shape == arr_before.shape

    def fn(SLICE):
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(10, 10))

        ax1.set_title('Before', fontsize=15)
        ax1.imshow(arr_before[SLICE, :, :], cmap=cmap)

        ax2.set_title('After', fontsize=15)
        ax2.imshow(arr_after[SLICE, :, :], cmap=cmap)

        plt.tight_layout()
        plt.show()

    interact(fn, SLICE=(0, arr_before.shape[0] - 1))





def rescale_linear(array: np.ndarray, new_min: int, new_max: int):
  minimum, maximum = np.min(array), np.max(array)
  m = (new_max - new_min) / (maximum - minimum)
  b = new_min - m * minimum
  return m * array + b





def explore_3D_data_with_mask(arr: np.ndarray, mask: np.ndarray, alpha: float = 0.4, cmap="jet"):
    assert arr.shape == mask.shape

    _arr = rescale_linear(arr, 0, 1)
    _mask = rescale_linear(mask, 0, 1)

    def fn(SLICE):
        plt.figure(figsize=(10, 10))
        plt.imshow(_arr[SLICE], cmap="gray")
        plt.imshow(_mask[SLICE], cmap=cmap, alpha=alpha)
        plt.axis("off")
        plt.show()

    interact(fn, SLICE=(0, arr.shape[0] - 1))





def show_transfer_data(data, SLICE_NUMBER=250, train=True, test=False):
    check_patient_train, check_patient_test = data

    view_train_patient = first(check_patient_train)
    view_test_patient = first(check_patient_test)

    if train:
        plt.figure("Visualization Train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"image {SLICE_NUMBER}")
        plt.imshow(view_train_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {SLICE_NUMBER}")
        plt.imshow(view_train_patient["label"][0, 0, :, :, SLICE_NUMBER])
        plt.show()

    if test:
        plt.figure("Visualization Test", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title(f"image {SLICE_NUMBER}")
        plt.imshow(view_test_patient["image"][0, 0, :, :, SLICE_NUMBER], cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(f"label {SLICE_NUMBER}")
        plt.imshow(view_test_patient["label"][0, 0, :, :, SLICE_NUMBER])
        plt.show()





def dice_val(pred, target):
    dice_val = DiceLoss(to_onehot_y = True, sigmoid = True, squared_pred=True)
    return 1 - dice_val(pred, target)





def calc_w(black_val, white_val):
    counts = np.array([black_val, white_val], dtype=float)
    w = 1 / (counts / counts.sum())
    w /= w.sum()
    return torch.tensor(w, dtype=torch.float32)




def calculate_pixels(data):
    val = np.zeros((1, 2))

    for batch in tqdm(data):
        batch_label = batch["label"] != 0
        _, count = np.unique(batch_label, return_counts=True)

        if len(count) == 1:
            count = np.append(count, 0)
        val += count

    print('The last values:', val)
    return val