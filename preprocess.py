import os
import glob
import helpers


from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,Spacingd,
    ScaleIntensityRanged, Resized, ToTensord, Orientationd, CropForegroundd
)

from monai.data import Dataset, DataLoader, CacheDataset
from monai.utils import set_determinism



def run_preprocessing(base_data_path="dataset/braindata/*/anat/*_T2w_brain.nii.gz",
                      mask_base_path="dataset/binary_epvs_groundtruth/mask/"):
    set_determinism(seed=999999)

    # =========================
    # Prepare dataset
    # =========================
    # Base paths
    image_pattern = "*/*_T2w_brain.nii.gz"

    data = []
    # Search for images
    for img_path in glob.glob(os.path.join(base_data_path, image_pattern)):
        subject_id = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        mask_path = os.path.join(mask_base_path, subject_id, f"{subject_id}_desc-mask_PVS.nii.gz")

        print(img_path)
        print(mask_path)

        if os.path.exists(mask_path):
            data.append({
                "image": img_path,
                "label": mask_path
            })
        else:
            print(f"Warning: Mask not found for {subject_id}")

    print(f"Loaded {len(data)} image-mask pairs.")

    # Split train/test
    cache = False
    train_size = int(0.8 * len(data))
    train_data, val_data = data[:train_size], data[train_size:]

    # =========================
    # MONAI transforms
    # =========================
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        ScaleIntensityRanged(keys=["image"], a_min=200, a_max=750, b_min=0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=[208, 256, 320]),
        ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 0.8), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS", labels=None),
        ScaleIntensityRanged(keys=["image"], a_min=200, a_max=750, b_min=0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=[208, 256, 320]),
        ToTensord(keys=["image", "label"]),
    ])

    if cache:
        train_ds = CacheDataset(data=train_data, transform=train_transforms, cache_rate=1.0)
        train_loader = DataLoader(train_ds, batch_size=1)

        val_ds = CacheDataset(data=val_data, transform=val_transforms, cache_rate=1.0)
        val_loader = DataLoader(val_ds, batch_size=1)


    else:
        train_ds = Dataset(data=train_data, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=1)

        val_ds = Dataset(data=val_data, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1)

    helpers.show_transfer_data((train_loader, val_loader))

    return train_loader, val_loader