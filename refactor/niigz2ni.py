import nibabel as nib
from pathlib import Path

# Root folder containing all your subjects
root_dir = Path("../dataset/braindata")

# Find all .nii.gz files recursively
nii_gz_files = list(root_dir.rglob("*.nii.gz"))

print(f"Found {len(nii_gz_files)} files.")

for gz_file in nii_gz_files:
    # Load the compressed NIfTI
    img = nib.load(str(gz_file))

    # Save as uncompressed .nii in the same folder
    nii_file = gz_file.with_suffix("")  # removes .gz
    nib.save(img, str(nii_file))

    print(f"Converted {gz_file} → {nii_file}")