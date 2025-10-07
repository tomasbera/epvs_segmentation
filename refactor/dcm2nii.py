import subprocess
import os
import shutil

base_folder = "dataset/sth/"
output_base = "dataset/nii_output/"
os.makedirs(output_base, exist_ok=True)

for subject in os.listdir(base_folder):
    subject_path = os.path.join(base_folder, subject)
    if not os.path.isdir(subject_path):
        continue  # skip files

    dicom_folder = os.path.join(subject_path, "2_Sag_3D_MPRAGE")
    if not os.path.isdir(dicom_folder):
        print(f"Skipping {subject}, folder not found: {dicom_folder}")
        continue

    temp_output = os.path.join(output_base, "temp")
    os.makedirs(temp_output, exist_ok=True)

    cmd = [
        "dcm2niix",
        "-z", "y",          # compress to .nii.gz
        "-o", temp_output,  # temporary output
        dicom_folder
    ]
    print(f"Converting {dicom_folder}")
    subprocess.run(cmd, check=True)

    nii_files = [f for f in os.listdir(temp_output) if f.endswith(".nii.gz")]
    if len(nii_files) == 0:
        print(f"No NIfTI files found for {subject}")
        continue

    nii_file = nii_files[0]
    src = os.path.join(temp_output, nii_file)
    dst = os.path.join(output_base, f"{subject}_T1w.nii.gz")  # consistent naming
    shutil.move(src, dst)
    print(f"Saved: {dst}")

    for f in os.listdir(temp_output):
        os.remove(os.path.join(temp_output, f))

print("All subjects converted.")