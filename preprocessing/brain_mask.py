import os
import glob
import subprocess

input_dir = "../dataset/braindata"

# Find all T2w images
t2_files = glob.glob(os.path.join(input_dir, "sub-*/anat/*_T2w.nii.gz"))

print(f"Found {len(t2_files)} T2w files to process.")

for idx, f in enumerate(t2_files, start=1):
    output_file = f.replace(".nii.gz", "_brain.nii.gz")

    cmd = [
        "hd-bet",
        "-i", f,
        "-o", output_file,
        "-device", "cpu",
        "--disable_tta"
    ]

    print(f"\n[{idx}/{len(t2_files)}] Processing file:")
    print(f" Input : {f}")
    print(f" Output: {output_file}")
    print(f" Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        if os.path.exists(output_file):
            print(f" ✅ Success: Output saved to {output_file}")
        else:
            print(f" ⚠️ Warning: Command ran but output not found for {f}")
    except subprocess.CalledProcessError as e:
        print(f" ❌ Error while processing {f}")
        print(f"    Error details: {e}")
