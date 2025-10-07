import subprocess
from pathlib import Path

root_dir = Path("../dataset/opennero")

for nii_file in root_dir.rglob("*_T2w.nii"):
    mnc_file = nii_file.with_suffix(".mnc")

    cmd = [
        "docker", "run", "-v",
        f"{root_dir.parent.absolute()}:/data",
        "mcin/civet:2.1.1",
        "nii2mnc",
        f"/data/{nii_file.relative_to(root_dir.parent).as_posix()}",
        f"/data/{mnc_file.relative_to(root_dir.parent).as_posix()}"
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd)