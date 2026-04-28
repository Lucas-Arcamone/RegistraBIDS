import subprocess
from pathlib import Path

def register_to_template(fixed,
                         moving, 
                         out_prefix, 
                         init_transform=None, 
                         log_file=None):
    fixed = str(fixed)
    moving = str(moving)
    out_prefix = str(out_prefix)

    warped = f"{out_prefix}_warped.nii.gz"
    inv_warped = f"{out_prefix}_inv_warped.nii.gz"
    log_file = Path(log_file) if log_file else Path(f"{out_prefix}.log")
    
    cmd = [
        "antsRegistration",
        "-d", "3",
        "-o", f"[{out_prefix},{warped},{inv_warped}]",
    ]

    if init_transform is not None:
        cmd += ["-r", str(init_transform)]

    cmd += [
        "-m", f"MI[{fixed},{moving},1,32,Regular,0.2]",
        "-t", "Rigid[1]",
        "-c", "200x100x100",
        "-s", "4x2x1",
        "-f", "4x2x1",

        "-m", f"MI[{fixed},{moving},1,32,Regular,0.2]",
        "-t", "Affine[1]",
        "-c", "200x100x100",
        "-s", "4x2x1",
        "-f", "4x2x1",

        "-m", f"Mattes[{fixed},{moving},1,32,Regular,0.6]",
        "-t", "SyN[0.5,3,1]",
        "-c", "200x100x100",
        "-s", "4x2x2",
        "-f", "4x2x2",

        "--use-histogram-matching",
        "--verbose", "1",
    ]

    print("Running ANTs registration:")
    print(" ".join(cmd))

    with open(log_file, "w") as f:
        subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True
        )



def register_to_reference(fixed, 
                          moving, 
                          out_prefix, 
                          log_file=None):
    fixed = str(fixed)
    moving = str(moving)
    out_prefix = str(out_prefix)

    warped = f"{out_prefix}_warped.nii.gz"
    inv_warped = f"{out_prefix}_inv_warped.nii.gz"

    log_file = Path(log_file) if log_file else Path(f"{out_prefix}.log")

    cmd = [
        "antsRegistration",
        "-d", "3",
        "-o", f"[{out_prefix},{warped},{inv_warped}]",
    ]

    cmd += [
        "-m", f"MI[{fixed},{moving},1,32,Regular,0.2]",
        "-t", "Rigid[1]",
        "-c", "200x100x100",
        "-s", "4x2x1",
        "-f", "4x2x1",

        "-m", f"MI[{fixed},{moving},1,32,Regular,0.2]",
        "-t", "Affine[1]",
        "-c", "200x100x100",
        "-s", "4x2x1",
        "-f", "4x2x1",

        "--use-histogram-matching",
        "--verbose", "1",
    ]

    print("Running ANTs registration:")
    print(" ".join(cmd))

    with open(log_file, "w") as f:
        subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True
        )


