import subprocess
import os
import tarfile


def download_files(wanted_files: list[str], out_dir, unzip=False):
    """Download selected files (expects wanted_files to be a list of file IDs)"""
    os.makedirs(out_dir, exist_ok=True)

    for i, file_id in enumerate(wanted_files):
        fname = f"image_{i+1}.tar.gz"
        out_path = os.path.join(out_dir, f"image_{i+1}.tar.gz")
        extract_dir = os.path.join(out_dir, f"image_{i+1}")

        if os.path.exists(out_path):
            print(f"[SKIP] {fname} already exists, skipping download.")
        else:
            print(f"[INFO] Downloading image_{i+1} ({file_id})...")
            cmd = ["gdown", file_id, "-O", out_path]

            proc = subprocess.run(cmd)
            if proc.returncode != 0:
                print(f"[ERROR] Failed to download {fname}")
            else:
                print(f"[OK] Saved {fname}")
        
        if unzip:
            if os.path.exists(extract_dir):
                print(f"[SKIP] Extract directory {extract_dir} already exists.")
            else:
                print(f"[INFO] Extracting into {extract_dir}...")
                os.makedirs(extract_dir, exist_ok=True)

                try:
                    with tarfile.open(out_path, "r:gz") as tar:
                        tar.extractall(path=extract_dir)
                    print(f"[OK] Extracted {fname}")
                except Exception as e:
                    print(f"[ERROR] Failed to extract {fname}: {e}")
