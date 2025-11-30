import subprocess
import os
import tarfile
import re
import gdown

def get_drive_filename(file_id: str) -> str:
    """
    Ask gdown for metadata and extract the original file name.
    """
    cmd = ["gdown", "--id", file_id, "--print"]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(proc.returncode)
        print("[WARN] Could not retrieve name, using ID as fallback.")
        return file_id + ".tar.gz"

    output = proc.stdout

    # Look for: file: something.tar.gz
    match = re.search(r"file:\s*(.+)", output)
    if match:
        return os.path.basename(match.group(1))

    print("[WARN] Could not find filename, fallback to ID.tar.gz")
    return file_id + ".tar.gz"


def download_files(wanted_files: list[str], out_dir, unzip=False):
    """Download selected files (expects wanted_files to be a list of file IDs)"""
    os.makedirs(out_dir, exist_ok=True)

    for url in wanted_files:
        # fname = get_drive_filename(file_id)
        fname = "image_ball.tar.gz"
        out_path = os.path.join(out_dir, fname)
        extract_dir = os.path.join(out_dir, fname.replace(".tar.gz", ""))

        if os.path.exists(out_path):
            print(f"[SKIP] {fname} already exists, skipping download.")
        else:
            print(f"[INFO] Downloading {fname} ({url})...")
            # cmd = ["gdown", file_id, "-O", out_path]
            gdown.download(url=url, output=out_path, fuzzy=True)

            # proc = subprocess.run(cmd)
            # if proc.returncode != 0:
            #     print(f"[ERROR] Failed to download {fname}")
            # else:
            #     print(f"[OK] Saved {fname}")
        
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
