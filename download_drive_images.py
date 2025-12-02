import os
import tarfile

import gdown

from paths import fix_path


def download_files(wanted_files: list[str], out_dir, unzip=False):
    """Download selected files (expects wanted_files to be a list of file IDs)"""
    os.makedirs(out_dir, exist_ok=True)

    for url in wanted_files:
        extract_name = url.replace("https://drive.google.com/file/d/", "")[:10]
        fname = extract_name + ".tar.gz"
        extract_dir = fix_path(os.path.join(out_dir, extract_name))
        out_path = os.path.join(out_dir, fname)

        if os.path.exists(out_path):
            print(f"[SKIP] {fname} already exists, skipping download.")
        else:
            print(f"[INFO] Downloading {fname} ({url})...")
            gdown.download(url=url, output=out_path, fuzzy=True)

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
