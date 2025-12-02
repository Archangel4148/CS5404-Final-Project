import os
import tarfile

import gdown

from paths import fix_path


def download_files(urls: list[str], out_dir, unzip=False):
    """Download selected Google Drive files to out_dir"""
    os.makedirs(out_dir, exist_ok=True)

    for url in urls:
        # Clean up the URL and build paths
        extract_name = url.replace("https://drive.google.com/file/d/", "")[:10]
        fname = extract_name + ".tar.gz"
        extract_dir = fix_path(os.path.join(out_dir, extract_name))
        out_path = os.path.join(out_dir, fname)

        # If the file doesn't exist, download it
        if os.path.exists(out_path):
            print(f"{fname} already exists, skipping download.")
        else:
            print(f"Downloading {fname} ({url})...")
            gdown.download(url=url, output=out_path, fuzzy=True)

        if unzip:
            # Unzip the downloaded file (if it exists)
            if os.path.exists(extract_dir):
                print(f"Extract directory {extract_dir} already exists, skipping.")
            else:
                print(f"Extracting into {extract_dir}...")
                os.makedirs(extract_dir, exist_ok=True)

                try:
                    # Extract the .tar file
                    with tarfile.open(out_path, "r:gz") as tar:
                        tar.extractall(path=extract_dir)
                    print(f"Extracted {fname}")
                except Exception as e:
                    print(f"ERROR: Failed to extract {fname}: {e}")
