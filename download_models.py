"""
Downloads the OpenVINO (Intel Open Model Zoo) IR models used by this project
into models/. Run once after installing requirements.

    python download_models.py
"""
import os
import urllib.request

import config

BASE_URL = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1"

MODEL_NAMES = [
    "face-detection-retail-0004",
    "landmarks-regression-retail-0009",
    "face-reidentification-retail-0095",
]


def download(url, dest):
    if os.path.exists(dest):
        print(f"  already have: {os.path.basename(dest)}")
        return
    print(f"  downloading:  {os.path.basename(dest)}")
    urllib.request.urlretrieve(url, dest)


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    for name in MODEL_NAMES:
        print(f"{name}:")
        xml_dest = os.path.join(config.MODELS_DIR, f"{name}.xml")
        bin_dest = os.path.join(config.MODELS_DIR, f"{name}.bin")
        download(f"{BASE_URL}/{name}/FP32/{name}.xml", xml_dest)
        download(f"{BASE_URL}/{name}/FP32/{name}.bin", bin_dest)

    print("\nAll models ready.")


if __name__ == "__main__":
    main()
