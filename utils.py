"""Shared helpers for encoding storage and dataset scanning."""
import os
import pickle

from face_engine import get_engine

import config

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_encodings():
    if os.path.exists(config.ENCODINGS_FILE):
        with open(config.ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data.get("encodings", []), data.get("names", [])
    return [], []


def save_encodings(encodings, names):
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)


def encode_image(path):
    """Returns a list of 256-d embeddings, one per face found in the image."""
    return get_engine().encode_image_file(path)


def scan_dataset():
    """
    Maps person name -> list of image paths. Supports two layouts:
      dataset/Name.jpg      (single image, name = file stem)
      dataset/Name/*.jpg    (multiple images, name = folder name)
    """
    people = {}
    if not os.path.isdir(config.DATASET_DIR):
        return people

    for entry in sorted(os.listdir(config.DATASET_DIR)):
        full_path = os.path.join(config.DATASET_DIR, entry)

        if os.path.isdir(full_path):
            images = [
                os.path.join(full_path, f)
                for f in sorted(os.listdir(full_path))
                if f.lower().endswith(VALID_EXTENSIONS)
            ]
            if images:
                people[entry] = images
        elif entry.lower().endswith(VALID_EXTENSIONS):
            name = os.path.splitext(entry)[0]
            people.setdefault(name, []).append(full_path)

    return people
