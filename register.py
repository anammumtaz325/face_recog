"""
Build encodings.pickle from everything in dataset/.

    python register.py                     Encode the current dataset
    python register.py --capture "Ali"      Capture webcam photos for a new
                                             person, then encode the dataset
"""
import argparse
import os
import time

import cv2

import config
from utils import save_encodings, encode_image, scan_dataset


def capture_images(name, count):
    folder = os.path.join(config.DATASET_DIR, name)
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        print("Could not open camera.")
        return

    print(f"Capturing {count} images for '{name}'  —  's' = capture, 'q' = stop")
    saved = 0
    while saved < count:
        ok, frame = cam.read()
        if not ok:
            break

        display = frame.copy()
        cv2.putText(display, f"{saved}/{count}   s=capture  q=quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Register", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            path = os.path.join(folder, f"{name}_{existing + saved + 1}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
            time.sleep(0.3)
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


def build_encodings():
    """Rebuilds encodings.pickle from dataset/. Returns (num_people, num_encodings)."""
    people = scan_dataset()
    if not people:
        print(f"No images found in {config.DATASET_DIR}")
        return 0, 0

    all_encodings, all_names = [], []
    for name, image_paths in people.items():
        count = 0
        for path in image_paths:
            encs = encode_image(path)
            if not encs:
                print(f"  [skip] no face found: {os.path.basename(path)}")
                continue
            all_encodings.append(encs[0])
            all_names.append(name)
            count += 1
        print(f"{name}: {count} image(s) encoded")

    save_encodings(all_encodings, all_names)
    print(f"\nSaved {len(all_names)} encodings for {len(people)} people -> {config.ENCODINGS_FILE}")
    return len(people), len(all_names)


def main():
    parser = argparse.ArgumentParser(description="Register faces and build encodings.pickle")
    parser.add_argument("--capture", metavar="NAME", help="Capture webcam photos for this person first")
    parser.add_argument("--count", type=int, default=config.NUM_IMAGES_TO_CAPTURE)
    args = parser.parse_args()

    if args.capture:
        capture_images(args.capture, args.count)

    build_encodings()


if __name__ == "__main__":
    main()
