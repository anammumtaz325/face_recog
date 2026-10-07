"""Real-time face recognition from the webcam using OpenVINO. Press 'q' to quit."""
import time

import cv2
import numpy as np

import config
from face_engine import get_engine, cosine_distance
from utils import load_encodings


def main():
    known_encodings, known_names = load_encodings()
    if not known_encodings:
        print("No encodings found. Run register.py first.")

    engine = get_engine()

    cam = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cam.isOpened():
        print("Could not open camera.")
        return

    prev_time = time.time()

    while True:
        ok, frame = cam.read()
        if not ok:
            break

        for box in engine.detect_faces(frame):
            name = "Unknown"
            embedding = engine.get_embedding(frame, box)
            if embedding is not None and known_encodings:
                distances = [cosine_distance(embedding, known) for known in known_encodings]
                best = int(np.argmin(distances))
                if distances[best] <= config.MATCH_THRESHOLD:
                    name = known_names[best]

            x1, y1, x2, y2 = box
            color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Face Recognition (OpenVINO)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
