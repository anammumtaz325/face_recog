"""
OpenVINO-based face detection, alignment, and embedding.

Three Intel Open Model Zoo models (see download_models.py) are chained:
  face-detection-retail-0004        -> face bounding boxes
  landmarks-regression-retail-0009  -> 5-point landmarks (eyes, nose, mouth)
  face-reidentification-retail-0095 -> 256-d embedding of the aligned face

Two embeddings are compared with cosine distance (0 = identical, 2 = opposite).
"""
import cv2
import numpy as np
from openvino import Core

import config

# Reference landmark positions for a well-aligned face, normalized to [0, 1].
# From the face-reidentification-retail-0095 model documentation.
_REFERENCE_LANDMARKS = np.array([
    (30.2946 / 96, 51.6963 / 112),   # left eye
    (65.5318 / 96, 51.5014 / 112),   # right eye
    (48.0252 / 96, 71.7366 / 112),   # nose tip
    (33.5493 / 96, 92.3655 / 112),   # left mouth corner
    (62.7299 / 96, 92.2041 / 112),   # right mouth corner
], dtype=np.float64)


def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _similarity_transform(points_from, points_to):
    """Least-squares similarity transform (rotation + uniform scale + translation)
    that maps points_from onto points_to."""
    mean_from = points_from.mean(axis=0)
    mean_to = points_to.mean(axis=0)
    centered_from = points_from - mean_from
    centered_to = points_to - mean_to

    std_from = centered_from.std()
    std_to = centered_to.std()
    centered_from /= std_from
    centered_to /= std_to

    u, _, vt = np.linalg.svd(centered_from.T @ centered_to)
    rotation = (u @ vt).T

    transform = np.empty((2, 3))
    transform[:, :2] = rotation * (std_to / std_from)
    transform[:, 2] = mean_to - transform[:, :2] @ mean_from
    return transform


class FaceEngine:
    """Loads the three models once and runs the full detect -> align -> embed pipeline."""

    def __init__(self):
        core = Core()
        self._det, self._det_in, self._det_out, self._det_size = self._load(core, config.DETECTION_MODEL_PATH)
        self._lm, self._lm_in, self._lm_out, self._lm_size = self._load(core, config.LANDMARKS_MODEL_PATH)
        self._reid, self._reid_in, self._reid_out, self._reid_size = self._load(core, config.REID_MODEL_PATH)

    @staticmethod
    def _load(core, model_path):
        compiled = core.compile_model(core.read_model(model_path), config.INFERENCE_DEVICE)
        input_port = compiled.input(0)
        output_port = compiled.output(0)
        _, _, h, w = input_port.shape
        return compiled, input_port, output_port, (int(w), int(h))

    @staticmethod
    def _to_blob(image, size):
        resized = cv2.resize(image, size)
        return resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def detect_faces(self, frame):
        """Returns a list of (x1, y1, x2, y2) face boxes in pixel coordinates."""
        h, w = frame.shape[:2]
        blob = self._to_blob(frame, self._det_size)
        detections = self._det([blob])[self._det_out][0][0]

        boxes = []
        for _, label, confidence, x1, y1, x2, y2 in detections:
            if label != 1 or confidence < config.DETECTION_CONFIDENCE:
                continue
            box = (
                max(0, int(x1 * w)), max(0, int(y1 * h)),
                min(w, int(x2 * w)), min(h, int(y2 * h)),
            )
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        return boxes

    def _get_landmarks(self, face_crop):
        """Returns 5 (x, y) points normalized to [0, 1] within the crop."""
        blob = self._to_blob(face_crop, self._lm_size)
        output = self._lm([blob])[self._lm_out]
        return output.reshape(5, 2).astype(np.float64)

    def _align(self, face_crop, landmarks):
        h, w = face_crop.shape[:2]
        scale = np.array([w, h])
        reference_points = _REFERENCE_LANDMARKS * scale
        detected_points = landmarks * scale

        transform = _similarity_transform(reference_points, detected_points)
        return cv2.warpAffine(face_crop, transform, (w, h), flags=cv2.WARP_INVERSE_MAP)

    def get_embedding(self, frame, box):
        """Crops, aligns, and embeds one face. Returns a 256-d vector, or None."""
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        landmarks = self._get_landmarks(crop)
        aligned = self._align(crop, landmarks)

        blob = self._to_blob(aligned, self._reid_size)
        output = self._reid([blob])[self._reid_out]
        return output.flatten()

    def encode_image_file(self, path):
        """Reads an image file and returns an embedding for every face found."""
        frame = cv2.imread(path)
        if frame is None:
            return []
        embeddings = []
        for box in self.detect_faces(frame):
            embedding = self.get_embedding(frame, box)
            if embedding is not None:
                embeddings.append(embedding)
        return embeddings


_engine = None


def get_engine():
    """Lazily loads the models once per process and reuses the same instance."""
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine
