import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")

MODELS_DIR = os.path.join(BASE_DIR, "models")
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "face-detection-retail-0004.xml")
LANDMARKS_MODEL_PATH = os.path.join(MODELS_DIR, "landmarks-regression-retail-0009.xml")
REID_MODEL_PATH = os.path.join(MODELS_DIR, "face-reidentification-retail-0095.xml")

INFERENCE_DEVICE = "CPU"       # "CPU", "GPU" (Intel iGPU), or "AUTO"
DETECTION_CONFIDENCE = 0.6     # minimum confidence to accept a face detection
MATCH_THRESHOLD = 0.4        # cosine distance below this = same person (lower = stricter)

CAMERA_INDEX = 0
NUM_IMAGES_TO_CAPTURE = 10
