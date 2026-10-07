# Face Recognition System

Real-time face recognition powered by Intel OpenVINO (CPU inference, no GPU
required). No dlib / CMake / C++ build tools needed.

## Structure

```
face recognition/
├── dataset/            Training images
├── models/             OpenVINO IR models (downloaded, not checked in)
├── config.py           Settings (thresholds, model paths, device)
├── face_engine.py       Detection + alignment + embedding (OpenVINO)
├── utils.py             Encoding storage, dataset scanning
├── register.py          Build encodings.pickle from the dataset (CLI)
├── recognize.py         Real-time webcam recognition (CLI)
├── app.py                Desktop GUI (Tkinter)
├── app_streamlit.py       Web GUI (Streamlit)
├── download_models.py    Fetches the 3 required models
├── encodings.pickle      Generated after running register.py / app.py
└── requirements.txt
```

## How it works

Three small Intel-pretrained models are chained together (from the
[Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)):

1. **face-detection-retail-0004** — finds face bounding boxes in a frame.
2. **landmarks-regression-retail-0009** — finds 5 points per face (eyes,
   nose, mouth corners), used to align the face upright.
3. **face-reidentification-retail-0095** — turns the aligned face into a
   256-d embedding vector.

Two faces are considered the same person when the cosine distance between
their embeddings is below `MATCH_THRESHOLD` (see `config.py`).

## Setup

```powershell
pip install -r requirements.txt
python download_models.py
```

`download_models.py` fetches the three `.xml`/`.bin` model files (FP32,
CPU-friendly) into `models/`. Run it once.

## GUI

Two GUI options are included — pick whichever you prefer, both call the same
underlying `register.py` / `recognize.py` logic.

### Desktop (Tkinter)

```powershell
python app.py
```

### Web (Streamlit)

```powershell
streamlit run app_streamlit.py
```

Opens in your browser at `http://localhost:8501`. Lets you upload a photo
from disk or take one with your webcam (via the browser), right from the
page.

Both GUIs work the same way:

- **Add Person:** enter a name, provide an image, click "Add Person". If the
  name is new, a folder is created for them; if it already exists, the image
  is added to their existing photos. Either way, `encodings.pickle` is
  rebuilt automatically in the background.
- **Start Camera Recognition:** opens the live webcam window (a native
  OpenCV window, separate from the GUI). Press `q` inside that window to
  close it.

## CLI (optional, same functionality without the GUI)

### Adding people

Two supported layouts, mix freely:

```
dataset/Ali.jpg          one image, name = file name
dataset/Ahmed/
    1.jpg                multiple images, name = folder name
    2.jpg
```

More images per person (different angles/lighting) improve accuracy, but one
clear image is enough to get started.

To add someone via webcam instead of a file:

```powershell
python register.py --capture "Ali"
```
(`s` = capture a photo, `q` = stop capturing)

### Build encodings

Run this any time the dataset changes:

```powershell
python register.py
```

### Run recognition

```powershell
python recognize.py
```

Press `q` to close the camera window. Recognized faces get a green box with
their name; unmatched faces are boxed in red as "Unknown".

## Tuning (`config.py`)

| Setting | Effect |
|---|---|
| `MATCH_THRESHOLD` | Lower = stricter matching, more "Unknown" results (cosine distance, 0 = identical) |
| `DETECTION_CONFIDENCE` | Minimum confidence to accept a detected face box |
| `INFERENCE_DEVICE` | `"CPU"` (default), or `"GPU"` if you have an Intel iGPU |
| `CAMERA_INDEX` | Change if you have more than one camera |

## Notes

- **Privacy:** get consent before storing anyone's face data.
- **Spoofing:** this system does not include liveness detection — a printed
  photo or screen can fool it. Add liveness checks before any security use.
- **Accuracy:** depends heavily on lighting, angle, and image quality.
