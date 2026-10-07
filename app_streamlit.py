"""
Streamlit GUI for the face recognition system.

    streamlit run app_streamlit.py

- Add Person: upload or take a photo + a name. If the name is new, a dataset
  folder is created for them; if it already exists, the photo is added to
  their existing photos. Either way, encodings.pickle is rebuilt automatically.
- Start Camera Recognition: launches the live webcam window (recognize.py)
  as a separate process so the web page stays responsive.
"""
import os
import subprocess
import sys
from collections import Counter

import streamlit as st

import config
import register
from utils import VALID_EXTENSIONS, load_encodings

st.set_page_config(page_title="Face Recognition System", page_icon="🙂", layout="centered")

st.title("Face Recognition System")

st.header("Add / Update Person")

name = st.text_input("Name")

source = st.radio("Image source", ["Upload a photo", "Take a photo"], horizontal=True)
image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"]) \
    if source == "Upload a photo" else st.camera_input("Take a photo")

if image_file is not None:
    st.image(image_file, width=200)

if st.button("Add Person", type="primary"):
    clean_name = name.strip()
    if not clean_name:
        st.warning("Please enter a name.")
    elif image_file is None:
        st.warning("Please provide an image.")
    else:
        folder = os.path.join(config.DATASET_DIR, clean_name)
        os.makedirs(folder, exist_ok=True)
        existing = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)]

        ext = os.path.splitext(getattr(image_file, "name", ""))[1].lower()
        if ext not in VALID_EXTENSIONS:
            ext = ".jpg"

        dest = os.path.join(folder, f"{clean_name}_{len(existing) + 1}{ext}")
        with open(dest, "wb") as f:
            f.write(image_file.getbuffer())

        with st.spinner("Training..."):
            num_people, num_encodings = register.build_encodings()

        st.success(f"'{clean_name}' saved. Database: {num_people} people, {num_encodings} face encodings.")

st.divider()

st.header("Real-Time Recognition")
st.caption("Opens a separate camera window. Press 'q' in that window to close it.")

if st.button("Start Camera Recognition"):
    subprocess.Popen([sys.executable, "recognize.py"], cwd=config.BASE_DIR)
    st.info("Camera window launching — check your taskbar.")

st.divider()

st.header("Registered People")
_, names = load_encodings()
if names:
    for person, count in sorted(Counter(names).items()):
        st.write(f"- **{person}** — {count} photo(s) encoded")
else:
    st.write("No one registered yet.")
