"""
Desktop GUI for the face recognition system.

    python app.py

- Add Person: pick a name + image. If the name is new, a dataset folder is
  created for them; if the name already exists, the image is added to their
  existing photos. Either way, encodings.pickle is rebuilt automatically.
- Start Camera Recognition: opens the live webcam window from recognize.py.
"""
import os
import shutil
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

import config
import register
import recognize
from utils import VALID_EXTENSIONS


class FaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition System")
        self.resizable(False, False)

        self.image_path = None
        self.recognition_thread = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}
        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0)

        ttk.Label(container, text="Add / Update Person", font=("Segoe UI", 12, "bold")) \
            .grid(row=0, column=0, columnspan=2, sticky="w", **pad)

        ttk.Label(container, text="Name:").grid(row=1, column=0, sticky="w", **pad)
        self.name_var = tk.StringVar()
        ttk.Entry(container, textvariable=self.name_var, width=30).grid(row=1, column=1, sticky="w", **pad)

        ttk.Button(container, text="Choose Image...", command=self._choose_image) \
            .grid(row=2, column=0, sticky="w", **pad)
        self.image_label = ttk.Label(container, text="No image selected", foreground="gray")
        self.image_label.grid(row=2, column=1, sticky="w", **pad)

        self.preview_label = ttk.Label(container)
        self.preview_label.grid(row=3, column=0, columnspan=2, **pad)

        ttk.Button(container, text="Add Person", command=self._add_person) \
            .grid(row=4, column=0, columnspan=2, sticky="we", **pad)

        ttk.Separator(container, orient="horizontal").grid(row=5, column=0, columnspan=2, sticky="we", pady=10)

        ttk.Label(container, text="Real-Time Recognition", font=("Segoe UI", 12, "bold")) \
            .grid(row=6, column=0, columnspan=2, sticky="w", **pad)

        self.recognize_btn = ttk.Button(
            container, text="Start Camera Recognition", command=self._start_recognition)
        self.recognize_btn.grid(row=7, column=0, columnspan=2, sticky="we", **pad)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(container, textvariable=self.status_var, foreground="#0a5", wraplength=360, justify="left") \
            .grid(row=8, column=0, columnspan=2, sticky="w", **pad)

    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Select a face image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")],
        )
        if not path:
            return
        self.image_path = path
        self.image_label.config(text=os.path.basename(path), foreground="black")

        img = Image.open(path)
        img.thumbnail((160, 160))
        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

    def _add_person(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing name", "Please enter a name.")
            return
        if not self.image_path:
            messagebox.showwarning("Missing image", "Please choose an image.")
            return

        folder = os.path.join(config.DATASET_DIR, name)
        os.makedirs(folder, exist_ok=True)

        existing = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)]
        ext = os.path.splitext(self.image_path)[1].lower()
        dest = os.path.join(folder, f"{name}_{len(existing) + 1}{ext}")
        shutil.copy(self.image_path, dest)

        self.status_var.set(f"Saved image for '{name}'. Training...")
        self._clear_image_fields()
        self.name_var.set("")

        threading.Thread(target=self._train_in_background, args=(name,), daemon=True).start()

    def _clear_image_fields(self):
        self.image_path = None
        self.image_label.config(text="No image selected", foreground="gray")
        self.preview_label.configure(image="")

    def _train_in_background(self, name):
        try:
            num_people, num_encodings = register.build_encodings()
            message = f"'{name}' saved. Database: {num_people} people, {num_encodings} face encodings."
        except Exception as exc:
            message = f"Training failed: {exc}"
        self.after(0, lambda: self.status_var.set(message))

    def _start_recognition(self):
        if self.recognition_thread and self.recognition_thread.is_alive():
            messagebox.showinfo("Already running", "Camera recognition is already running.")
            return

        self.status_var.set("Camera starting... press 'q' in that window to stop.")
        self.recognize_btn.config(state="disabled")
        self.recognition_thread = threading.Thread(target=self._run_recognition, daemon=True)
        self.recognition_thread.start()
        self.after(300, self._poll_recognition_thread)

    def _run_recognition(self):
        try:
            recognize.main()
        except Exception as exc:
            self.after(0, lambda: self.status_var.set(f"Camera error: {exc}"))

    def _poll_recognition_thread(self):
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.after(300, self._poll_recognition_thread)
        else:
            self.recognize_btn.config(state="normal")
            self.status_var.set("Camera closed. Ready.")


if __name__ == "__main__":
    FaceApp().mainloop()
