import os
import threading
import time
from tkinter import filedialog

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from hera import config
from hera.pipeline import Pipeline
from hera.recorder import VideoRecorder


class HeraApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hera - AI Face Swap")
        self.geometry("900x700")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.pipeline = Pipeline()
        self.recorder = VideoRecorder()
        self._preview_image = None
        self._running = False

        self._build_ui()
        self._load_models()

    def _build_ui(self):
        # Main layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Preview panel
        self.preview_label = ctk.CTkLabel(self, text="Loading models...")
        self.preview_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Controls frame
        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        controls.grid_columnconfigure((0, 1, 2, 3), weight=1)

        # Reference face button
        self.ref_button = ctk.CTkButton(
            controls, text="Select Reference Face",
            command=self._select_reference
        )
        self.ref_button.grid(row=0, column=0, padx=5, pady=5)

        # Record button
        self.record_button = ctk.CTkButton(
            controls, text="Start Recording",
            fg_color="red", hover_color="darkred",
            command=self._toggle_recording
        )
        self.record_button.grid(row=0, column=1, padx=5, pady=5)

        # Enhancement toggle
        self.enhance_var = ctk.BooleanVar(value=config.ENHANCE_ENABLED)
        self.enhance_check = ctk.CTkCheckBox(
            controls, text="Face Enhancement",
            variable=self.enhance_var,
            command=self._toggle_enhance
        )
        self.enhance_check.grid(row=0, column=2, padx=5, pady=5)

        # Detection interval slider
        slider_frame = ctk.CTkFrame(controls)
        slider_frame.grid(row=0, column=3, padx=5, pady=5)
        ctk.CTkLabel(slider_frame, text="Detection Freq").pack()
        self.detection_slider = ctk.CTkSlider(
            slider_frame, from_=1, to=5, number_of_steps=4,
            command=self._update_detection_interval
        )
        self.detection_slider.set(config.FACE_DETECTION_INTERVAL)
        self.detection_slider.pack()

        # Status bar
        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        self.status_frame.grid_columnconfigure((0, 1, 2), weight=1)

        self.fps_label = ctk.CTkLabel(self.status_frame, text="FPS: --")
        self.fps_label.grid(row=0, column=0, padx=10)

        self.rec_label = ctk.CTkLabel(self.status_frame, text="")
        self.rec_label.grid(row=0, column=1, padx=10)

        self.status_label = ctk.CTkLabel(self.status_frame, text="Status: Loading")
        self.status_label.grid(row=0, column=2, padx=10)

    def _load_models(self):
        def _load():
            try:
                self.pipeline.load_models(
                    progress_callback=lambda msg: self.after(
                        0, lambda: self.status_label.configure(text=f"Status: {msg}")
                    )
                )
                # Try loading default reference face
                if os.path.exists(config.DEFAULT_REFERENCE_PATH):
                    self.pipeline.set_source_face(config.DEFAULT_REFERENCE_PATH)

                self.after(0, self._start_preview)
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))

        threading.Thread(target=_load, daemon=True).start()

    def _start_preview(self):
        self._running = True
        self.pipeline.start()
        self.status_label.configure(text="Status: Running")
        self._update_preview()

    def _update_preview(self):
        if not self._running:
            return

        frame = self.pipeline.get_frame()
        if frame is not None:
            # Write to recorder if recording
            if self.recorder.is_recording:
                self.recorder.write_frame(frame)

            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Scale to fit preview
            preview_w = self.preview_label.winfo_width()
            preview_h = self.preview_label.winfo_height()
            if preview_w > 1 and preview_h > 1:
                image.thumbnail((preview_w, preview_h), Image.LANCZOS)

            self._preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self._preview_image, text="")

        # Update status
        self.fps_label.configure(text=f"FPS: {self.pipeline.fps:.1f}")
        if self.recorder.is_recording:
            duration = self.recorder.duration
            self.rec_label.configure(text=f"REC {duration:.1f}s", text_color="red")
        else:
            self.rec_label.configure(text="")

        self.after(33, self._update_preview)  # ~30 FPS refresh

    def _select_reference(self):
        path = filedialog.askopenfilename(
            title="Select Reference Face",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")],
            initialdir=config.REFERENCES_DIR,
        )
        if path:
            try:
                self.pipeline.set_source_face(path)
                self.status_label.configure(text=f"Status: Reference loaded")
            except ValueError as e:
                self.status_label.configure(text=f"Error: {e}")

    def _toggle_recording(self):
        if self.recorder.is_recording:
            final_path = self.recorder.stop()
            self.record_button.configure(text="Start Recording")
            self.status_label.configure(text=f"Saved: {os.path.basename(final_path)}")
        else:
            self.recorder.start(config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT)
            self.record_button.configure(text="Stop Recording")
            self.status_label.configure(text="Status: Recording...")

    def _toggle_enhance(self):
        config.ENHANCE_ENABLED = self.enhance_var.get()

    def _update_detection_interval(self, value):
        config.FACE_DETECTION_INTERVAL = int(value)

    def _on_close(self):
        self._running = False
        if self.recorder.is_recording:
            self.recorder.stop()
        self.pipeline.stop()
        self.destroy()
