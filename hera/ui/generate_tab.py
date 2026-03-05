import os
import subprocess
import threading
import time
from tkinter import filedialog

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

from hera import config
from hera.capture import CameraCapture
from hera.generate_pipeline import GeneratePipeline
from hera.recorder import VideoRecorder


class GenerateTab:
    def __init__(self, parent, status_callback=None):
        self.parent = parent
        self.status_callback = status_callback or (lambda msg: None)
        self.pipeline = GeneratePipeline()

        self._source_image_path = None
        self._driving_video_path = None
        self._source_thumbnail = None
        self._driving_thumbnail = None
        self._result_thumbnail = None
        self._models_loaded = False
        self._recording_driving = False
        self._camera = None
        self._driving_recorder = None

        self._build_ui()

    def _build_ui(self):
        self.frame = ctk.CTkFrame(self.parent)
        self.frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.frame.grid_columnconfigure((0, 1), weight=1)

        # --- Source Image Section ---
        source_frame = ctk.CTkFrame(self.frame)
        source_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        source_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(source_frame, text="Source Image", font=("", 16, "bold")).pack(pady=(10, 5))

        self.source_preview = ctk.CTkLabel(source_frame, text="No image selected", width=256, height=256)
        self.source_preview.pack(pady=5)

        self.source_button = ctk.CTkButton(
            source_frame, text="Select Image",
            command=self._select_source_image
        )
        self.source_button.pack(pady=5)

        self.source_path_label = ctk.CTkLabel(source_frame, text="", wraplength=250)
        self.source_path_label.pack(pady=(0, 10))

        # --- Driving Video Section ---
        driving_frame = ctk.CTkFrame(self.frame)
        driving_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        driving_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(driving_frame, text="Driving Video", font=("", 16, "bold")).pack(pady=(10, 5))

        self.driving_preview = ctk.CTkLabel(driving_frame, text="No video selected", width=256, height=256)
        self.driving_preview.pack(pady=5)

        driving_buttons = ctk.CTkFrame(driving_frame)
        driving_buttons.pack(pady=5)

        self.driving_file_button = ctk.CTkButton(
            driving_buttons, text="Select Video File",
            command=self._select_driving_video
        )
        self.driving_file_button.pack(side="left", padx=5)

        self.driving_record_button = ctk.CTkButton(
            driving_buttons, text="Record from Webcam",
            fg_color="red", hover_color="darkred",
            command=self._toggle_webcam_recording
        )
        self.driving_record_button.pack(side="left", padx=5)

        self.driving_path_label = ctk.CTkLabel(driving_frame, text="", wraplength=250)
        self.driving_path_label.pack(pady=(0, 10))

        # --- Generate Controls ---
        controls_frame = ctk.CTkFrame(self.frame)
        controls_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        controls_frame.grid_columnconfigure(0, weight=1)

        button_row = ctk.CTkFrame(controls_frame)
        button_row.pack(pady=10)

        self.generate_button = ctk.CTkButton(
            button_row, text="Generate",
            width=200, height=40, font=("", 16, "bold"),
            command=self._start_generate
        )
        self.generate_button.pack(side="left", padx=10)

        self.cancel_button = ctk.CTkButton(
            button_row, text="Cancel",
            width=100, height=40,
            state="disabled",
            command=self._cancel_generate
        )
        self.cancel_button.pack(side="left", padx=10)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(controls_frame, width=500)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(controls_frame, text="Ready")
        self.progress_label.pack(pady=(0, 10))

        # --- Result Section ---
        result_frame = ctk.CTkFrame(self.frame)
        result_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        result_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(result_frame, text="Result", font=("", 16, "bold")).pack(pady=(10, 5))

        result_row = ctk.CTkFrame(result_frame)
        result_row.pack(pady=5)

        self.result_preview = ctk.CTkLabel(result_row, text="No output yet", width=256, height=192)
        self.result_preview.pack(side="left", padx=10)

        result_info = ctk.CTkFrame(result_row)
        result_info.pack(side="left", padx=10)

        self.result_path_label = ctk.CTkLabel(result_info, text="", wraplength=300)
        self.result_path_label.pack(pady=5)

        self.open_button = ctk.CTkButton(
            result_info, text="Open Video",
            state="disabled",
            command=self._open_result
        )
        self.open_button.pack(pady=5)

    def load_models(self, done_callback=None):
        def _load():
            try:
                def _on_progress(msg):
                    self.parent.after(0, lambda m=msg: self.status_callback(m))

                self.pipeline.load_models(progress_callback=_on_progress)
                self._models_loaded = True
                if done_callback:
                    self.parent.after(0, done_callback)
            except Exception as e:
                err_msg = f"LivePortrait error: {e}"
                self.parent.after(0, lambda m=err_msg: self.status_callback(m))
        threading.Thread(target=_load, daemon=True).start()

    def _select_source_image(self):
        path = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")],
            initialdir=config.REFERENCES_DIR,
        )
        if path:
            self._source_image_path = path
            self._show_image_preview(path, self.source_preview, 256)
            self.source_path_label.configure(text=os.path.basename(path))

    def _select_driving_video(self):
        path = filedialog.askopenfilename(
            title="Select Driving Video",
            filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv")],
            initialdir=config.DRIVING_VIDEOS_DIR if os.path.exists(config.DRIVING_VIDEOS_DIR) else None,
        )
        if path:
            self._driving_video_path = path
            self._show_video_thumbnail(path, self.driving_preview, 256)
            self.driving_path_label.configure(text=os.path.basename(path))

    def _toggle_webcam_recording(self):
        if self._recording_driving:
            self._stop_webcam_recording()
        else:
            self._start_webcam_recording()

    def _start_webcam_recording(self):
        os.makedirs(config.DRIVING_VIDEOS_DIR, exist_ok=True)
        self._camera = CameraCapture()
        self._driving_recorder = VideoRecorder()

        try:
            self._camera.start()
        except RuntimeError as e:
            self.status_callback(f"Camera error: {e}")
            return

        self._driving_recorder.start(
            config.CAPTURE_WIDTH, config.CAPTURE_HEIGHT,
        )
        self._recording_driving = True
        self.driving_record_button.configure(text="Stop Recording")
        self.status_callback("Recording driving video...")
        self._webcam_record_loop()

    def _webcam_record_loop(self):
        if not self._recording_driving:
            return
        frame = self._camera.read()
        if frame is not None:
            self._driving_recorder.write_frame(frame)
            # Show preview
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image.thumbnail((256, 256), Image.LANCZOS)
            self._driving_thumbnail = ImageTk.PhotoImage(image)
            self.driving_preview.configure(image=self._driving_thumbnail, text="")

        self.parent.after(33, self._webcam_record_loop)

    def _stop_webcam_recording(self):
        self._recording_driving = False
        self.driving_record_button.configure(text="Record from Webcam")

        if self._driving_recorder and self._driving_recorder.is_recording:
            final_path = self._driving_recorder.stop()
            if final_path:
                # Move to driving_videos dir
                import shutil
                dest = os.path.join(config.DRIVING_VIDEOS_DIR, os.path.basename(final_path))
                shutil.move(final_path, dest)
                self._driving_video_path = dest
                self.driving_path_label.configure(text=os.path.basename(dest))
                self.status_callback(f"Driving video saved: {os.path.basename(dest)}")

        if self._camera:
            self._camera.release()
            self._camera = None

    def _start_generate(self):
        if not self._source_image_path:
            self.status_callback("Please select a source image first.")
            return
        if not self._driving_video_path:
            self.status_callback("Please select a driving video first.")
            return
        if not self._models_loaded:
            self.status_callback("LivePortrait models not loaded yet. Please wait...")
            return

        self.generate_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress_bar.set(0)
        self.progress_label.configure(text="Starting generation...")
        self.status_callback("Generating...")

        def _progress(status, detail):
            if status == "generating":
                self.parent.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
                self.parent.after(0, lambda: self.progress_bar.start())
                self.parent.after(0, lambda: self.progress_label.configure(text="Generating..."))
            elif status == "done":
                self.parent.after(0, lambda: self.progress_bar.stop())
                self.parent.after(0, lambda: self.progress_bar.configure(mode="determinate"))
                self.parent.after(0, lambda: self.progress_bar.set(1.0))

        def _done():
            self.parent.after(0, self._on_generate_done)

        self.pipeline.generate(
            self._source_image_path,
            self._driving_video_path,
            progress_callback=_progress,
            done_callback=_done,
        )

    def _cancel_generate(self):
        self.pipeline.cancel()
        self.cancel_button.configure(state="disabled")
        self.progress_label.configure(text="Cancelling...")
        self.status_callback("Cancelling generation...")

    def _on_generate_done(self):
        self.generate_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

        if self.pipeline.error:
            self.progress_label.configure(text=f"Error: {self.pipeline.error}")
            self.status_callback(f"Generation error: {self.pipeline.error}")
            return

        output = self.pipeline.output_path
        if output and os.path.exists(output):
            self.progress_bar.set(1.0)
            self.progress_label.configure(text=f"Done! Saved to {os.path.basename(output)}")
            self.result_path_label.configure(text=output)
            self.open_button.configure(state="normal")
            self._show_video_thumbnail(output, self.result_preview, 256)
            self.status_callback(f"Generated: {os.path.basename(output)}")
        else:
            self.progress_label.configure(text="Generation cancelled.")
            self.status_callback("Generation cancelled.")

    def _open_result(self):
        output = self.pipeline.output_path
        if output and os.path.exists(output):
            subprocess.Popen(["open", output])

    def _show_image_preview(self, path, label, size):
        try:
            image = Image.open(path)
            image.thumbnail((size, size), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo, text="")
            # Keep reference to prevent garbage collection
            if label == self.source_preview:
                self._source_thumbnail = photo
            else:
                self._result_thumbnail = photo
        except Exception:
            label.configure(text="Could not load image")

    def _show_video_thumbnail(self, path, label, size):
        try:
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                image.thumbnail((size, size), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                label.configure(image=photo, text="")
                if label == self.driving_preview:
                    self._driving_thumbnail = photo
                else:
                    self._result_thumbnail = photo
            else:
                label.configure(text="Could not read video")
        except Exception:
            label.configure(text="Could not load video")

    def cleanup(self):
        if self._recording_driving:
            self._stop_webcam_recording()
        if self.pipeline.is_generating:
            self.pipeline.cancel()
