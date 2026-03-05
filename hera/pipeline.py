import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np

from hera import config
from hera.capture import CameraCapture
from hera.face_analyser import FaceAnalyser
from hera.processors.face_swapper import FaceSwapper
from hera.processors.face_enhancer import FaceEnhancer


class Pipeline:
    def __init__(self):
        self.camera = CameraCapture()
        self.face_analyser = None
        self.face_swapper = None
        self.face_enhancer = None
        self.source_face = None

        self._capture_queue = Queue(maxsize=2)
        self._swap_queue = Queue(maxsize=2)
        self._output_queue = Queue(maxsize=2)

        self._running = False
        self._threads = []
        self._fps = 0.0
        self._recording = False
        self._recorder = None

    def load_models(self, progress_callback=None):
        """Load all ML models. Call before start()."""
        if progress_callback:
            progress_callback("Loading face analyser...")
        self.face_analyser = FaceAnalyser()

        if progress_callback:
            progress_callback("Loading face swapper...")
        self.face_swapper = FaceSwapper()

        if progress_callback:
            progress_callback("Loading face enhancer...")
        self.face_enhancer = FaceEnhancer()

        if progress_callback:
            progress_callback("Models loaded.")

    def set_source_face(self, image_path):
        """Load reference face from image file."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.source_face = self.face_analyser.get_face_from_image(image)

    def start(self):
        if self._running:
            return
        self._running = True
        self.camera.start()

        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._swap_loop, daemon=True),
            threading.Thread(target=self._enhance_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self.camera.release()

    def get_frame(self):
        """Get the latest processed frame (non-blocking)."""
        try:
            return self._output_queue.get_nowait()
        except Empty:
            return None

    @property
    def fps(self):
        return self._fps

    def _capture_loop(self):
        while self._running:
            frame = self.camera.read()
            if frame is None:
                continue
            # Drop stale frames
            if self._capture_queue.full():
                try:
                    self._capture_queue.get_nowait()
                except Empty:
                    pass
            self._capture_queue.put((time.time(), frame))

    def _swap_loop(self):
        fps_timer = time.time()
        frame_count = 0

        while self._running:
            try:
                timestamp, frame = self._capture_queue.get(timeout=0.1)
            except Empty:
                continue

            # Drop stale frames (older than 100ms)
            if time.time() - timestamp > 0.1:
                continue

            if self.source_face is not None:
                target_face = self.face_analyser.get_one_face(frame)
                if target_face is not None:
                    frame = self.face_swapper.swap_face(frame, self.source_face, target_face)

            # Track FPS
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Drop stale in swap queue
            if self._swap_queue.full():
                try:
                    self._swap_queue.get_nowait()
                except Empty:
                    pass
            self._swap_queue.put((time.time(), frame))

    def _enhance_loop(self):
        while self._running:
            try:
                timestamp, frame = self._swap_queue.get(timeout=0.1)
            except Empty:
                continue

            if time.time() - timestamp > 0.1:
                continue

            if config.ENHANCE_ENABLED and self.source_face is not None:
                target_face = self.face_analyser.get_one_face(frame, use_cache=True)
                if target_face is not None:
                    frame = self.face_enhancer.enhance_face(frame, target_face.bbox)

            # Drop stale in output queue
            if self._output_queue.full():
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    pass
            self._output_queue.put(frame)
