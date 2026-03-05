import os
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import sounddevice as sd

from hera import config


class VideoRecorder:
    def __init__(self):
        self._writer = None
        self._recording = False
        self._start_time = None
        self._output_path = None
        self._audio_path = None
        self._final_path = None
        self._audio_thread = None
        self._audio_data = []
        self._sample_rate = 44100

    @property
    def is_recording(self):
        return self._recording

    @property
    def duration(self):
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def start(self, width, height, fps=None):
        if self._recording:
            return

        os.makedirs(config.RECORDINGS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = os.path.join(config.RECORDINGS_DIR, f"hera_{timestamp}_video.mp4")
        self._audio_path = os.path.join(config.RECORDINGS_DIR, f"hera_{timestamp}_audio.wav")
        self._final_path = os.path.join(config.RECORDINGS_DIR, f"hera_{timestamp}.mp4")

        fps = fps or config.CAPTURE_FPS
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        self._writer = cv2.VideoWriter(self._output_path, fourcc, fps, (width, height))

        if not self._writer.isOpened():
            # Fallback to mp4v if avc1 not available
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self._output_path, fourcc, fps, (width, height))

        self._audio_data = []
        self._recording = True
        self._start_time = time.time()

        # Start audio capture in background
        self._audio_thread = threading.Thread(target=self._record_audio, daemon=True)
        self._audio_thread.start()

    def write_frame(self, frame):
        if self._recording and self._writer is not None:
            self._writer.write(frame)

    def stop(self):
        if not self._recording:
            return None

        self._recording = False

        if self._audio_thread is not None:
            self._audio_thread.join(timeout=2.0)

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._start_time = None

        # Save audio and mux
        final = self._mux_audio_video()
        return final

    def _record_audio(self):
        try:
            with sd.InputStream(samplerate=self._sample_rate, channels=1, dtype="float32") as stream:
                while self._recording:
                    data, _ = stream.read(1024)
                    self._audio_data.append(data.copy())
        except Exception as e:
            import logging
            logging.getLogger("hera.recorder").warning(f"Audio capture failed: {e}")

    def _mux_audio_video(self):
        """Mux video and audio using FFmpeg."""
        import wave

        if not self._audio_data:
            # No audio, just rename video
            os.rename(self._output_path, self._final_path)
            return self._final_path

        # Save audio as WAV
        audio = np.concatenate(self._audio_data, axis=0)
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(self._audio_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self._sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Mux with FFmpeg
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", self._output_path,
                    "-i", self._audio_path,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    self._final_path,
                ],
                capture_output=True,
                check=True,
            )
            # Clean up temp files
            os.remove(self._output_path)
            os.remove(self._audio_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # FFmpeg not available or failed, keep video-only
            if os.path.exists(self._output_path):
                os.rename(self._output_path, self._final_path)
            if os.path.exists(self._audio_path):
                os.remove(self._audio_path)

        return self._final_path
