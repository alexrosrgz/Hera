import cv2
from hera import config


class CameraCapture:
    def __init__(self, device_index=0):
        self.device_index = device_index
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAPTURE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.release()
