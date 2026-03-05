import insightface
import numpy as np
from hera import config


class FaceAnalyser:
    def __init__(self):
        self.app = insightface.app.FaceAnalysis(
            name=config.FACE_ANALYSER_MODEL,
            providers=config.EXECUTION_PROVIDERS,
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._cached_faces = None
        self._frame_count = 0

    def detect_faces(self, frame):
        return self.app.get(frame)

    def get_one_face(self, frame, use_cache=True):
        if use_cache and self._cached_faces is not None and self._frame_count % config.FACE_DETECTION_INTERVAL != 0:
            self._frame_count += 1
            return self._cached_faces

        faces = self.detect_faces(frame)
        self._frame_count += 1

        if not faces:
            self._cached_faces = None
            return None

        # Return the largest face by bounding box area
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        self._cached_faces = largest
        return largest

    def get_face_from_image(self, image):
        """Get a face from a static image (for reference face loading)."""
        faces = self.detect_faces(image)
        if not faces:
            raise ValueError("No face detected in reference image")
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
