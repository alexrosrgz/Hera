import cv2
import numpy as np
import onnxruntime
from hera import config


class FaceEnhancer:
    def __init__(self):
        self.session = onnxruntime.InferenceSession(
            config.ENHANCER_MODEL_PATH,
            providers=config.EXECUTION_PROVIDERS,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, 512, 512]
        self.resolution = self.input_shape[2]  # 512

    def enhance_face(self, frame, face_bbox):
        """Enhance face region in frame using GFPGAN."""
        if not config.ENHANCE_ENABLED:
            return frame

        bbox = face_bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Add padding around face
        h, w = frame.shape[:2]
        pad_w = int((x2 - x1) * 0.3)
        pad_h = int((y2 - y1) * 0.3)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Crop face region
        face_crop = frame[y1:y2, x1:x2]
        orig_size = (face_crop.shape[1], face_crop.shape[0])

        # Preprocess: resize, normalize, transpose to NCHW
        input_img = cv2.resize(face_crop, (self.resolution, self.resolution))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = (input_img - 0.5) / 0.5  # Normalize to [-1, 1]
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dim

        # Run inference
        output = self.session.run(None, {self.input_name: input_img})[0]

        # Postprocess: reverse normalization, transpose back
        output = np.squeeze(output)
        output = np.transpose(output, (1, 2, 0))  # CHW -> HWC
        output = (output * 0.5 + 0.5) * 255.0
        output = np.clip(output, 0, 255).astype(np.uint8)

        # Resize back and paste
        output = cv2.resize(output, orig_size)
        result = frame.copy()
        result[y1:y2, x1:x2] = output
        return result
