import cv2
import numpy as np
import insightface
from hera import config


class FaceSwapper:
    def __init__(self):
        self.model = insightface.model_zoo.get_model(
            config.SWAPPER_MODEL_PATH,
            providers=config.EXECUTION_PROVIDERS,
        )

    def swap_face(self, frame, source_face, target_face):
        """Swap target_face in frame with source_face appearance."""
        result = self.model.get(frame, target_face, source_face, paste_back=True)
        result = self._color_correct(result, frame, target_face)
        return result

    def _color_correct(self, swapped, original, face):
        """Match skin tone of swapped face to original lighting using LAB color space."""
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Clamp to frame bounds
        h, w = original.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return swapped

        # Extract face regions
        orig_face = original[y1:y2, x1:x2]
        swap_face = swapped[y1:y2, x1:x2]

        # Convert to LAB
        orig_lab = cv2.cvtColor(orig_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        swap_lab = cv2.cvtColor(swap_face, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Match mean and std of L channel
        for ch in range(3):
            orig_mean, orig_std = orig_lab[:, :, ch].mean(), orig_lab[:, :, ch].std()
            swap_mean, swap_std = swap_lab[:, :, ch].mean(), swap_lab[:, :, ch].std()
            if swap_std < 1e-6:
                continue
            swap_lab[:, :, ch] = (swap_lab[:, :, ch] - swap_mean) * (orig_std / swap_std) + orig_mean

        swap_lab = np.clip(swap_lab, 0, 255).astype(np.uint8)
        corrected_face = cv2.cvtColor(swap_lab, cv2.COLOR_LAB2BGR)

        result = swapped.copy()
        result[y1:y2, x1:x2] = corrected_face
        return result
