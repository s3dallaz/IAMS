"""
Presentation-attack (spoof) detection for the Intelligent Attendance Monitoring System.

Blink and head-turn challenges prove that a face moves like a person, but a
video of the student replayed on a phone also blinks and turns. With a plain
webcam there is no depth signal, so these checks look for the replay itself:

- AntiSpoofDetector: MiniFASNet models (Silent-Face-Anti-Spoofing, Apache-2.0,
  ONNX exports from github.com/yakhyo/face-anti-spoofing) classify a face crop
  as a real face or a photo / screen replay. Two models with different crop
  scales are averaged. Runs with OpenCV's DNN module (~5 ms per model).
- ScreenDetector: MediaPipe's EfficientDet-Lite0 object detector (COCO) finds
  a phone, laptop or TV whose box covers the face, i.e. a face shown on a screen.

Both load their model files from the project root and disable themselves
(available == False, with one warning) if a file is missing, so the app keeps
working without them.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import config
import cv2
import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent.parent

# (model file, crop scale around the face box) - the scales the models were trained with
ANTI_SPOOF_MODELS: Tuple[Tuple[str, float], ...] = (
    ("MiniFASNetV2.onnx", 2.7),
    ("MiniFASNetV1SE.onnx", 4.0),
)
ANTI_SPOOF_INPUT_SIZE = 80   # both models take 80x80 BGR crops, 0-255 floats
ANTI_SPOOF_REAL_CLASS = 1    # outputs: 0 = print attack, 1 = real face, 2 = replay attack

SCREEN_MODEL_FILE = "efficientdet_lite0.tflite"
SCREEN_LABELS = ["cell phone", "laptop", "tv"]  # COCO categories that show a face on a screen


def crop_around_face(frame: np.ndarray, bbox: Tuple[int, int, int, int], scale: float) -> Optional[np.ndarray]:
    """Crop a square-ish region `scale` times the face box, centred on it and clipped to the frame."""
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None
    # Shrink the scale if the enlarged box would not fit in the frame (as in the reference code)
    scale = min((frame_h - 1) / h, (frame_w - 1) / w, scale)
    center_x, center_y = x + w / 2, y + h / 2
    new_w, new_h = w * scale, h * scale
    x1 = max(0, int(center_x - new_w / 2))
    y1 = max(0, int(center_y - new_h / 2))
    x2 = min(frame_w - 1, int(center_x + new_w / 2))
    y2 = min(frame_h - 1, int(center_y + new_h / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2 + 1, x1:x2 + 1]


class AntiSpoofDetector:
    """Scores how likely a face crop is a real face (vs. a photo or screen)."""

    def __init__(self, model_dir: Optional[Path] = None):
        self.logger = get_logger(__name__)
        self._nets: List[Tuple[cv2.dnn.Net, float]] = []
        model_dir = Path(model_dir) if model_dir is not None else PROJECT_ROOT
        for file_name, scale in ANTI_SPOOF_MODELS:
            path = model_dir / file_name
            if not path.exists():
                self.logger.warning(f"Anti-spoof model not found: {path} (download it to enable this check)")
                continue
            try:
                self._nets.append((cv2.dnn.readNetFromONNX(str(path)), scale))
            except Exception as e:
                self.logger.error(f"Failed to load anti-spoof model {path}: {e}")
        if self._nets:
            self.logger.info(f"AntiSpoofDetector initialized with {len(self._nets)} model(s)")

    @property
    def available(self) -> bool:
        return bool(self._nets)

    def real_score(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[float]:
        """
        Probability (0-1) that the face in `bbox` is a real face, averaged over
        the loaded models. None if no model is loaded or the crop is unusable.
        """
        scores = []
        for net, scale in self._nets:
            crop = crop_around_face(frame, bbox, scale)
            if crop is None or crop.size == 0:
                continue
            crop = cv2.resize(crop, (ANTI_SPOOF_INPUT_SIZE, ANTI_SPOOF_INPUT_SIZE))
            blob = np.transpose(crop.astype(np.float32), (2, 0, 1))[np.newaxis]
            net.setInput(blob)
            logits = net.forward()[0]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            scores.append(float(probs[ANTI_SPOOF_REAL_CLASS]))
        return sum(scores) / len(scores) if scores else None


class ScreenDetector:
    """Finds a phone / laptop / TV screen that covers the face."""

    def __init__(self, model_path: Optional[Path] = None, score_threshold: Optional[float] = None):
        self.logger = get_logger(__name__)
        self._detector = None
        path = Path(model_path) if model_path is not None else PROJECT_ROOT / SCREEN_MODEL_FILE
        threshold = score_threshold if score_threshold is not None else config.SCREEN_DETECTION_CONFIDENCE
        if not path.exists():
            self.logger.warning(f"Screen detector model not found: {path} (download it to enable this check)")
            return
        try:
            options = vision.ObjectDetectorOptions(
                base_options=python.BaseOptions(model_asset_path=str(path)),
                score_threshold=threshold,
                category_allowlist=SCREEN_LABELS,
                max_results=5,
            )
            self._detector = vision.ObjectDetector.create_from_options(options)
            self.logger.info(f"ScreenDetector initialized (threshold {threshold})")
        except Exception as e:
            self.logger.error(f"Failed to load screen detector {path}: {e}")

    @property
    def available(self) -> bool:
        return self._detector is not None

    def detect_screens(self, frame: np.ndarray) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
        """All phone / laptop / TV boxes in the frame as (label, score, (x, y, w, h))."""
        if self._detector is None:
            return []
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        result = self._detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
        screens = []
        for detection in result.detections:
            category = detection.categories[0]
            box = detection.bounding_box
            screens.append((category.category_name, float(category.score),
                            (box.origin_x, box.origin_y, box.width, box.height)))
        return screens

    def screen_covering_face(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Label of a screen whose box covers most of the face (the face is shown
        on it), or None. A phone held next to the face does not count.
        """
        fx, fy, fw, fh = face_bbox
        face_area = fw * fh
        if face_area <= 0:
            return None
        for label, _score, (sx, sy, sw, sh) in self.detect_screens(frame):
            overlap_w = max(0, min(fx + fw, sx + sw) - max(fx, sx))
            overlap_h = max(0, min(fy + fh, sy + sh) - max(fy, sy))
            if overlap_w * overlap_h / face_area >= config.SCREEN_FACE_OVERLAP:
                return label
        return None
