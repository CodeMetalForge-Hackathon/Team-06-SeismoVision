"""Fast, single-target cardboard segmentation for the tabletop demo."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CardboardResult:
    mask: np.ndarray
    outline: Optional[np.ndarray]
    area_fraction: float


class CardboardDetector:
    """Finds the dominant central cardboard structure; it is not a general object model."""

    def __init__(self, minimum_area_fraction: float = 0.025):
        self.minimum_area_fraction = minimum_area_fraction
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame: np.ndarray) -> CardboardResult:
        # Normalising luminance keeps the brown-cardboard range useful in dim indoor scenes.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
        hsv = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (4, 35, 35), (35, 255, 255))

        # Join the tower's cardboard panels across small seams and black tape gaps.
        side = max(9, min(41, (min(frame.shape[:2]) // 12) | 1))
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((side, side), np.uint8))
        joined = cv2.morphologyEx(joined, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_area = float(frame.shape[0] * frame.shape[1])
        image_center = np.array([frame.shape[1] / 2.0, frame.shape[0] / 2.0])

        best: Optional[np.ndarray] = None
        best_score = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * self.minimum_area_fraction:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
            normalized_distance = np.linalg.norm(center - image_center) / np.linalg.norm(image_center)
            score = area * (2.0 - min(1.0, normalized_distance))
            if score > best_score:
                best, best_score = contour, score

        selected = np.zeros(mask.shape, dtype=np.uint8)
        if best is None:
            return CardboardResult(selected, None, 0.0)
        cv2.drawContours(selected, [best], -1, 255, thickness=cv2.FILLED)
        hull = cv2.convexHull(best)
        perimeter = cv2.arcLength(hull, True)
        outline = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        return CardboardResult(selected, outline, cv2.contourArea(best) / image_area)


def draw_cardboard_outline(frame: np.ndarray, result: CardboardResult) -> np.ndarray:
    """Returns the camera preview with the detected cardboard boundary in yellow."""
    preview = frame.copy()
    if result.outline is None:
        cv2.putText(preview, "CARDBOARD: NOT FOUND", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return preview
    cv2.polylines(preview, [result.outline], True, (0, 255, 255), 3)
    for point in result.outline.reshape(-1, 2):
        cv2.circle(preview, tuple(int(v) for v in point), 6, (0, 255, 255), cv2.FILLED)
    cv2.putText(preview, "CARDBOARD TARGET", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return preview
