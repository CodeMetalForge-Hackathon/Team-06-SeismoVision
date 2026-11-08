"""Scene-specific cardboard localization for the controlled table demo."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Rect


@dataclass(frozen=True)
class StructureBoxes:
    upper: Rect
    lower: Rect
    full: Rect


def detect_structure(frame: np.ndarray) -> StructureBoxes | None:
    height, width = frame.shape[:2]
    if height < 120 or width < 160:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cardboard = cv2.inRange(hsv, (7, 35, 35), (35, 255, 245))
    count, _, stats, _ = cv2.connectedComponentsWithStats(cardboard)
    candidates = []
    for x, y, w, h, area in stats[1:count]:
        if x < 3 or x + w > width - 3 or y < 3:
            continue
        if not (0.18 * width < w < 0.72 * width and 0.06 * height < h < 0.35 * height and w > 1.3 * h):
            continue
        if y > 0.5 * height or area < 0.012 * width * height:
            continue
        center_penalty = 1.0 + abs(x + w / 2 - width / 2) / (width / 2)
        candidates.append((area / center_penalty, (int(x), int(y), int(w), int(h))))
    if not candidates:
        return None
    _, upper = max(candidates)
    x, y, w, h = upper
    center = x + w // 2
    half = min(int(w * 0.43), int(width * 0.28))
    left, right = max(0, center - half), min(width, center + half)
    start = y + h
    lower_mask = cardboard[start:min(height, int(0.96 * height)), left:right]
    if lower_mask.size == 0:
        return None
    row_counts = np.count_nonzero(lower_mask, axis=1)
    visible_rows = np.flatnonzero(row_counts > max(5, 0.16 * (right - left)))
    if len(visible_rows) < 0.18 * height:
        return None
    bottom = start + int(visible_rows[-1]) + 1
    if bottom > 0.94 * height or bottom - start < 0.25 * height:
        return None
    body = cardboard[start + int(visible_rows[0]):bottom, left:right]
    column_counts = np.count_nonzero(body, axis=0)
    visible_columns = np.flatnonzero(column_counts > 0.12 * body.shape[0])
    if len(visible_columns) < 0.1 * width:
        return None
    body_left = left + int(visible_columns[0])
    body_right = left + int(visible_columns[-1]) + 1
    lower = (body_left, start, body_right - body_left, bottom - start)
    full_left, full_right = min(x, body_left), max(x + w, body_right)
    return StructureBoxes(upper, lower, (full_left, y, full_right - full_left, bottom - y))


def select_background_reference(frame: np.ndarray, target: Rect) -> Rect | None:
    height, width = frame.shape[:2]
    x, y, w, h = target
    margin = max(8, width // 40)
    regions = []
    if x - margin >= 0.18 * width:
        regions.append((margin, max(margin, y), x - 2 * margin, min(h, height - y - margin)))
    if width - x - w - margin >= 0.18 * width:
        regions.append((x + w + margin, max(margin, y), width - x - w - 2 * margin, min(h, height - y - margin)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scored = []
    for region in regions:
        left, top, rw, rh = region
        if rw < 40 or rh < 40:
            continue
        crop = gray[top:top + rh, left:left + rw]
        points = cv2.goodFeaturesToTrack(crop, maxCorners=100, qualityLevel=0.02, minDistance=7)
        count = 0 if points is None else len(points)
        if count >= 12:
            scored.append((count, region))
    return max(scored)[1] if scored else None
