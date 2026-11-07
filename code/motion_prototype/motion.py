"""Experimental anchored image-motion measurement; no structural safety inference."""
from collections import deque
import math

import cv2
import numpy as np


class MotionMap:
    """Call update(BGR_frame, capture_time_seconds); returns JSON-safe measurements.

    Rectangles are (x, y, width, height) in original image pixels. Initialization
    must use a sharp, quiet frame. Reference must be independent of the target.
    Recreate this object to explicitly reset the anchor after camera repositioning.
    """

    def __init__(self, building, reference, grid=(3, 3), threshold_px=0.75):
        self.building = tuple(building)
        self.reference = tuple(reference)
        self.grid = tuple(grid)
        if len(self.grid) != 2 or any(type(v) is not int or v < 1 for v in self.grid):
            raise ValueError("grid must contain two positive integers")
        if not math.isfinite(threshold_px) or threshold_px <= 0:
            raise ValueError("threshold_px must be positive and finite")
        self.threshold = float(threshold_px)
        self.anchor = None
        self.last_time = None
        self.cells = []
        self.history = {}

    @staticmethod
    def _rect(rect, shape):
        if len(rect) != 4 or any(type(v) is not int for v in rect):
            raise ValueError("ROI must contain four integers")
        x, y, w, h = rect
        if min(x, y) < 0 or min(w, h) < 16 or x + w > shape[1] or y + h > shape[0]:
            raise ValueError("ROI must be inside frame and at least 16x16 pixels")

    @staticmethod
    def _detect(gray, rect):
        x, y, w, h = rect
        mask = np.zeros_like(gray)
        # Avoid patches mixing the target with its surroundings at cell edges.
        mask[y + 5:y + h - 5, x + 5:x + w - 5] = 255
        points = cv2.goodFeaturesToTrack(gray, 60, 0.02, 6, mask=mask, blockSize=7)
        return np.empty((0, 1, 2), np.float32) if points is None else points

    def _track(self, gray, points):
        if len(points) < 6:
            return np.empty((0, 2)), np.empty((0, 2))
        opts = dict(winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        new, status, error = cv2.calcOpticalFlowPyrLK(self.anchor, gray, points, None, **opts)
        if new is None or status is None or error is None or not np.isfinite(new).all():
            return np.empty((0, 2)), np.empty((0, 2))
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.anchor, new, None, **opts)
        if back is None or back_status is None:
            return np.empty((0, 2)), np.empty((0, 2))
        p, q = points.reshape(-1, 2), new.reshape(-1, 2)
        good = (status.ravel() == 1) & (back_status.ravel() == 1)
        good &= np.isfinite(back.reshape(-1, 2)).all(axis=1)
        good &= np.linalg.norm(back.reshape(-1, 2) - p, axis=1) < 0.75
        good &= error.ravel() < 20
        good &= (q[:, 0] >= 10) & (q[:, 0] < gray.shape[1] - 10)
        good &= (q[:, 1] >= 10) & (q[:, 1] < gray.shape[0] - 10)
        return p[good], q[good]

    def _initialize(self, gray):
        self._rect(self.building, gray.shape)
        self._rect(self.reference, gray.shape)
        x, y, w, h = self.building
        rx, ry, rw, rh = self.reference
        if max(x, rx) < min(x + w, rx + rw) and max(y, ry) < min(y + h, ry + rh):
            raise ValueError("building and reference must not overlap")
        rows, cols = self.grid
        if min(w // cols, h // rows) < 24:
            raise ValueError("grid cells must be at least 24x24 pixels")
        self.anchor = gray.copy()
        self.ref_points = self._detect(gray, self.reference)
        for row in range(rows):
            for col in range(cols):
                left, top = x + col * w // cols, y + row * h // rows
                rect = (left, top, x + (col + 1) * w // cols - left,
                        y + (row + 1) * h // rows - top)
                self.cells.append((rect, self._detect(gray, rect)))
                self.history[len(self.cells) - 1] = deque(maxlen=600)

    def _unknown(self, timestamp, reason):
        for history in self.history.values():
            history.clear()
        return dict(timestamp=timestamp, state="UNRELIABLE", reason=reason,
                    threshold_px=self.threshold, reference=None,
                    cells=[dict(id=i, roi=list(rect), state="UNRELIABLE",
                                displacement_px=None, magnitude_px=None, dominant_hz=None,
                                tracked_points=0)
                           for i, (rect, _) in enumerate(self.cells)])

    @staticmethod
    def _frequency(history):
        if len(history) < 90 or history[-1][0] - history[0][0] < 3:
            return None
        data = np.array(history)
        times, xy = data[:, 0], data[:, 1:]
        dt = float(np.median(np.diff(times)))
        if np.max(np.diff(times)) > 2.5 * dt:
            return None
        uniform = np.arange(times[0], times[-1], dt)
        values = np.column_stack([np.interp(uniform, times, xy[:, axis]) for axis in (0, 1)])
        if np.max(np.ptp(values, axis=0)) < 0.5:
            return None
        # Remove a linear trend so a slow drift does not masquerade as vibration.
        t = uniform - uniform[0]
        values -= np.column_stack([t, np.ones(len(t))]) @ np.linalg.lstsq(
            np.column_stack([t, np.ones(len(t))]), values, rcond=None)[0]
        power = np.sum(np.abs(np.fft.rfft(values * np.hanning(len(t))[:, None], axis=0)) ** 2, axis=1)
        hz = np.fft.rfftfreq(len(t), dt)
        valid = (hz >= max(0.2, 3 / (times[-1] - times[0]))) & (hz <= min(5, 0.4 / dt))
        if not np.any(valid):
            return None
        band = power[valid]
        if np.max(band) < 6 * max(float(np.median(band)), 1e-9):
            return None
        return float(hz[valid][np.argmax(band)])

    def update(self, frame, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("timestamps must be finite and strictly increasing")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be uint8 BGR")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.anchor is not None and gray.shape != self.anchor.shape:
            raise ValueError("frame dimensions changed; reset tracker")
        if self.last_time is not None and timestamp - self.last_time > 0.5:
            for history in self.history.values():
                history.clear()
        self.last_time = timestamp
        if self.anchor is None:
            self._initialize(gray)
            return self._unknown(timestamp, "anchor_initialized; measurement needs another frame")

        p, q = self._track(gray, self.ref_points)
        if len(p) < max(8, 0.5 * len(self.ref_points)):
            return self._unknown(timestamp, "insufficient_reference_tracks")
        transform, inliers = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC,
                                                       ransacReprojThreshold=0.8)
        if transform is None or inliers is None or not np.isfinite(transform).all():
            return self._unknown(timestamp, "reference_fit_failed")
        keep = inliers.ravel().astype(bool)
        if keep.sum() < 8 or keep.mean() < 0.75:
            return self._unknown(timestamp, "reference_inconsistent")
        coverage = np.ptp(p[keep], axis=0) / np.array(self.reference[2:])
        if np.min(coverage) < 0.4:
            return self._unknown(timestamp, "reference_points_too_clustered")
        fit_error = np.linalg.norm(q[keep] - (p[keep] @ transform[:, :2].T + transform[:, 2]), axis=1)
        scale = float(np.linalg.norm(transform[:, 0]))
        angle = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
        if not 0.95 <= scale <= 1.05 or abs(angle) > 5 or np.percentile(fit_error, 90) > 0.6:
            return self._unknown(timestamp, "reference_geometry_out_of_range")
        cells = []
        for i, (rect, points) in enumerate(self.cells):
            old, new = self._track(gray, points)
            cell = dict(id=i, roi=list(rect), state="UNRELIABLE", displacement_px=None,
                        magnitude_px=None, dominant_hz=None, tracked_points=len(old))
            if len(old) >= max(6, 0.6 * len(points)):
                residual = new - (old @ transform[:, :2].T + transform[:, 2])
                displacement = np.median(residual, axis=0)
                spread = float(np.median(np.linalg.norm(residual - displacement, axis=1)))
                # Poorly agreeing points may be occlusion, mixed objects or deformation.
                if spread < 0.75 and np.linalg.norm(displacement) < 20:
                    magnitude = float(np.linalg.norm(displacement))
                    history = self.history[i]
                    history.append((timestamp, *displacement))
                    while history and timestamp - history[0][0] > 10:
                        history.popleft()
                    cell.update(state="MOTION_ABOVE_THRESHOLD" if magnitude > self.threshold else "NO_MOTION_ABOVE_THRESHOLD",
                                displacement_px=displacement.tolist(), magnitude_px=magnitude,
                                dominant_hz=self._frequency(history))
            if cell["state"] == "UNRELIABLE":
                self.history[i].clear()
            cells.append(cell)
        # An alert in a measurable cell remains visible even if other cells are lost.
        states = [c["state"] for c in cells]
        state = ("MOTION_ABOVE_THRESHOLD" if "MOTION_ABOVE_THRESHOLD" in states else
                 "UNRELIABLE" if "UNRELIABLE" in states else "NO_MOTION_ABOVE_THRESHOLD")
        return dict(timestamp=timestamp, state=state, threshold_px=self.threshold,
                    reason="image_motion_only", reference=dict(transform=transform.tolist(),
                    tracked_points=len(p), inlier_fraction=float(keep.mean()),
                    fit_error_px=float(np.median(fit_error))), cells=cells)


def annotate(frame, report):
    """Draw a discrete motion map in current-frame coordinates; gray means unknown."""
    output = frame.copy()
    transform = report.get("reference")
    matrix = None if transform is None else np.array(transform["transform"])
    for cell in report["cells"]:
        x, y, w, h = cell["roi"]
        corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=float)
        if matrix is not None:
            corners = corners @ matrix[:, :2].T + matrix[:, 2]
            if cell["displacement_px"] is not None:
                corners += cell["displacement_px"]
        color = (130, 130, 130) if cell["state"] == "UNRELIABLE" else (
            (0, 120, 255) if cell["state"] == "MOTION_ABOVE_THRESHOLD" else (220, 180, 40))
        cv2.polylines(output, [corners.astype(np.int32)], True, color, 2)
        label = "unknown" if cell["magnitude_px"] is None else f'{cell["magnitude_px"]:.2f}px'
        cv2.putText(output, label, tuple(corners[0].astype(int) + (4, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(output, report["state"], (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(output, "IMAGE MOTION ONLY - NOT COLLAPSE RISK", (10, frame.shape[0]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
    return output
