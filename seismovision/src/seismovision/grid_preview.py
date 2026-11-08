"""Live user-bounded building grid preview."""

import argparse
from dataclasses import replace
from typing import Dict, Optional

import cv2
import numpy as np

from .capture import make_source
from .config import AnalysisConfig, CameraConfig, MagnificationConfig, Rect, load_grid_config
from .grid_mapping import BuildingGridMapper, GridCellState, grid_cells, refine_building_box
from .magnification import RoiMotionMagnifier


def _select_regions(frame: np.ndarray) -> tuple[Rect, Rect]:
    seed = _select_roi(frame, "Draw rough building box, then Enter")
    if seed is None:
        raise RuntimeError("A building box is required")
    reference = _select_roi(frame, "Draw stable reference box, then Enter")
    if reference is None:
        raise RuntimeError("A stable reference box is required")
    return refine_building_box(frame, seed), reference


def _select_roi(frame: np.ndarray, title: str) -> Optional[Rect]:
    roi = tuple(int(value) for value in cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False))
    cv2.destroyWindow(title)
    return roi if roi[2] > 0 and roi[3] > 0 else None  # type: ignore[return-value]


def _confirm_regions(frame: np.ndarray, building: Rect, reference: Rect) -> tuple[Rect, Rect, list[Rect]]:
    """Lets the operator edit regions directly on the reference frame."""
    cells = grid_cells(building)
    editor = {"selected": None, "start": None, "draft": None, "replacement": None}

    def mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if editor["selected"] is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            editor["start"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and editor["start"] is not None:
            start_x, start_y = editor["start"]
            editor["draft"] = (min(start_x, x), min(start_y, y), abs(x - start_x), abs(y - start_y))
        elif event == cv2.EVENT_LBUTTONUP and editor["start"] is not None:
            start_x, start_y = editor["start"]
            editor["start"] = None
            editor["draft"] = None
            if abs(x - start_x) > 2 and abs(y - start_y) > 2:
                editor["replacement"] = (min(start_x, x), min(start_y, y), abs(x - start_x), abs(y - start_y))

    cv2.namedWindow("Confirm grid regions")
    cv2.setMouseCallback("Confirm grid regions", mouse)
    while True:
        replacement = editor["replacement"]
        if replacement is not None:
            if editor["selected"] == "building":
                building, cells = replacement, grid_cells(replacement)
            elif editor["selected"] == "reference":
                reference = replacement
            else:
                cells[int(editor["selected"])] = replacement
            editor["selected"] = editor["replacement"] = None
        preview = frame.copy()
        x, y, width, height = building
        cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.putText(preview, "building [b]", (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        x, y, width, height = reference
        cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 255, 0), 2)
        cv2.putText(preview, "reference [s]", (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        for index, (x, y, width, height) in enumerate(cells):
            cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 0), 1)
            cv2.putText(preview, str(index), (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if editor["draft"] is not None:
            x, y, width, height = editor["draft"]
            cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 255, 255), 2)
        selected = editor["selected"]
        message = "Enter: start | b/s/0-7: select box, then drag on this frame | q: quit"
        if selected is not None:
            message = f"Selected {selected}: drag its replacement box on this frame"
        cv2.putText(preview, message, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        cv2.imshow("Confirm grid regions", preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (10, 13):
            cv2.destroyWindow("Confirm grid regions")
            return building, reference, cells
        if key == ord("b"):
            editor["selected"] = "building"
        elif key == ord("s"):
            editor["selected"] = "reference"
        elif ord("0") <= key <= ord("7"):
            editor["selected"] = key - ord("0")
        elif key == ord("q"):
            raise KeyboardInterrupt


def _spectrogram(panel: np.ndarray, bounds: tuple[int, int, int, int], index: int, history: object, max_frequency: float) -> None:
    left, top, right, bottom = bounds
    cv2.putText(panel, f"cell {index}: FFT amplitude", (left + 4, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1)
    graph_top = top + 20
    cv2.rectangle(panel, (left, graph_top), (right, bottom), (100, 100, 100), 1)
    columns = list(history)
    if columns:
        data = np.asarray(columns, dtype=np.float32).T
        limit = float(np.max(np.log1p(data)))
        if limit > 0:
            image = np.uint8(255 * np.log1p(data) / limit)
            image = cv2.applyColorMap(np.flipud(image), cv2.COLORMAP_TURBO)
            image = cv2.resize(image, (right - left - 2, bottom - graph_top - 2), interpolation=cv2.INTER_NEAREST)
            panel[graph_top + 1:bottom - 1, left + 1:right - 1] = image
    cv2.putText(panel, f"{max_frequency:.1f}Hz", (left + 3, graph_top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)
    cv2.putText(panel, "time ->", (right - 48, bottom - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (230, 230, 230), 1)


def _draw_grid(frame: np.ndarray, mapper: BuildingGridMapper, states: Dict[str, GridCellState]) -> np.ndarray:
    output = frame.copy()
    x, y, width, height = mapper.bbox
    cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 255), 2)
    for index, state in enumerate(states.values()):
        x, y, width, height = state.roi
        anomaly = state.anomaly_score is not None and state.anomaly_score >= 1.0
        color = (0, 0, 255) if anomaly else (0, 255, 0)
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)
        for point in state.motion.points:
            cv2.circle(output, tuple(point.astype(int)), 2, color, cv2.FILLED)
        center = (x + width // 2, y + height // 2)
        vector = np.clip(
            state.position * 20.0,
            (-width // 2 + 5, -height // 2 + 18),
            (width // 2 - 5, height // 2 - 5),
        ).astype(int)
        cv2.arrowedLine(output, center, (center[0] + vector[0], center[1] + vector[1]), (0, 255, 255), 2, tipLength=0.25)
        hz = "--" if state.frequency_hz is None else f"{state.frequency_hz:.2f}Hz"
        label = f"{index} {hz}" + (" ANOMALY" if anomaly else "")
        cv2.putText(output, label, (x + 3, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.putText(output, f"{state.position[0]:+.1f}, {state.position[1]:+.1f}px", (x + 3, y + height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
    score = mapper.response_anomaly_score
    message = "response anomaly: collecting"
    color = (0, 180, 255)
    if score is not None:
        message = f"response anomaly: {score:.2f}x sensitivity"
        color = (0, 0, 255) if score >= 1.0 else (0, 220, 0)
    cv2.putText(output, message, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    cv2.putText(output, "screening only - not a stability prediction", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (210, 210, 210), 1)
    return output


def _draw_status(width: int, mapper: BuildingGridMapper, states: Dict[str, GridCellState]) -> np.ndarray:
    """Shows a screening result and data quality without claiming structural safety."""
    panel = np.full((86, width, 3), 18, dtype=np.uint8)
    confidences = [state.motion.confidence for state in states.values()]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    reliable = sum(value >= mapper.analysis.minimum_confidence for value in confidences)
    score = mapper.response_anomaly_score
    if reliable < 3:
        headline, color = "STABILITY SCREEN: MEASUREMENT UNRELIABLE", (0, 180, 255)
    elif score is None:
        headline, color = "STABILITY SCREEN: COLLECTING RESPONSE DATA", (0, 180, 255)
    elif score >= 1.0:
        headline, color = "STABILITY SCREEN: RESPONSE ANOMALY OBSERVED", (0, 0, 255)
    else:
        headline, color = "STABILITY SCREEN: NO RESPONSE ANOMALY CURRENTLY", (0, 220, 0)
    cv2.putText(panel, headline, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    anomaly = "--" if score is None else f"{score:.2f}x configured sensitivity"
    cv2.putText(panel, f"Measurement confidence: {confidence * 100:.0f}% ({reliable}/8 reliable cells) | response anomaly: {anomaly}", (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1)
    cv2.putText(panel, "Tracking confidence is not a probability that the building is stable or unstable.", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="User-bounded 4x2 cardboard-building motion map")
    parser.add_argument("--camera", choices=("picamera2", "usb"), default="picamera2")
    parser.add_argument("--config", help="JSON settings for camera, analysis, and magnification")
    parser.add_argument("--video")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    settings = load_grid_config(args.config) if args.config else None
    camera = settings.camera if settings else CameraConfig(width=640, height=480, fps=30)
    camera = replace(camera, width=args.width or camera.width, height=args.height or camera.height, fps=args.fps or camera.fps)
    analysis = settings.analysis if settings else AnalysisConfig()
    magnification = settings.magnification if settings else MagnificationConfig()
    source = make_source(camera, args.camera, args.video)
    mapper = None
    magnifiers = ()
    moved, settled = False, 0
    try:
        while True:
            ok, frame, timestamp = source.read()
            if not ok or frame is None:
                break
            if mapper is None:
                preview = frame.copy()
                cv2.putText(preview, "Draw building and stable-reference boxes in the selection windows", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
                cv2.imshow("SeismoVision grid map", preview)
                building_box, reference_box = _select_regions(frame)
                building_box, reference_box, cells = _confirm_regions(frame, building_box, reference_box)
                mapper = BuildingGridMapper(building_box, reference_box, analysis, cells)
                mapper.tracker.validate(frame)
                magnifiers = tuple(RoiMotionMagnifier(roi, magnification, camera.fps) for roi in mapper.cells) if magnification.enabled else ()
            else:
                camera_motion, states = mapper.update(frame, timestamp)
                if camera_motion > mapper.analysis.max_camera_motion_px:
                    moved, settled = True, 0
                elif moved:
                    settled += 1
                    if settled >= mapper.analysis.camera_settle_frames:
                        mapper.reset_measurement()
                        for magnifier in magnifiers:
                            magnifier.reset()
                        moved, settled = False, 0
                        continue
                magnified = frame
                for magnifier in magnifiers:
                    magnified = magnifier.apply(magnified, timestamp)
                preview = _draw_grid(magnified, mapper, states)
                dashboard_height = max(frame.shape[0], 720)
                left = np.full((dashboard_height, frame.shape[1], 3), 10, dtype=np.uint8)
                left[:frame.shape[0]] = preview
                right = np.full((dashboard_height, 520, 3), 25, dtype=np.uint8)
                graph_width, graph_height = right.shape[1] // 2, right.shape[0] // 4
                for index, history in enumerate(mapper.spectrogram_history.values()):
                    row, column = divmod(index, 2)
                    _spectrogram(right, (column * graph_width, row * graph_height, (column + 1) * graph_width - 2, (row + 1) * graph_height - 2), index, history, mapper.analysis.max_frequency_hz)
                dashboard = cv2.hconcat((left, right))
                cv2.imshow("SeismoVision grid map", cv2.vconcat((dashboard, _draw_status(dashboard.shape[1], mapper, states))))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                mapper, moved, settled = None, False, 0
                magnifiers = ()
            elif key == ord("q"):
                break
    finally:
        source.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
