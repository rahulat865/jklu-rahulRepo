"""
comparator.py — Baseline vs current image comparison (SSIM / absdiff)
                + ExhibitTracker for human-aware misplacement detection.

The ExhibitTracker class:
  - Stores baseline object positions (only non-person detections)
  - Compares each new frame's detections against the baseline
  - Ignores ALL "person" detections so walking humans never trigger alerts
  - Implements a 5-second occlusion buffer: if an exhibit disappears
    (e.g. person standing in front), it waits 5 seconds before firing
    a "missing" alert
"""

import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ═══════════════════════════════════════════════════════════════════════════
# Pixel-level comparison helpers (unchanged — used for crack / damage diff)
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_for_diff(image: np.ndarray, blur_ksize: int = 21) -> np.ndarray:
    """Convert frame to grayscale and denoise with Gaussian blur."""
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)


def _extract_changed_regions(binary_mask: np.ndarray, min_change_area: int = 800):
    """Return contours and total changed area after area filtering."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept_contours = []
    total_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_change_area:
            continue
        kept_contours.append(cnt)
        total_area += int(area)

    return kept_contours, total_area


def compute_ssim(baseline: np.ndarray, current: np.ndarray, blur_ksize: int = 21):
    """
    Compute Structural Similarity Index between two images.
    Returns:
        score    : float in [0, 1]  (1 = identical)
        diff_img : absolute difference heatmap (BGR, coloured)
    """
    h, w = baseline.shape[:2]
    current_resized = cv2.resize(current, (w, h))

    gray_base = _preprocess_for_diff(baseline, blur_ksize=blur_ksize)
    gray_curr = _preprocess_for_diff(current_resized, blur_ksize=blur_ksize)

    score, diff = ssim(gray_base, gray_curr, full=True)
    diff = (diff * 255).astype(np.uint8)
    diff_coloured = cv2.applyColorMap(255 - diff, cv2.COLORMAP_JET)

    return round(score, 4), diff_coloured, (255 - diff)


def compute_absdiff(baseline: np.ndarray, current: np.ndarray, blur_ksize: int = 21):
    """Compute absolute grayscale difference after denoising."""
    h, w = baseline.shape[:2]
    current_resized = cv2.resize(current, (w, h))

    gray_base = _preprocess_for_diff(baseline, blur_ksize=blur_ksize)
    gray_curr = _preprocess_for_diff(current_resized, blur_ksize=blur_ksize)

    abs_diff = cv2.absdiff(gray_base, gray_curr)
    score = 1.0 - (float(np.mean(abs_diff)) / 255.0)
    diff_coloured = cv2.applyColorMap(abs_diff, cv2.COLORMAP_JET)
    return round(max(0.0, min(1.0, score)), 4), diff_coloured, abs_diff


def compare_images(baseline: np.ndarray, current: np.ndarray,
                   threshold: float = 0.85,
                   method: str = "absdiff",
                   blur_ksize: int = 21,
                   min_change_area: int = 2000,
                   diff_threshold: int = 40):
    """High-level comparison returning a result dict."""
    if method == "absdiff":
        score, diff_img, diff_gray = compute_absdiff(baseline, current, blur_ksize=blur_ksize)
    else:
        score, diff_img, diff_gray = compute_ssim(baseline, current, blur_ksize=blur_ksize)

    _, binary_mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)
    binary_mask = cv2.dilate(binary_mask, None, iterations=2)

    changed_contours, changed_area = _extract_changed_regions(binary_mask, min_change_area=min_change_area)
    if method == "ssim":
        score_indicates_change = score < threshold
        damage_detected = score_indicates_change and (changed_area > 0)
    else:
        damage_detected = changed_area > 0

    return {
        "ssim_score": score,
        "threshold": threshold,
        "damage_detected": damage_detected,
        "diff_image": diff_img,
        "binary_mask": binary_mask,
        "changed_contours": changed_contours,
        "changed_area": changed_area,
        "method": method,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ExhibitTracker — human-aware misplacement & foreign-object detection
# ═══════════════════════════════════════════════════════════════════════════

# Classes that must ALWAYS be ignored for tracking
PERSON_CLASS = "person"

# Classes considered foreign / unwanted in a protected museum area
FOREIGN_CLASSES = {
    "bottle", "cup", "fork", "knife", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "teddy bear",
}


class ExhibitTracker:
    """
    Tracks museum exhibit objects across frames, ignoring humans.

    Workflow:
        1. Call set_baseline() once with the initial frame's detections.
        2. Call update() on every subsequent frame.
        3. Read the returned (moved, foreign, missing) lists.

    Occlusion tolerance:
        If a baseline object is not detected in the current frame but a
        person IS nearby (within the object's bounding-box area), we assume
        temporary occlusion and wait up to `occlusion_timeout` seconds
        before raising a missing alert.
    """

    def __init__(self, movement_threshold: float = 50.0,
                 occlusion_timeout: float = 5.0,
                 min_confidence: float = 0.35):
        """
        Args:
            movement_threshold: pixel displacement to qualify as "moved"
            occlusion_timeout:  seconds to wait before declaring "missing"
            min_confidence:     minimum YOLO confidence to consider a detection
        """
        self.movement_threshold = movement_threshold
        self.occlusion_timeout = occlusion_timeout
        self.min_confidence = min_confidence

        # Baseline storage:  list[dict] with label, bbox, center
        self.baseline_objects: list[dict] = []
        # Track when each baseline object was last seen
        # Maps index → timestamp (float)
        self.last_seen: dict[int, float] = {}
        self.baseline_set = False

    def set_baseline(self, detections: list[dict]):
        """
        Store baseline object positions.
        AUTOMATICALLY filters out person detections.

        Args:
            detections: list of detection dicts from detector.detect_scene_objects()
        """
        # Filter out persons — only track non-person objects
        self.baseline_objects = [
            d for d in detections
            if d["label"] != PERSON_CLASS and d["confidence"] >= self.min_confidence
        ]
        # Initialize last_seen to current time for all baseline objects
        now = time.time()
        self.last_seen = {i: now for i in range(len(self.baseline_objects))}
        self.baseline_set = True

    def update(self, current_detections: list[dict], person_detections: list[dict] = None):
        """
        Compare current frame detections against baseline.
        Persons are ALWAYS filtered out from current_detections.

        Args:
            current_detections: all detections from the current frame
            person_detections:  person-only detections (for occlusion check)

        Returns:
            moved_objects   : list[dict] — baseline objects that shifted > threshold
            foreign_objects : list[dict] — new objects not in baseline (unwanted classes)
            missing_objects : list[dict] — baseline objects gone > occlusion_timeout
        """
        if not self.baseline_set:
            return [], [], []

        if person_detections is None:
            person_detections = []

        # ── Step 1: Filter out persons from current detections ──────────
        current_non_person = [
            d for d in current_detections
            if d["label"] != PERSON_CLASS and d["confidence"] >= self.min_confidence
        ]

        now = time.time()
        moved_objects = []
        foreign_objects = []
        missing_objects = []

        # ── Step 2: Match current detections to baseline ────────────────
        # For each baseline object, find the closest current detection
        # with the same label
        matched_current_indices = set()

        for b_idx, baseline in enumerate(self.baseline_objects):
            b_label = baseline["label"]
            b_cx, b_cy = baseline["center"]

            # Find all current detections with the same label
            best_match_idx = None
            best_distance = float("inf")

            for c_idx, current in enumerate(current_non_person):
                if c_idx in matched_current_indices:
                    continue
                if current["label"] != b_label:
                    continue
                cx, cy = current["center"]
                dist = ((cx - b_cx) ** 2 + (cy - b_cy) ** 2) ** 0.5
                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = c_idx

            if best_match_idx is not None:
                matched_current_indices.add(best_match_idx)
                # Object found — update last_seen
                self.last_seen[b_idx] = now

                # Check if it moved beyond threshold
                if best_distance > self.movement_threshold:
                    moved = current_non_person[best_match_idx].copy()
                    moved["shift"] = round(best_distance, 1)
                    moved["baseline_center"] = (b_cx, b_cy)
                    moved["baseline_bbox"] = baseline["bbox"]
                    moved_objects.append(moved)
            else:
                # Object NOT found in current frame
                # Check if a person is occluding it (person bbox overlaps
                # the baseline object's bbox)
                is_occluded = self._is_occluded_by_person(baseline, person_detections)

                if is_occluded:
                    # Person is blocking the view — don't count time
                    self.last_seen[b_idx] = now
                else:
                    # Check how long it has been missing
                    elapsed = now - self.last_seen.get(b_idx, now)
                    if elapsed > self.occlusion_timeout:
                        missing = baseline.copy()
                        missing["missing_seconds"] = round(elapsed, 1)
                        missing_objects.append(missing)

        # ── Step 3: Detect foreign objects ──────────────────────────────
        # Any current detection NOT matched to a baseline object
        # and belonging to FOREIGN_CLASSES
        for c_idx, current in enumerate(current_non_person):
            if c_idx in matched_current_indices:
                continue
            if current["label"] in FOREIGN_CLASSES:
                foreign_objects.append(current)

        return moved_objects, foreign_objects, missing_objects

    @staticmethod
    def _is_occluded_by_person(obj: dict, person_detections: list[dict]) -> bool:
        """
        Check if a person's bounding box overlaps significantly with
        the object's baseline bounding box (IoU-based check).
        """
        if not person_detections:
            return False

        ox1, oy1, ox2, oy2 = obj["bbox"]
        obj_area = max((ox2 - ox1) * (oy2 - oy1), 1)

        for person in person_detections:
            px1, py1, px2, py2 = person["bbox"]

            # Compute intersection
            ix1 = max(ox1, px1)
            iy1 = max(oy1, py1)
            ix2 = min(ox2, px2)
            iy2 = min(oy2, py2)

            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                # If person covers ≥30% of the object area, consider it occluded
                if intersection / obj_area >= 0.30:
                    return True

        return False
