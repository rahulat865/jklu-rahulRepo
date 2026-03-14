"""
app.py — Streamlit dashboard for AI-Powered Exhibit Monitoring (Multi-Camera).

Uses background threads so ALL cameras can stream concurrently.
Global Start All / Stop All / Capture Baseline buttons control every camera.

Human-aware pipeline:
  - Person detections are EXCLUDED from misplacement / foreign-object tracking
  - Persons are masked out before pixel-level SSIM comparison
  - 5-second occlusion buffer prevents false "missing" alerts

Run with:  streamlit run app.py
"""

import cv2
import time
import threading
import numpy as np
import streamlit as st
from PIL import Image

from detector import DamageDetector, UNWANTED_CLASSES, PERSON_CLASS
from comparator import compare_images, ExhibitTracker
from utils import trigger_alert


# ═══════════════════════════════════════════════════════════════════════════
# CameraThread — background frame grabber
# ═══════════════════════════════════════════════════════════════════════════

class CameraThread:
    """Runs cv2.VideoCapture in a daemon thread so the main loop stays fast."""

    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self._thread = None

    def start(self):
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self.cap.isOpened():
            self.cap.release()
        self.frame = None

    def is_opened(self):
        return self.cap.isOpened() and self.running


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def make_camera(name: str, cam_type: str, source) -> dict:
    return {
        "name": name,
        "type": cam_type,
        "source": source,
        "baseline": None,
        "baseline_stabilize_frames": 0,
    }


def process_frame(frame, cam, detector, tracker, crack_sensitivity,
                  ssim_threshold, movement_threshold):
    """
    Run the full detection pipeline on one frame.
    Returns (annotated, alerts_list, metrics_dict).

    KEY CHANGE: all detection logic now ignores the "person" class.
    """
    cam_name = cam["name"]
    alerts = []

    # ── Step 1: Detect ALL objects (including persons) ──────────────
    all_detections = detector.detect_scene_objects(frame)

    # Separate persons from non-persons
    person_detections = [d for d in all_detections if d["label"] == PERSON_CLASS]
    non_person_detections = [d for d in all_detections if d["label"] != PERSON_CLASS]
    person_count = len(person_detections)

    # ── Step 2: Unwanted / foreign object detection (YOLO) ─────────
    unwanted_detections = [d for d in non_person_detections if d["label"] in UNWANTED_CLASSES]
    annotated = detector.draw_object_detections(frame, unwanted_detections)

    # Draw person bounding boxes in a subtle blue (informational only)
    for p in person_detections:
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 1)
        cv2.putText(annotated, "person", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)

    # ── Step 3: Crack detection (only when no baseline) ────────────
    crack_found, crack_count = False, 0
    if cam["baseline"] is None:
        annotated, crack_found, crack_count = detector.detect_cracks(
            annotated, sensitivity=crack_sensitivity,
        )

    # ── Step 4: Baseline comparison (person-masked) ────────────────
    ssim_score = None
    moved_objects, foreign_objects, missing_objects = [], [], []

    if cam["baseline"] is not None:
        if cam["baseline_stabilize_frames"] > 0:
            cam["baseline_stabilize_frames"] -= 1
        else:
            # Update tracker movement threshold from slider
            tracker.movement_threshold = movement_threshold

            # --- ExhibitTracker: misplacement + foreign + missing ---
            # Pass ALL detections and person detections separately
            moved_objects, foreign_objects, missing_objects = tracker.update(
                all_detections, person_detections
            )

            # --- Calculate bboxes to ignore for structural damage ---
            ignore_bboxes = []
            for m in moved_objects:
                ignore_bboxes.append(m["bbox"])
                if "baseline_bbox" in m:
                    ignore_bboxes.append(m["baseline_bbox"])
            for f in foreign_objects:
                ignore_bboxes.append(f["bbox"])
            for m in missing_objects:
                ignore_bboxes.append(m["bbox"])

            # --- SSIM comparison with person and object masking ---
            # Mask out persons from BOTH baseline and current frame
            # so people walking don't trigger structural-change alerts
            masked_baseline = detector.mask_persons(cam["baseline"], person_detections)
            masked_current = detector.mask_persons(frame, person_detections)
            
            # Mask out moved, foreign, and missing object bounding boxes 
            # so they don't trigger false structural damage alerts
            masked_baseline = detector.mask_bboxes(masked_baseline, ignore_bboxes)
            masked_current = detector.mask_bboxes(masked_current, ignore_bboxes)

            result = compare_images(
                masked_baseline, masked_current, ssim_threshold,
                method="absdiff", blur_ksize=21,
                min_change_area=800, diff_threshold=30,
            )
            ssim_score = result["ssim_score"]

            # Draw SSIM change contours (structural damage indicators)
            if result["changed_contours"]:
                for cnt in result["changed_contours"]:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(annotated, "STRUCTURAL CHANGE",
                                (x, max(14, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw moved objects (orange)
            for moved in moved_objects:
                x1, y1, x2, y2 = moved["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(annotated, f"MOVED {moved['label']} ({moved['shift']:.0f}px)",
                            (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            # Draw foreign objects (green)
            for foreign in foreign_objects:
                x1, y1, x2, y2 = foreign["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"FOREIGN {foreign['label']}",
                            (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw missing/removed objects (red)
            for missing in missing_objects:
                x1, y1, x2, y2 = missing["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated,
                            f"REMOVED/MISPLACED {missing['label']} ({missing['missing_seconds']:.0f}s)",
                            (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # ── Generate alerts ────────────────────────────────────
            if result["damage_detected"]:
                num = len(result.get("changed_contours", []))
                alerts.append(trigger_alert(
                    f"[{cam_name}] STRUCTURAL CHANGE — {num} region(s)",
                    severity="HIGH",
                ))

    # Unwanted object alerts
    if unwanted_detections:
        labels = ", ".join(d["label"] for d in unwanted_detections)
        alerts.append(trigger_alert(f"[{cam_name}] Unwanted objects: {labels}"))

    # Crack alerts
    if crack_found:
        alerts.append(trigger_alert(f"[{cam_name}] Cracks detected ({crack_count} regions)"))

    # Misplacement alerts
    if moved_objects:
        labels = ", ".join(f"{m['label']}({m['shift']:.0f}px)" for m in moved_objects)
        alerts.append(trigger_alert(
            f"[{cam_name}] MISPLACED: {labels}",
            severity="HIGH",
        ))

    # Foreign object alerts
    if foreign_objects:
        labels = ", ".join(f["label"] for f in foreign_objects)
        alerts.append(trigger_alert(
            f"[{cam_name}] FOREIGN OBJECT: {labels}",
            severity="HIGH",
        ))

    # Missing/Removed object alerts
    if missing_objects:
        labels = ", ".join(f"{m['label']}({m['missing_seconds']:.0f}s)" for m in missing_objects)
        alerts.append(trigger_alert(
            f"[{cam_name}] REMOVED/MISPLACED: {labels}",
            severity="CRITICAL",
        ))

    return annotated, alerts, {
        "objects": len(unwanted_detections),
        "cracks": crack_count if crack_found else 0,
        "ssim": ssim_score,
        "persons": person_count,
        "moved": len(moved_objects),
        "foreign": len(foreign_objects),
        "missing": len(missing_objects),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Page config & CSS
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AI-Powered Exhibit Monitoring System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 1.2rem 2rem; border-radius: 12px;
        margin-bottom: 1.5rem; text-align: center;
    }
    .main-header h1 { color: #fff; margin: 0; font-size: 2rem; letter-spacing: 1px; }
    .main-header p  { color: #b0b0d0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    .metric-card {
        background: #1e1e2f; border: 1px solid #3a3a5c;
        border-radius: 10px; padding: 0.7rem; text-align: center; margin-bottom: 0.5rem;
    }
    .metric-card h3 { color: #a78bfa; margin: 0; font-size: 0.75rem; text-transform: uppercase; }
    .metric-card .value { color: #fff; font-size: 1.4rem; font-weight: 700; }

    .alert-box {
        background: #2d1b1b; border-left: 4px solid #ef4444;
        border-radius: 6px; padding: 0.6rem 0.8rem; margin-bottom: 0.5rem;
        color: #fca5a5; font-size: 0.82rem;
    }
    .alert-box .ts { color: #888; font-size: 0.7rem; }

    .safe-box {
        background: #1b2d1b; border-left: 4px solid #22c55e;
        border-radius: 6px; padding: 0.8rem 1rem; color: #86efac; font-size: 0.9rem;
    }

    .cam-card {
        background: #1a1a2e; border: 1px solid #3a3a5c;
        border-radius: 8px; padding: 0.6rem 0.8rem; margin-bottom: 0.4rem;
    }
    .cam-card .cam-name { color: #e0e0ff; font-weight: 600; font-size: 0.9rem; }
    .cam-card .cam-type { color: #888; font-size: 0.75rem; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    [data-testid="stAppDeployButton"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>AI-Powered Exhibit Monitoring System</h1>
    <p>Multi-camera real-time detection — humans ignored for exhibit tracking</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════

if "detector" not in st.session_state:
    st.session_state.detector = DamageDetector()
if "cameras" not in st.session_state:
    st.session_state.cameras = [make_camera("Camera 1", "Laptop Webcam", 0)]
if "cam_threads" not in st.session_state:
    st.session_state.cam_threads = {}          # name → CameraThread
if "cam_trackers" not in st.session_state:
    st.session_state.cam_trackers = {}         # name → ExhibitTracker
if "all_running" not in st.session_state:
    st.session_state.all_running = False
if "alerts" not in st.session_state:
    st.session_state.alerts = []

detector: DamageDetector = st.session_state.detector


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Controls")

    # ── Add Camera ──────────────────────────────────────────────────
    with st.expander("➕ Add Camera", expanded=False):
        new_name = st.text_input("Camera Name",
                                  value=f"Camera {len(st.session_state.cameras) + 1}",
                                  key="new_cam_name")
        new_type = st.radio("Camera Type", ["Laptop Webcam", "IP Webcam"], key="new_cam_type")
        if new_type == "Laptop Webcam":
            new_source = int(st.number_input("Camera index", 0, 10, 0, key="new_cam_idx"))
        else:
            new_source = st.text_input("IP Webcam URL",
                                        value="http://192.168.1.5:8080/video",
                                        key="new_cam_url")
        if st.button("✅ Add Camera", key="add_cam_btn"):
            st.session_state.cameras.append(make_camera(new_name, new_type, new_source))
            st.success(f"Added: {new_name}")
            st.rerun()

    # ── Camera List ─────────────────────────────────────────────────
    st.subheader("📹 Cameras")
    for i, cam in enumerate(st.session_state.cameras):
        running = cam["name"] in st.session_state.cam_threads and st.session_state.cam_threads[cam["name"]].running
        status = "🟢 Live" if running else "⚪ Stopped"
        bl = "✅ Baseline" if cam["baseline"] is not None else "No baseline"
        st.markdown(
            f'<div class="cam-card">'
            f'<span class="cam-name">{cam["name"]} — {status}</span><br>'
            f'<span class="cam-type">{cam["type"]} | {bl}</span>'
            f'</div>', unsafe_allow_html=True,
        )
        if st.button(f"❌ Remove {cam['name']}", key=f"rm_{i}"):
            thread = st.session_state.cam_threads.pop(cam["name"], None)
            if thread:
                thread.stop()
            st.session_state.cam_trackers.pop(cam["name"], None)
            st.session_state.cameras.pop(i)
            st.rerun()

    if not st.session_state.cameras:
        st.info("No cameras. Use ➕ Add Camera.")

    # ── Detection Settings ──────────────────────────────────────────
    st.subheader("🎯 Detection Settings")
    confidence = st.slider("YOLO confidence", 0.1, 1.0, 0.40, 0.05)
    detector.confidence = confidence
    crack_sensitivity = st.select_slider("Crack sensitivity",
                                          options=["low", "medium", "high"], value="high")
    ssim_threshold = st.slider("SSIM threshold", 0.50, 1.0, 0.85, 0.05)
    movement_threshold = st.slider("Movement threshold (px)", 20, 200, 50, 5)
    occlusion_timeout = st.slider("Occlusion timeout (sec)", 1, 15, 5, 1)

    if st.button("🗑️ Clear All Alerts"):
        st.session_state.alerts = []


# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════

tab_camera, tab_upload, tab_compare = st.tabs([
    "📹 Live Cameras", "📤 Upload Image", "🔍 Before / After"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Multi-camera live view
# ═══════════════════════════════════════════════════════════════════════════
with tab_camera:
    if not st.session_state.cameras:
        st.warning("No cameras configured. Add one from the sidebar.")
    else:
        # ── Global control buttons ──────────────────────────────────
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            start_all = st.button("▶️ Start All Cameras", key="start_all", use_container_width=True)
        with btn_col2:
            stop_all = st.button("⏹️ Stop All Cameras", key="stop_all", use_container_width=True)
        with btn_col3:
            capture_all = st.button("📸 Capture Baseline (All)", key="capture_all", use_container_width=True)

        # ── Alerts column ───────────────────────────────────────────
        main_col, alert_col = st.columns([4, 1])

        with alert_col:
            st.subheader("🚨 Alerts")
            alert_placeholder = st.empty()

        # ── Handle Stop All ─────────────────────────────────────────
        if stop_all:
            st.session_state.all_running = False
            for name, thread in st.session_state.cam_threads.items():
                thread.stop()
            st.session_state.cam_threads = {}
            st.session_state.cam_trackers = {}
            st.info("⏹️ All cameras stopped.")

        # ── Handle Capture All (from last frames) ───────────────────
        if capture_all:
            for cam in st.session_state.cameras:
                thread = st.session_state.cam_threads.get(cam["name"])
                if thread and thread.running:
                    frame = thread.read()
                    if frame is not None:
                        cam["baseline"] = frame.copy()
                        cam["baseline_stabilize_frames"] = 5

                        # Create / reset ExhibitTracker for this camera
                        tracker = ExhibitTracker(
                            movement_threshold=movement_threshold,
                            occlusion_timeout=occlusion_timeout,
                            min_confidence=detector.confidence,
                        )
                        # Capture baseline objects (persons are auto-filtered)
                        baseline_detections = detector.detect_scene_objects(frame)
                        tracker.set_baseline(baseline_detections)
                        st.session_state.cam_trackers[cam["name"]] = tracker

            st.success("📸 Baselines captured for all active cameras! (persons excluded)")

        # ── Handle Start All ────────────────────────────────────────
        if start_all:
            for cam in st.session_state.cameras:
                name = cam["name"]
                if name not in st.session_state.cam_threads or not st.session_state.cam_threads[name].running:
                    thread = CameraThread(cam["source"])
                    ok = thread.start()
                    if ok:
                        st.session_state.cam_threads[name] = thread
                    else:
                        st.error(f"❌ Cannot open {name} ({cam['source']})")

                # Ensure every camera has a tracker
                if name not in st.session_state.cam_trackers:
                    st.session_state.cam_trackers[name] = ExhibitTracker(
                        movement_threshold=movement_threshold,
                        occlusion_timeout=occlusion_timeout,
                        min_confidence=detector.confidence,
                    )

            st.session_state.all_running = True

            # ── Create placeholders in a grid ───────────────────────
            with main_col:
                num_cams = len(st.session_state.cameras)
                cols_per_row = 2 if num_cams > 1 else 1
                placeholders = {}
                metric_placeholders = {}

                rows_needed = (num_cams + cols_per_row - 1) // cols_per_row
                cam_idx = 0
                for _ in range(rows_needed):
                    cols = st.columns(cols_per_row)
                    for col in cols:
                        if cam_idx < num_cams:
                            cam = st.session_state.cameras[cam_idx]
                            with col:
                                st.markdown(f"**📹 {cam['name']}**")
                                placeholders[cam["name"]] = st.empty()
                                mc1, mc2, mc3, mc4 = st.columns(4)
                                with mc1:
                                    metric_placeholders[f"{cam['name']}_obj"] = st.empty()
                                with mc2:
                                    metric_placeholders[f"{cam['name']}_crack"] = st.empty()
                                with mc3:
                                    metric_placeholders[f"{cam['name']}_ssim"] = st.empty()
                                with mc4:
                                    metric_placeholders[f"{cam['name']}_persons"] = st.empty()
                            cam_idx += 1

            # ── Main streaming loop ─────────────────────────────────
            while st.session_state.all_running:
                any_active = False

                for cam in st.session_state.cameras:
                    name = cam["name"]
                    thread = st.session_state.cam_threads.get(name)
                    if thread is None or not thread.running:
                        continue

                    frame = thread.read()
                    if frame is None:
                        continue

                    any_active = True

                    # Get or create tracker for this camera
                    tracker = st.session_state.cam_trackers.get(name)
                    if tracker is None:
                        tracker = ExhibitTracker(
                            movement_threshold=movement_threshold,
                            occlusion_timeout=occlusion_timeout,
                            min_confidence=detector.confidence,
                        )
                        st.session_state.cam_trackers[name] = tracker

                    # Update occlusion timeout from slider
                    tracker.occlusion_timeout = occlusion_timeout

                    # Run detection pipeline
                    annotated, new_alerts, metrics = process_frame(
                        frame, cam, detector, tracker,
                        crack_sensitivity, ssim_threshold, movement_threshold,
                    )

                    # Accumulate alerts
                    for a in new_alerts:
                        st.session_state.alerts.insert(0, a)
                    st.session_state.alerts = st.session_state.alerts[:50]

                    # Render camera feed
                    if name in placeholders:
                        placeholders[name].image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            channels="RGB", use_container_width=True,
                        )

                    # Render per-camera metrics
                    obj_key = f"{name}_obj"
                    crack_key = f"{name}_crack"
                    ssim_key = f"{name}_ssim"
                    persons_key = f"{name}_persons"

                    if obj_key in metric_placeholders:
                        metric_placeholders[obj_key].markdown(
                            f'<div class="metric-card"><h3>Objects</h3>'
                            f'<div class="value">{metrics["objects"]}</div></div>',
                            unsafe_allow_html=True,
                        )
                    if crack_key in metric_placeholders:
                        metric_placeholders[crack_key].markdown(
                            f'<div class="metric-card"><h3>Cracks</h3>'
                            f'<div class="value">{metrics["cracks"]}</div></div>',
                            unsafe_allow_html=True,
                        )
                    if ssim_key in metric_placeholders:
                        s = f'{metrics["ssim"]:.2%}' if metrics["ssim"] is not None else "N/A"
                        metric_placeholders[ssim_key].markdown(
                            f'<div class="metric-card"><h3>SSIM</h3>'
                            f'<div class="value">{s}</div></div>',
                            unsafe_allow_html=True,
                        )
                    if persons_key in metric_placeholders:
                        metric_placeholders[persons_key].markdown(
                            f'<div class="metric-card"><h3>Persons</h3>'
                            f'<div class="value">{metrics["persons"]}</div></div>',
                            unsafe_allow_html=True,
                        )

                # Render alerts
                alert_html = ""
                for a in st.session_state.alerts[:15]:
                    alert_html += (
                        f'<div class="alert-box">'
                        f'<span class="ts">{a["timestamp"]}</span><br>{a["message"]}'
                        f'</div>'
                    )
                if not st.session_state.alerts:
                    alert_html = '<div class="safe-box">✅ All clear — no issues detected</div>'
                alert_placeholder.markdown(alert_html, unsafe_allow_html=True)

                if not any_active:
                    break

                time.sleep(0.15)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Upload Image for detection
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.subheader("Upload an Image for Detection")
    uploaded_img = st.file_uploader("Choose an image…", type=["png", "jpg", "jpeg"],
                                    key="upload_detect")
    if uploaded_img:
        file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     caption="Original Image", use_container_width=True)

        annotated, detections = detector.detect_objects(img)
        annotated, crack_found, crack_count = detector.detect_cracks(
            annotated, sensitivity=crack_sensitivity,
        )

        with col_det:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption="Detection Results", use_container_width=True)

        if detections or crack_found:
            st.error("🚨 **Damage / Anomaly Detected!**")
            if detections:
                st.write("**Unwanted Objects:**")
                for d in detections:
                    st.write(f"- **{d['label']}** — confidence {d['confidence']:.0%}")
            if crack_found:
                st.write(f"**Possible Cracks:** {crack_count} region(s)")
        else:
            st.success("✅ No damage or unwanted objects detected.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Before / After Comparison
# ═══════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.subheader("Before / After Comparison")

    cam_names = [c["name"] for c in st.session_state.cameras]
    compare_cam_idx = st.selectbox("Compare using baseline from:", range(len(cam_names)),
                                    format_func=lambda i: cam_names[i], key="compare_cam")
    compare_cam = st.session_state.cameras[compare_cam_idx] if cam_names else None

    col_before, col_after = st.columns(2)

    with col_before:
        st.write("**Baseline (Before)**")
        baseline_up = st.file_uploader("Upload baseline", type=["png", "jpg", "jpeg"],
                                        key="bl_upload")
        if baseline_up:
            file_bytes = np.asarray(bytearray(baseline_up.read()), dtype=np.uint8)
            baseline_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif compare_cam and compare_cam["baseline"] is not None:
            baseline_img = compare_cam["baseline"]
        else:
            baseline_img = None

        if baseline_img is not None:
            st.image(cv2.cvtColor(baseline_img, cv2.COLOR_BGR2RGB),
                     caption="Baseline", use_container_width=True)
        else:
            st.info("Upload or capture a baseline image first.")

    with col_after:
        st.write("**Current (After)**")
        current_up = st.file_uploader("Upload current image", type=["png", "jpg", "jpeg"],
                                       key="curr_upload")
        if current_up:
            file_bytes = np.asarray(bytearray(current_up.read()), dtype=np.uint8)
            current_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB),
                     caption="Current", use_container_width=True)
        else:
            current_img = None
            st.info("Upload a current image to compare.")

    if baseline_img is not None and current_img is not None:
        st.markdown("---")
        result = compare_images(
            baseline_img, current_img, ssim_threshold,
            method="absdiff", blur_ksize=21,
            min_change_area=800, diff_threshold=30,
        )

        col_diff, col_metrics = st.columns([3, 1])
        with col_diff:
            st.image(cv2.cvtColor(result["diff_image"], cv2.COLOR_BGR2RGB),
                     caption="Difference Heatmap", use_container_width=True)
        with col_metrics:
            score = result["ssim_score"]
            st.markdown(
                f'<div class="metric-card"><h3>SSIM Score</h3>'
                f'<div class="value">{score:.2%}</div></div>',
                unsafe_allow_html=True,
            )
            if result["damage_detected"]:
                st.error(f"🚨 Change detected! SSIM {score:.2%}")
            else:
                st.success(f"✅ Stable. SSIM {score:.2%}")
