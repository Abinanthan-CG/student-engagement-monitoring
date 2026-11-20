import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import threading
import time
from collections import deque

# Page config first
st.set_page_config(
    page_title="Student Engagement Monitor",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# 1. MediaPipe Setup
# -----------------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -----------------------------------------------------------------------------
# 2. Mathematical Utility Functions
# -----------------------------------------------------------------------------

# 3D generic face model points (in mm, relative to nose tip at origin).
# Using 10 well-spread, stable landmark indices for a robust solvePnP estimate.
FACE_MODEL_3D = np.array([
    [0.0,      0.0,      0.0   ],  # 1  - Nose tip
    [0.0,    -63.6,    -12.5   ],  # 152 - Chin
    [-43.3,   32.7,    -26.0   ],  # 33  - Left eye outer corner
    [43.3,    32.7,    -26.0   ],  # 263 - Right eye outer corner
    [-28.9,  -28.9,    -24.1   ],  # 61  - Left mouth corner
    [28.9,   -28.9,    -24.1   ],  # 291 - Right mouth corner
    [-2.0,    -2.0,    -10.0   ],  # 4   - Nose base
    [0.0,     27.1,    -39.5   ],  # 8   - Upper lip
    [-20.0,   40.0,    -35.0   ],  # 468 - Left inner brow
    [20.0,    40.0,    -35.0   ],  # 473 - Right inner brow
], dtype=np.float64)

FACE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291, 4, 8, 468, 473]

def calculate_ear(eye_landmarks):
    """
    Eye Aspect Ratio: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    Detects eye closure — low EAR for 25+ frames = Drowsy.
    """
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    h  = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    if h < 1e-6:
        return 0.3  # safe fallback
    return (v1 + v2) / (2.0 * h)


def get_head_pose(face_landmarks, image_shape):
    """
    Compute Yaw, Pitch, Roll from solvePnP using direct atan2 decomposition
    of the rotation matrix — much more reliable than decomposeProjectionMatrix.

    Returns:
        (yaw, pitch, roll) in degrees, or None on failure.
    """
    h, w = image_shape[:2]

    # Build 2D image points from selected landmarks
    img_pts = []
    for idx in FACE_LANDMARK_INDICES:
        lm = face_landmarks.landmark[idx]
        img_pts.append([lm.x * w, lm.y * h])
    img_pts = np.array(img_pts, dtype=np.float64)

    # Approximate camera intrinsics (reasonable for a typical webcam)
    focal = w  # focal ≈ image width in pixels is a common approximation
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [focal, 0,     cx],
        [0,     focal, cy],
        [0,     0,     1 ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(
        FACE_MODEL_3D, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None

    # Convert rotation vector → rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Direct Euler extraction from rotation matrix (ZYX convention).
    # This avoids decomposeProjectionMatrix which produces unreliable results.
    # Reference: standard aerospace XYZ → ZYX Euler decomposition.
    pitch_rad = np.arcsin(-rmat[2, 0])
    yaw_rad   = np.arctan2(rmat[2, 1], rmat[2, 2])
    roll_rad  = np.arctan2(rmat[1, 0], rmat[0, 0])

    pitch = np.degrees(pitch_rad)
    yaw   = np.degrees(yaw_rad)
    roll  = np.degrees(roll_rad)

    return yaw, pitch, roll


# -----------------------------------------------------------------------------
# 3. Thread-Safe State Manager
# -----------------------------------------------------------------------------
class StateManager:
    def __init__(self):
        self.lock    = threading.Lock()
        self.history = []

    def add_score(self, status):
        score_map = {"Engaged": 100, "Distracted": 50, "Drowsy": 0, "No Face Detected": -1}
        score = score_map.get(status, -1)
        if score < 0:
            return  # don't log "No Face"
        with self.lock:
            self.history.append(score)
            if len(self.history) > 120:
                self.history.pop(0)

    def get_history(self):
        with self.lock:
            return list(self.history)


if "state_manager" not in st.session_state:
    st.session_state.state_manager = StateManager()
state_manager = st.session_state.state_manager

# Session-state defaults for thresholds
_DEFAULTS = {"ear_thresh": 0.20, "yaw_thresh": 30.0, "pitch_thresh": 30.0}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -----------------------------------------------------------------------------
# 4. WebRTC Video Processor
# -----------------------------------------------------------------------------
CALIB_FRAMES  = 40   # How many frames used for baseline calibration
EAR_WINDOW    = 6    # Frames to average EAR over
POSE_WINDOW   = 10   # Frames to median-filter pose over
DROWSY_FRAMES = 25   # Consecutive frames below EAR threshold → Drowsy
DIST_FRAMES   = 15   # Consecutive frames above pose threshold → Distracted
RECOVER_FRAMES = 5   # Consecutive "good" frames needed to return to Engaged

class EngagementProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Thresholds (updated from main thread)
        self.ear_threshold   = _DEFAULTS["ear_thresh"]
        self.yaw_threshold   = _DEFAULTS["yaw_thresh"]
        self.pitch_threshold = _DEFAULTS["pitch_thresh"]

        # Calibration
        self.calibrating     = True
        self.calib_count     = 0
        self.calib_yaws      = []
        self.calib_pitches   = []
        self.baseline_yaw    = 0.0
        self.baseline_pitch  = 0.0

        # Temporal smoothing buffers
        self.ear_buf   = deque(maxlen=EAR_WINDOW)
        self.yaw_buf   = deque(maxlen=POSE_WINDOW)
        self.pitch_buf = deque(maxlen=POSE_WINDOW)

        # Hysteresis counters
        self.drowsy_count    = 0
        self.distract_count  = 0
        self.recover_count   = 0
        self.current_status  = "Calibrating"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape

        # MediaPipe requires RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True
        img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        status = "No Face Detected"
        color  = (180, 180, 180)

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]

            # ── A. Head Pose ────────────────────────────────────────────────
            pose = get_head_pose(face_lm, img.shape)
            if pose is not None:
                raw_yaw, raw_pitch, raw_roll = pose
                # Smooth pose with rolling median buffer
                self.yaw_buf.append(raw_yaw)
                self.pitch_buf.append(raw_pitch)
                smooth_yaw   = float(np.median(self.yaw_buf))
                smooth_pitch = float(np.median(self.pitch_buf))

                # ── Calibration phase ───────────────────────────────────────
                if self.calibrating:
                    self.calib_yaws.append(smooth_yaw)
                    self.calib_pitches.append(smooth_pitch)
                    self.calib_count += 1
                    if self.calib_count >= CALIB_FRAMES:
                        self.baseline_yaw   = float(np.median(self.calib_yaws))
                        self.baseline_pitch = float(np.median(self.calib_pitches))
                        self.calibrating    = False

                    # Show calibration overlay
                    progress = int(self.calib_count / CALIB_FRAMES * w)
                    cv2.rectangle(img, (0, h - 14), (progress, h), (255, 180, 0), -1)
                    cv2.putText(img, "Calibrating... look straight at screen",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

                # ── Calibration-relative angles ─────────────────────────────
                rel_yaw   = smooth_yaw   - self.baseline_yaw
                rel_pitch = smooth_pitch - self.baseline_pitch
            else:
                rel_yaw = rel_pitch = raw_roll = 0.0

            # ── B. Eye Aspect Ratio ──────────────────────────────────────────
            right_eye_idx = [33,  160, 158, 133, 153, 144]
            left_eye_idx  = [362, 385, 387, 263, 373, 380]

            re = np.array([[face_lm.landmark[i].x * w, face_lm.landmark[i].y * h] for i in right_eye_idx])
            le = np.array([[face_lm.landmark[i].x * w, face_lm.landmark[i].y * h] for i in left_eye_idx])
            avg_ear = (calculate_ear(re) + calculate_ear(le)) / 2.0
            self.ear_buf.append(avg_ear)
            smooth_ear = float(np.mean(self.ear_buf))

            # ── C. Hysteresis-based Engagement Logic ─────────────────────────
            eyes_closing  = smooth_ear < self.ear_threshold
            head_away     = abs(rel_yaw) > self.yaw_threshold or abs(rel_pitch) > self.pitch_threshold

            if eyes_closing:
                self.drowsy_count   += 1
                self.recover_count   = 0
            elif head_away:
                self.distract_count += 1
                self.recover_count   = 0
                self.drowsy_count    = max(0, self.drowsy_count - 1)
            else:
                self.recover_count  += 1
                self.drowsy_count    = max(0, self.drowsy_count - 1)
                self.distract_count  = max(0, self.distract_count - 1)

            # State transitions with hysteresis
            if self.drowsy_count >= DROWSY_FRAMES:
                self.current_status = "Drowsy"
            elif self.distract_count >= DIST_FRAMES:
                self.current_status = "Distracted"
            elif self.recover_count >= RECOVER_FRAMES:
                self.current_status = "Engaged"
                self.drowsy_count   = 0
                self.distract_count = 0

            status = self.current_status

            # ── D. Colors ────────────────────────────────────────────────────
            color_map = {
                "Engaged":    (0, 210, 0),
                "Distracted": (0, 145, 255),
                "Drowsy":     (0, 0, 230),
            }
            color = color_map.get(status, (180, 180, 180))

            # ── E. Draw Overlays ─────────────────────────────────────────────
            # Border
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)

            # Status label with background
            label = f"  {status}  "
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            cv2.rectangle(img, (10, 10), (10 + lw + 4, 10 + lh + 16), color, -1)
            cv2.putText(img, label, (14, 10 + lh + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

            # Diagnostic info
            cv2.putText(img, f"EAR: {smooth_ear:.3f}  (thresh {self.ear_threshold:.2f})",
                        (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, f"Yaw: {rel_yaw:+.1f}  Pitch: {rel_pitch:+.1f}  (thresh \u00b1{int(self.yaw_threshold)}\u00b0)",
                        (20, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(img, f"Drowsy ctr: {self.drowsy_count}/{DROWSY_FRAMES}  "
                             f"Distract ctr: {self.distract_count}/{DIST_FRAMES}",
                        (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

        else:
            cv2.putText(img, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            # Reset counters when face disappears
            self.drowsy_count   = 0
            self.distract_count = 0
            self.recover_count  = 0

        state_manager.add_score(status)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------------------------------------------------------
# 5. Custom CSS — Minimalist Industrial UI
# -----------------------------------------------------------------------------
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #090B0E;
    --surface: #0F1318;
    --surface-2: #141921;
    --sidebar: #0C0F14;
    --border: #1D2430;
    --border-hover: #2C3647;
    --text: #F1F5F9;
    --text-2: #94A3B8;
    --muted: #64748B;
    --accent: #3B82F6;
    --engaged: #10B981;
    --distracted: #F59E0B;
    --drowsy: #EF4444;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

html, body, [class*="css"], .stApp {
    font-family: var(--font-sans);
    background-color: var(--bg) !important;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
}

/* Hide boilerplate Streamlit chrome; preserve sidebar toggle */
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stDecoration"] {
    display: none !important;
}
header[data-testid="stHeader"] {
    background: transparent !important;
}

.block-container {
    max-width: 1280px;
    padding: 1.5rem 2rem 2.5rem 2rem !important;
}

/* Collapsed-by-default sidebar styling */
section[data-testid="stSidebar"] {
    background-color: var(--sidebar) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 1.1rem 0 !important;
}

/* Clean native widgets */
div[data-testid="stSlider"] label p {
    color: #CBD5E1 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
div[data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: #93C5FD !important;
    font-family: var(--font-mono) !important;
    font-size: 0.74rem !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
    background: var(--accent) !important;
}
.stButton { width: 100% !important; }
.stButton > button {
    width: 100% !important;
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-hover) !important;
    border-radius: 7px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.9rem !important;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.stButton > button:hover {
    background: #1A2230 !important;
    border-color: var(--accent) !important;
}

/* Minimal expander for hidden secondary diagnostics */
div[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    margin-top: 10px;
}
div[data-testid="stExpander"] summary {
    color: var(--text-2) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}
div[data-testid="stExpander"] summary:hover {
    color: var(--text) !important;
}

/* WebRTC frame integration */
iframe[title*="streamlit_webrtc"] {
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    background: var(--surface) !important;
}

/* Top header */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding-bottom: 16px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--border);
}
.app-brand {
    display: flex;
    align-items: center;
    gap: 11px;
}
.app-mark {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: var(--surface);
    border: 1px solid var(--border-hover);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    flex-shrink: 0;
}
.app-title {
    font-size: 1.08rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text);
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 5px 11px;
    border-radius: 6px;
    white-space: nowrap;
}
.status-pill.live {
    color: var(--engaged);
    background: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.28);
}
.status-pill.idle {
    color: var(--text-2);
    background: var(--surface);
    border: 1px solid var(--border);
}
.dot-live {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--engaged);
    animation: pulseLive 1.8s infinite;
}
.dot-idle {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--muted);
}
@keyframes pulseLive {
    0%   { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.55); }
    70%  { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Unified 3-column KPI bar */
.kpi-bar {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 18px;
}
.kpi-cell {
    padding: 15px 20px;
    border-right: 1px solid var(--border);
}
.kpi-cell:last-child { border-right: none; }
.kpi-k {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 7px;
}
.kpi-v {
    font-family: var(--font-mono);
    font-size: 1.55rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.kpi-v-state {
    font-family: var(--font-sans);
    font-size: 1.4rem;
    font-weight: 600;
    line-height: 1.1;
    letter-spacing: -0.02em;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}
.kpi-u {
    font-size: 0.85rem;
    font-weight: 400;
    color: var(--muted);
    margin-left: 2px;
}

/* Minimal section labels & chart panel */
.pane-label {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-2);
    margin-bottom: 9px;
}
.pane-label span:last-child {
    color: var(--muted);
    font-weight: 400;
}
.chart-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
}
.chart-idle {
    height: 248px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    border-radius: 6px;
    background:
        linear-gradient(to right, rgba(29, 36, 48, 0.4) 1px, transparent 1px) 0 0 / 56px 100%,
        linear-gradient(to bottom, rgba(29, 36, 48, 0.4) 1px, transparent 1px) 0 0 / 100% 62px;
}
.chart-idle-t {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-2);
}
.chart-idle-s {
    font-size: 0.76rem;
    color: var(--muted);
    margin-top: 4px;
}

/* Hidden diagnostics styling (inside collapsed expander & sidebar) */
.diag-bar {
    display: flex;
    width: 100%;
    height: 7px;
    background: var(--surface-2);
    border-radius: 4px;
    overflow: hidden;
    gap: 2px;
    margin-bottom: 12px;
}
.diag-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 12px;
}
.diag-item {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
}
.diag-k {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.diag-v {
    font-family: var(--font-mono);
    font-size: 0.98rem;
    font-weight: 600;
    color: var(--text);
    font-variant-numeric: tabular-nums;
}
.sb-label {
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.sb-note {
    font-size: 0.77rem;
    color: var(--text-2);
    line-height: 1.45;
    margin-bottom: 10px;
}
.guide-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
    font-size: 0.77rem;
    color: var(--text-2);
}
.guide-row {
    display: flex;
    align-items: center;
    gap: 8px;
}
.guide-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}

@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .kpi-bar { grid-template-columns: 1fr; }
    .kpi-cell { border-right: none; border-bottom: 1px solid var(--border); }
    .kpi-cell:last-child { border-bottom: none; }
    .diag-grid { grid-template-columns: 1fr; }
}
</style>""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5b. Minimal UI Render Helpers (Presentation Only)
# -----------------------------------------------------------------------------
_SVG_MARK = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="3"/><path d="M12 2v3"/><path d="M12 19v3"/>'
    '<path d="M2 12h3"/><path d="M19 12h3"/><circle cx="12" cy="12" r="8" stroke-dasharray="3 3"/></svg>'
)


def _render_header(is_playing: bool) -> str:
    pill = (
        '<span class="status-pill live"><span class="dot-live"></span>LIVE</span>'
        if is_playing
        else '<span class="status-pill idle"><span class="dot-idle"></span>STANDBY</span>'
    )
    return (
        '<div class="app-header">'
        '<div class="app-brand">'
        f'<div class="app-mark">{_SVG_MARK}</div>'
        '<span class="app-title">Student Engagement Monitor</span>'
        '</div>'
        f'{pill}'
        '</div>'
    )


def _render_kpi_bar(history: list, is_playing: bool) -> str:
    n = len(history)
    if n > 0:
        last = history[-1]
        if last == 100:
            s_text, s_col = "Engaged", "#10B981"
        elif last == 50:
            s_text, s_col = "Distracted", "#F59E0B"
        else:
            s_text, s_col = "Drowsy", "#EF4444"
        eng_pct = round(history.count(100) / n * 100)
        avg_val = round(sum(history) / n)
        eng_html = f'{eng_pct}<span class="kpi-u">%</span>'
        avg_html = f'{avg_val}<span class="kpi-u">/100</span>'
    else:
        s_text = "Calibrating" if is_playing else "Standby"
        s_col = "#3B82F6" if is_playing else "#64748B"
        eng_html = "&mdash;"
        avg_html = "&mdash;"

    return (
        '<div class="kpi-bar">'
        '<div class="kpi-cell">'
        '<div class="kpi-k">Status</div>'
        f'<div class="kpi-v-state" style="color:{s_col};">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{s_col};display:inline-block;"></span>'
        f'{s_text}</div>'
        '</div>'
        '<div class="kpi-cell">'
        '<div class="kpi-k">Engaged</div>'
        f'<div class="kpi-v">{eng_html}</div>'
        '</div>'
        '<div class="kpi-cell">'
        '<div class="kpi-k">Attention Score</div>'
        f'<div class="kpi-v">{avg_html}</div>'
        '</div>'
        '</div>'
    )


def _render_chart(history: list, is_playing: bool) -> str:
    meta = "LIVE" if is_playing else "IDLE"
    if not history:
        title = "Calibrating baseline..." if is_playing else "Stream offline"
        sub = (
            "Hold still and face the screen (~2 seconds)"
            if is_playing
            else "Click START on the feed to begin monitoring"
        )
        inner = f'<div class="chart-idle"><div class="chart-idle-t">{title}</div><div class="chart-idle-s">{sub}</div></div>'
    else:
        w, h = 560, 248
        pad_l, pad_r, pad_t, pad_b = 34, 14, 18, 24
        pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
        n = len(history)
        pts = [
            (
                round(pad_l + (pw if n == 1 else (i / (n - 1)) * pw), 1),
                round(pad_t + (1.0 - max(0.0, min(100.0, float(v))) / 100.0) * ph, 1),
            )
            for i, v in enumerate(history)
        ]
        line_pts = " ".join(f"{x},{y}" for x, y in pts)
        y0 = pad_t + ph
        y50 = pad_t + ph / 2
        y100 = pad_t
        area_pts = f"{pts[0][0]},{y0} {line_pts} {pts[-1][0]},{y0}"
        lx, ly = pts[-1]
        lv = history[-1]
        dot_c = "#10B981" if lv == 100 else ("#F59E0B" if lv == 50 else "#EF4444")

        inner = (
            f'<svg viewBox="0 0 {w} {h}" width="100%" height="248" preserveAspectRatio="none" style="display:block;">'
            '<defs><linearGradient id="areaG" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0%" stop-color="#3B82F6" stop-opacity="0.22"/>'
            '<stop offset="100%" stop-color="#3B82F6" stop-opacity="0.0"/>'
            '</linearGradient></defs>'
            f'<line x1="{pad_l}" y1="{y100}" x2="{w - pad_r}" y2="{y100}" stroke="rgba(16,185,129,0.22)" stroke-dasharray="3 4"/>'
            f'<line x1="{pad_l}" y1="{y50}" x2="{w - pad_r}" y2="{y50}" stroke="rgba(245,158,11,0.20)" stroke-dasharray="3 4"/>'
            f'<line x1="{pad_l}" y1="{y0}" x2="{w - pad_r}" y2="{y0}" stroke="#1D2430"/>'
            f'<text x="{pad_l - 8}" y="{y100 + 3}" text-anchor="end" fill="#64748B" font-family="JetBrains Mono,monospace" font-size="10">100</text>'
            f'<text x="{pad_l - 8}" y="{y50 + 3}" text-anchor="end" fill="#64748B" font-family="JetBrains Mono,monospace" font-size="10">50</text>'
            f'<text x="{pad_l - 8}" y="{y0 + 3}" text-anchor="end" fill="#64748B" font-family="JetBrains Mono,monospace" font-size="10">0</text>'
            f'<polygon points="{area_pts}" fill="url(#areaG)"/>'
            f'<polyline points="{line_pts}" fill="none" stroke="#60A5FA" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            f'<circle cx="{lx}" cy="{ly}" r="5.5" fill="{dot_c}" opacity="0.24"/>'
            f'<circle cx="{lx}" cy="{ly}" r="3" fill="{dot_c}" stroke="#0F1318" stroke-width="1.5"/>'
            f'<text x="{pad_l}" y="{h - 5}" fill="#64748B" font-family="JetBrains Mono,monospace" font-size="9.5">T-{n}</text>'
            f'<text x="{w - pad_r}" y="{h - 5}" text-anchor="end" fill="#94A3B8" font-family="JetBrains Mono,monospace" font-size="9.5">NOW</text>'
            '</svg>'
        )

    return (
        f'<div class="pane-label"><span>Engagement Timeline</span><span>{meta}</span></div>'
        f'<div class="chart-box">{inner}</div>'
    )


def _render_diagnostics(history: list) -> str:
    n = len(history)
    if n > 0:
        e_n, d_n, dr_n = history.count(100), history.count(50), history.count(0)
        e_p, d_p = round(e_n / n * 100), round(d_n / n * 100)
        dr_p = max(0, 100 - e_p - d_p)
        bar = (
            f'<div style="width:{e_p}%;background:#10B981;"></div>'
            f'<div style="width:{d_p}%;background:#F59E0B;"></div>'
            f'<div style="width:{dr_p}%;background:#EF4444;"></div>'
        )
        e_s, d_s, dr_s = f"{e_p}% ({e_n})", f"{d_p}% ({d_n})", f"{dr_p}% ({dr_n})"
    else:
        bar = '<div style="width:100%;background:#1D2430;"></div>'
        e_s = d_s = dr_s = "&mdash;"

    ear_v = st.session_state.get("ear_thresh", 0.20)
    yaw_v = st.session_state.get("yaw_thresh", 30.0)
    pitch_v = st.session_state.get("pitch_thresh", 30.0)
    return (
        f'<div class="diag-bar">{bar}</div>'
        '<div class="diag-grid">'
        f'<div class="diag-item"><div class="diag-k"><span class="guide-dot" style="background:#10B981;"></span>Engaged</div><div class="diag-v">{e_s}</div></div>'
        f'<div class="diag-item"><div class="diag-k"><span class="guide-dot" style="background:#F59E0B;"></span>Distracted</div><div class="diag-v">{d_s}</div></div>'
        f'<div class="diag-item"><div class="diag-k"><span class="guide-dot" style="background:#EF4444;"></span>Drowsy</div><div class="diag-v">{dr_s}</div></div>'
        '</div>'
        '<div class="diag-grid">'
        f'<div class="diag-item"><div class="diag-k">EAR Cutoff</div><div class="diag-v">&lt; {ear_v:.2f} ({DROWSY_FRAMES}f)</div></div>'
        f'<div class="diag-item"><div class="diag-k">Yaw Limit</div><div class="diag-v">&plusmn;{int(yaw_v)}&deg; ({DIST_FRAMES}f)</div></div>'
        f'<div class="diag-item"><div class="diag-k">Pitch Limit</div><div class="diag-v">&plusmn;{int(pitch_v)}&deg; ({RECOVER_FRAMES}f rec)</div></div>'
        '</div>'
    )


# -----------------------------------------------------------------------------
# 6. Sidebar — Hidden by Default (Settings, Calibration & Reference)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sb-label">Detection Settings</div>'
        '<div class="sb-note">Thresholds sync live with the active video stream.</div>',
        unsafe_allow_html=True,
    )

    st.session_state.ear_thresh = st.slider(
        "EAR Threshold (Drowsiness)",
        min_value=0.10, max_value=0.40,
        value=st.session_state.get("ear_thresh", 0.20), step=0.01,
        help="Eye Aspect Ratio below this for 25+ frames = Drowsy."
    )
    st.session_state.yaw_thresh = st.slider(
        "Yaw Limit (deg)",
        min_value=10.0, max_value=60.0,
        value=st.session_state.get("yaw_thresh", 30.0), step=1.0,
        help="Calibrated horizontal turn beyond this = Distracted."
    )
    st.session_state.pitch_thresh = st.slider(
        "Pitch Limit (deg)",
        min_value=10.0, max_value=60.0,
        value=st.session_state.get("pitch_thresh", 30.0), step=1.0,
        help="Calibrated vertical tilt beyond this = Distracted."
    )

    st.markdown("---")
    st.markdown(
        '<div class="sb-label">Calibration</div>'
        '<div class="sb-note">Reset to capture a new neutral head angle over 40 frames.</div>',
        unsafe_allow_html=True,
    )
    if st.button("Recalibrate Baseline"):
        st.session_state["reset_calib"] = True

    st.markdown("---")
    with st.expander("Status Guide", expanded=False):
        st.markdown(
            '<div class="guide-list">'
            '<div class="guide-row"><span class="guide-dot" style="background:#10B981;"></span><span><strong>Engaged</strong> &mdash; Eyes open, facing screen</span></div>'
            '<div class="guide-row"><span class="guide-dot" style="background:#F59E0B;"></span><span><strong>Distracted</strong> &mdash; Head turned &gt;15 frames</span></div>'
            '<div class="guide-row"><span class="guide-dot" style="background:#EF4444;"></span><span><strong>Drowsy</strong> &mdash; Eyes closed &gt;25 frames</span></div>'
            '<div class="guide-row"><span class="guide-dot" style="background:#3B82F6;"></span><span><strong>Calibrating</strong> &mdash; Hold still at start (~2s)</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# 7. Main Workspace — Minimal Focused Layout
# -----------------------------------------------------------------------------
header_placeholder = st.empty()
kpi_placeholder = st.empty()

col_feed, col_chart = st.columns([1.4, 1.0], gap="large")

with col_feed:
    st.markdown('<div class="pane-label"><span>Live Feed</span><span>WEBRTC</span></div>', unsafe_allow_html=True)

    ctx = webrtc_streamer(
        key="engagement-monitor",
        video_processor_factory=EngagementProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        },
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # ── Sync thresholds & handle calibration reset from main thread ──
    if ctx.state.playing and ctx.video_processor:
        vp = ctx.video_processor
        vp.ear_threshold   = st.session_state.ear_thresh
        vp.yaw_threshold   = st.session_state.yaw_thresh
        vp.pitch_threshold = st.session_state.pitch_thresh

        if st.session_state.get("reset_calib", False):
            vp.calibrating    = True
            vp.calib_count    = 0
            vp.calib_yaws     = []
            vp.calib_pitches  = []
            vp.current_status = "Calibrating"
            vp.drowsy_count   = 0
            vp.distract_count = 0
            vp.recover_count  = 0
            st.session_state["reset_calib"] = False

with col_chart:
    chart_placeholder = st.empty()

# Hidden secondary diagnostics (collapsed by default)
with st.expander("Session Breakdown & Parameters", expanded=False):
    diag_placeholder = st.empty()
    st.caption("Hold still for ~2 seconds after starting the camera to auto-calibrate your resting head position. Open the sidebar (top-left) to adjust detection thresholds or recalibrate.")


# -----------------------------------------------------------------------------
# 8. Live Telemetry Update Loop
# -----------------------------------------------------------------------------
is_playing = bool(ctx.state.playing)
header_placeholder.markdown(_render_header(is_playing), unsafe_allow_html=True)

initial_history = state_manager.get_history()
kpi_placeholder.markdown(_render_kpi_bar(initial_history, is_playing), unsafe_allow_html=True)
chart_placeholder.markdown(_render_chart(initial_history, is_playing), unsafe_allow_html=True)
diag_placeholder.markdown(_render_diagnostics(initial_history), unsafe_allow_html=True)

if ctx.state.playing:
    while True:
        time.sleep(0.5)
        history = state_manager.get_history()
        kpi_placeholder.markdown(_render_kpi_bar(history, True), unsafe_allow_html=True)
        chart_placeholder.markdown(_render_chart(history, True), unsafe_allow_html=True)
        diag_placeholder.markdown(_render_diagnostics(history), unsafe_allow_html=True)
        if not ctx.state.playing:
            break
