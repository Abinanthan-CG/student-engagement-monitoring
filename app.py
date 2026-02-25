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
st.set_page_config(page_title="Student Engagement Monitoring", layout="wide")

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
# 5. Custom CSS — Premium Dark UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Main background ────────────────────────────────────────── */
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #30363d; }

/* ── Metric cards ───────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2430 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px 18px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: #58a6ff; }
.metric-label { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
                color: #8b949e; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #e6edf3; line-height: 1; }
.metric-sub   { font-size: 0.72rem; color: #6e7681; margin-top: 4px; }

/* ── Status badge ───────────────────────────────────────────── */
.status-engaged    { color: #3fb950; }
.status-distracted { color: #f0883e; }
.status-drowsy     { color: #f85149; }
.status-default    { color: #8b949e; }

/* ── Section header ─────────────────────────────────────────── */
.section-header {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    color: #58a6ff; text-transform: uppercase; margin-bottom: 12px;
    border-bottom: 1px solid #21262d; padding-bottom: 6px;
}

/* ── Hero banner ────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d1117 0%, #1a2332 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(31,111,235,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(63,185,80,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title { font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin-bottom: 8px; }
.hero-sub   { font-size: 0.95rem; color: #8b949e; line-height: 1.6; }

/* ── Chart container ────────────────────────────────────────── */
.chart-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px;
    margin-top: 16px;
}

/* ── Streamlit overrides ────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #58a6ff);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 8px 20px; width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
div[data-testid="stSlider"] > label { color: #8b949e !important; font-size: 0.85rem !important; }
.stSlider [data-baseweb="slider"] .rc-slider-track { background: #1f6feb !important; }
h1, h2, h3 { color: #e6edf3 !important; }
.stInfo { background: #161b22 !important; border: 1px solid #1f6feb !important; color: #8b949e !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. Sidebar — Settings (hidden by default, user opens it)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="section-header">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    st.caption("Fine-tune how sensitive the AI is to your movements.")

    st.session_state.ear_thresh = st.slider(
        "👁️ EAR Threshold (Drowsiness)",
        min_value=0.10, max_value=0.40,
        value=st.session_state.get("ear_thresh", 0.20), step=0.01,
        help="Eye Aspect Ratio below this for 25+ frames = Drowsy."
    )
    st.session_state.yaw_thresh = st.slider(
        "↔️ Yaw Threshold (Left/Right)",
        min_value=10.0, max_value=60.0,
        value=st.session_state.get("yaw_thresh", 30.0), step=1.0,
        help="Calibrated sideways angle beyond this = Distracted."
    )
    st.session_state.pitch_thresh = st.slider(
        "↕️ Pitch Threshold (Up/Down)",
        min_value=10.0, max_value=60.0,
        value=st.session_state.get("pitch_thresh", 30.0), step=1.0,
        help="Calibrated up/down angle beyond this = Distracted."
    )

    st.markdown("---")
    st.markdown('<div class="section-header">🔧 Calibration</div>', unsafe_allow_html=True)
    st.caption("Moved positions? Recalibrate so the AI resets to your new natural head angle.")
    if st.button("🔄 Reset Calibration"):
        st.session_state["reset_calib"] = True

    st.markdown("---")
    st.markdown('<div class="section-header">📖 Status Guide</div>', unsafe_allow_html=True)
    st.markdown("""
- 🟢 **Engaged** — Eyes open, facing screen
- 🟠 **Distracted** — Head turned >15 frames
- 🔴 **Drowsy** — Eyes closed >25 frames
- ⏳ **Calibrating** — Hold still at start
    """)

    st.markdown("---")
    st.caption("Built with MediaPipe · OpenCV · Streamlit-WebRTC")


# -----------------------------------------------------------------------------
# 7. Main Page — Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">👨‍🎓 Student Engagement Monitor</div>
    <div class="hero-sub">
        Real-time AI attention analysis using <strong>Eye Aspect Ratio</strong> (EAR) and 
        <strong>calibrated 3D Head Pose estimation</strong>.<br/>
        When you start the camera, hold still for ~2 seconds — the system will calibrate to <em>your</em> natural head position.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Live status metric cards above the webcam ──
m1, m2, m3 = st.columns(3)
_status_hist = state_manager.get_history()
_last_status = "—"
if _status_hist:
    _s = _status_hist[-1]
    _last_status = "Engaged" if _s == 100 else ("Distracted" if _s == 50 else "Drowsy")
_engaged_pct = (
    round(_status_hist.count(100) / len(_status_hist) * 100) if _status_hist else "—"
)
_avg_score = round(sum(_status_hist) / len(_status_hist)) if _status_hist else "—"

with m1:
    cls = {"Engaged": "status-engaged", "Distracted": "status-distracted",
           "Drowsy": "status-drowsy"}.get(_last_status, "status-default")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Current Status</div>
        <div class="metric-value {cls}">{_last_status}</div>
        <div class="metric-sub">Updated every frame</div>
    </div>""", unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Time Engaged</div>
        <div class="metric-value" style="color:#3fb950;">{_engaged_pct}{'%' if isinstance(_engaged_pct, int) else ''}</div>
        <div class="metric-sub">of monitored session</div>
    </div>""", unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Score</div>
        <div class="metric-value" style="color:#58a6ff;">{_avg_score}{'/100' if isinstance(_avg_score, int) else ''}</div>
        <div class="metric-sub">0=Drowsy · 50=Distracted · 100=Engaged</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Webcam Stream ──
st.markdown('<div class="section-header">📷 Live Monitor</div>', unsafe_allow_html=True)

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

# -----------------------------------------------------------------------------
# 8. Live Engagement Chart
# -----------------------------------------------------------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📈 Live Engagement Score</div>', unsafe_allow_html=True)
chart_placeholder = st.empty()

if ctx.state.playing:
    import pandas as pd
    while True:
        history = state_manager.get_history()
        time.sleep(0.5)
        if history:
            df = pd.DataFrame({"Engagement Score (0=Drowsy · 50=Distracted · 100=Engaged)": history})
            chart_placeholder.line_chart(df, use_container_width=True, height=200)
        else:
            chart_placeholder.markdown(
                '<div class="metric-card" style="color:#8b949e;">⏳ Collecting data...</div>',
                unsafe_allow_html=True)
        if not ctx.state.playing:
            break
else:
    chart_placeholder.info("▶️ Click **START** above to begin live monitoring. Settings are in the ← sidebar.")
