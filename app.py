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
# 5. Streamlit UI
# -----------------------------------------------------------------------------
st.title("👨‍🎓 Student Engagement Monitoring System")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
    <strong>Welcome!</strong> This application monitors a student's webcam feed in real-time.
    It uses <em>Eye Aspect Ratio (EAR)</em> and <em>calibrated Head Pose estimation</em> to determine 
    if the student is <strong>Engaged</strong>, <strong>Distracted</strong>, or <strong>Drowsy</strong>.
    <br/><br/>
    When you first start, hold still and look at the screen for ~2 seconds while the system calibrates to your natural head position.
</div>
<br/>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Settings")
    st.write("Adjust detection sensitivity:")

    st.session_state.ear_thresh = st.slider(
        "EAR Threshold (Drowsiness)",
        min_value=0.10, max_value=0.40, value=0.20, step=0.01,
        help="Eye Aspect Ratio below this value for 25 frames → Drowsy."
    )
    st.session_state.yaw_thresh = st.slider(
        "Yaw Threshold (Left/Right)",
        min_value=10.0, max_value=60.0, value=30.0, step=1.0,
        help="Calibrated sideways head turn beyond this angle → Distracted."
    )
    st.session_state.pitch_thresh = st.slider(
        "Pitch Threshold (Up/Down)",
        min_value=10.0, max_value=60.0, value=30.0, step=1.0,
        help="Calibrated up/down head tilt beyond this angle → Distracted."
    )

    st.markdown("---")
    st.markdown("""
    ### 📊 Status Guide
    - 🟢 **Engaged** (score 100): Eyes open, facing screen.
    - 🟠 **Distracted** (score 50): Head turned away for >15 frames.
    - 🔴 **Drowsy** (score 0): Eyes closed for >25 frames.
    - ⏳ **Calibrating**: Hold still & look at screen.
    """)

    st.markdown("---")
    if st.button("🔄 Reset Calibration"):
        # Signal the processor to re-calibrate next time
        st.session_state["reset_calib"] = True

with col1:
    st.subheader("📷 Live Monitor")

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

# Push updated thresholds and handle calibration reset from main thread
if ctx.state.playing and ctx.video_processor:
    vp = ctx.video_processor
    vp.ear_threshold   = st.session_state.ear_thresh
    vp.yaw_threshold   = st.session_state.yaw_thresh
    vp.pitch_threshold = st.session_state.pitch_thresh

    if st.session_state.get("reset_calib", False):
        vp.calibrating      = True
        vp.calib_count      = 0
        vp.calib_yaws       = []
        vp.calib_pitches    = []
        vp.current_status   = "Calibrating"
        vp.drowsy_count     = 0
        vp.distract_count   = 0
        vp.recover_count    = 0
        st.session_state["reset_calib"] = False

# -----------------------------------------------------------------------------
# 6. Live Engagement Chart
# -----------------------------------------------------------------------------
st.subheader("📈 Live Engagement Score")
chart_placeholder = st.empty()

if ctx.state.playing:
    while True:
        history = state_manager.get_history()
        time.sleep(0.5)
        if history:
            import pandas as pd
            df = pd.DataFrame({"Engagement Score": history})
            chart_placeholder.line_chart(df)
        else:
            chart_placeholder.write("Collecting data...")
        if not ctx.state.playing:
            break
else:
    chart_placeholder.info("▶️ Start the webcam feed above to begin monitoring.")
