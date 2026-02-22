import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import mediapipe as mp
import threading
import time

# Let's set the page config early so everything uses wide layout
st.set_page_config(page_title="Student Engagement Monitoring", layout="wide")

# -----------------------------------------------------------------------------
# 1. Imports & Setup
# -----------------------------------------------------------------------------
# MediaPipe setup for Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# -----------------------------------------------------------------------------
# 2. Mathematical Utility Functions
# -----------------------------------------------------------------------------
def calculate_ear(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR).
    The EAR formula is used to detect eye closure associated with drowsiness and blinking.
    It computes the ratio of distances between the vertical eye landmarks and horizontal eye landmarks.
    
    Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2.0 * ||p1 - p4||)
    Where:
      p1, p4: horizontal landmarks (eye corners)
      p2, p3, p5, p6: vertical landmarks (upper and lower eyelids)
      
    Args:
        eye_landmarks (list or np.ndarray): 6 standard eye landmarks.
        
    Returns:
        float: the calculated eye aspect ratio.
    """
    # Vertical distances
    v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # Horizontal distance
    h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # Calculate EAR
    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_head_pose(face_landmarks, image_shape):
    """
    Calculate the Head Pose (Yaw, Pitch, Roll) using OpenCV's solvePnP.
    
    solvePnP (Perspective-n-Point) is a mathematical algorithm that estimates the 3D pose 
    of an object from its 2D image points, provided we have a 3D model of the object and 
    the camera's intrinsic parameters.
    
    Here, we use a generic 3D face model and map 6 critical facial landmarks to it:
    1. Nose Tip
    2. Chin
    3. Left Eye Left Corner
    4. Right Eye Right Corner
    5. Left Mouth Corner
    6. Right Mouth Corner
    
    Args:
        face_landmarks: The 2D facial landmarks extracted from MediaPipe Face Mesh.
        image_shape: The (height, width) of the video frame, used to approximate the camera matrix.
        
    Returns:
        tuple: (yaw, pitch, roll) angles in degrees.
    """
    h, w = image_shape[:2]
    
    # 2D image points from MediaPipe landmarks
    # Indices correspond to specific facial features standard in MediaPipe:
    # 1: Nose Tip, 152: Chin, 33: Left Eye Corner, 263: Right Eye Corner, 
    # 61: Left Mouth Corner, 291: Right Mouth Corner
    image_points = np.array([
        (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h)),     # Nose tip
        (int(face_landmarks.landmark[152].x * w), int(face_landmarks.landmark[152].y * h)), # Chin
        (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h)),   # Left eye left corner
        (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h)), # Right eye right corner
        (int(face_landmarks.landmark[61].x * w), int(face_landmarks.landmark[61].y * h)),   # Left Mouth corner
        (int(face_landmarks.landmark[291].x * w), int(face_landmarks.landmark[291].y * h))  # Right mouth corner
    ], dtype="double")
    
    # 3D model points of a generic face
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # Approximate camera parameters
    # We estimate the focal length roughly as the width of the image.
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1)) # Assume no lens distortion
    
    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Convert rotation vector to a rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Compute the Euler angles from the rotation matrix
    # The matrix represents the 3D rotation of the head relative to the camera.
    # decomposeProjectionMatrix extracts the Euler angles directly.
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    
    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    
    return yaw, pitch, roll


# -----------------------------------------------------------------------------
# 3. State Management
# -----------------------------------------------------------------------------
# We use a global thread-safe list to store engagement history.
# In a stream, the processor runs on a background thread.
# Streamlit runs on the main thread, so we need a lock for concurrent access.
class StateManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = []
        
    def add_score(self, status):
        score = 0
        if status == "Engaged":
            score = 100
        elif status == "Distracted":
            score = 50
        elif status == "Drowsy":
            score = 0
            
        with self.lock:
            self.history.append(score)
            # Keep only the last 100 values to avoid memory overflow
            if len(self.history) > 100:
                self.history.pop(0)
                
    def get_history(self):
        with self.lock:
            return list(self.history)

if "state_manager" not in st.session_state:
    st.session_state.state_manager = StateManager()

state_manager = st.session_state.state_manager

# Context block config for thresholds
if "ear_thresh" not in st.session_state:
    st.session_state.ear_thresh = 0.20
if "yaw_thresh" not in st.session_state:
    st.session_state.yaw_thresh = 20.0
if "pitch_thresh" not in st.session_state:
    st.session_state.pitch_thresh = 20.0

# -----------------------------------------------------------------------------
# 4. The WebRTC Video Processor Class
# -----------------------------------------------------------------------------
class EngagementProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize Face Mesh model here so it persists across frames
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Critical constraint: enables iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drowsy_frames = 0
        self.ear_threshold = 0.20
        self.yaw_threshold = 20.0
        self.pitch_threshold = 20.0
        self.consecutive_frames_threshold = 20
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        
        # Convert frame to numpy array (BGR mode for OpenCV)
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        
        # Convert the BGR image to RGB before processing with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        
        # Draw on the image, we must make it writeable again
        image_rgb.flags.writeable = True
        # Convert back to BGR for OpenCV drawing and final output
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        status = "No Face Detected"
        color = (200, 200, 200) # Gray
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # --- A. Calculate Head Pose ---
                try:
                    yaw, pitch, roll = get_head_pose(face_landmarks, image_bgr.shape)
                except Exception as e:
                    yaw = pitch = roll = 0.0
                
                # --- B. Calculate EAR ---
                # Landmarks for right eye (MediaPipe indices)
                right_eye_indices = [33, 160, 158, 133, 153, 144]
                # Landmarks for left eye (MediaPipe indices)
                left_eye_indices = [362, 385, 387, 263, 373, 380]
                
                # Extract coordinates
                re_coords = np.array([
                    [face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] 
                    for idx in right_eye_indices
                ])
                le_coords = np.array([
                    [face_landmarks.landmark[idx].x * w, face_landmarks.landmark[idx].y * h] 
                    for idx in left_eye_indices
                ])
                
                right_ear = calculate_ear(re_coords)
                left_ear = calculate_ear(le_coords)
                avg_ear = (right_ear + left_ear) / 2.0
                
                # --- C. Engagement Logic ---
                if avg_ear < self.ear_threshold:
                    self.drowsy_frames += 1
                else:
                    self.drowsy_frames = 0
                
                if self.drowsy_frames > self.consecutive_frames_threshold:
                    status = "Drowsy"
                    color = (0, 0, 255) # Red (BGR)
                elif abs(yaw) > self.yaw_threshold or abs(pitch) > self.pitch_threshold:
                    status = "Distracted"
                    color = (0, 165, 255) # Orange (BGR)
                else:
                    status = "Engaged"
                    color = (0, 255, 0) # Green (BGR)
                
                # --- D. Visual Output Drawing ---
                # Define some y-offsets to stack text nicely
                cv2.putText(
                    image_bgr, 
                    f"Status: {status}", 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, 
                    color, 
                    2, 
                    lineType=cv2.LINE_AA
                )
                
                cv2.putText(
                    image_bgr, 
                    f"EAR: {avg_ear:.2f}", 
                    (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (255, 255, 255), 
                    2
                )
                
                cv2.putText(
                    image_bgr, 
                    f"Head Pose - Yaw: {int(yaw)} Pitch: {int(pitch)} Roll: {int(roll)}", 
                    (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2
                )
                
                # Draw a bounding box around the frame indicating state
                cv2.rectangle(image_bgr, (0, 0), (w, h), color, 6)
                
                # We only process the first detected face for simplicity
                break
        else:
            # Output "No Face Detected" if no face was found
            cv2.putText(
                image_bgr, 
                "No Face Detected", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0, 
                color, 
                2, 
                lineType=cv2.LINE_AA
            )
            
        # Push score to state manager for live plotting
        state_manager.add_score(status)
        
        # Return the processed frame
        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


# -----------------------------------------------------------------------------
# 5. The Streamlit UI Layout
# -----------------------------------------------------------------------------
st.title("👨‍🎓 Student Engagement Monitoring System")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px;'>
    <strong>Welcome!</strong> This application monitors a student's webcam feed in real-time. 
    It determines whether the student is <em>Engaged</em>, <em>Distracted</em>, or <em>Drowsy</em> 
    based on their Eye Aspect Ratio (EAR) and Head Pose estimation.
</div>
<br/>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ Settings")
    st.write("Adjust the sensitivity thresholds for the AI model:")
    
    st.session_state.ear_thresh = st.slider(
        "EAR Threshold (Drowsiness)", 
        min_value=0.10, 
        max_value=0.40, 
        value=0.20, 
        step=0.01,
        help="If the Eye Aspect Ratio drops below this value for 20 frames, the user is marked as 'Drowsy'."
    )
    
    st.session_state.yaw_thresh = st.slider(
        "Yaw Angle Threshold (Sideways)", 
        min_value=10.0, 
        max_value=50.0, 
        value=20.0, 
        step=1.0,
        help="If the head turns beyond this angle (left or right), the user is marked as 'Distracted'."
    )
    
    st.session_state.pitch_thresh = st.slider(
        "Pitch Angle Threshold (Up/Down)", 
        min_value=10.0, 
        max_value=50.0, 
        value=20.0, 
        step=1.0,
        help="If the head tilts beyond this angle (up or down), the user is marked as 'Distracted'."
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📊 How it works
    - **Engaged (Score=100)**: Looking at the screen, eyes open.
    - **Distracted (Score=50)**: Head tilted beyond threshold.
    - **Drowsy (Score=0)**: Eyes closed (low EAR) for extended periods.
    """)

with col1:
    st.subheader("📷 Live Monitor")
    
    # We create a placeholder for the live graph
    graph_col, metric_col = st.columns([4, 1])
    
    # Initialize the streamer.
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
    
# Display metrics chart below the webcam
st.subheader("📈 Live Engagement Score")
chart_placeholder = st.empty()

# Update processor thresholds if stream is playing
if ctx.state.playing and ctx.video_processor:
    ctx.video_processor.ear_threshold = st.session_state.ear_thresh
    ctx.video_processor.yaw_threshold = st.session_state.yaw_thresh
    ctx.video_processor.pitch_threshold = st.session_state.pitch_thresh

# We can run a small loop inside streamlit to update the chart 
# if the stream is actively playing.
if ctx.state.playing:
    while True:
        history = state_manager.get_history()
        # Sleep slightly to prevent maxing out the CPU on the main thread
        time.sleep(0.5) 
        if history:
            chart_placeholder.line_chart(history)
        else:
            chart_placeholder.write("Collecting data...")
        # Check if streamer stopped internally, though streamlit typically forces a rerun 
        # on state change, breaking out of loop is safe practice.
        if not ctx.state.playing:
            break
else:
    chart_placeholder.info("Start the webcam feed to view live engagement scores.")
