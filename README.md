<p align="center">
  <img src="assets/header.png" alt="Student Engagement Monitoring System" width="100%"/>
</p>

<h1 align="center">🎓 Student Engagement Monitoring System</h1>

<p align="center">
  <em>Real-time AI-powered attention analysis directly in your browser — no installations, no compromise.</em>
</p>

<p align="center">
  <a href="https://github.com/Abinanthan-CG/student-engagement-monitoring/stargazers">
    <img src="https://img.shields.io/github/stars/Abinanthan-CG/student-engagement-monitoring?style=for-the-badge&color=yellow" alt="Stars"/>
  </a>
  <a href="https://github.com/Abinanthan-CG/student-engagement-monitoring/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/MediaPipe-0.10-orange?style=for-the-badge" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" alt="OpenCV"/>
</p>

<br/>

> 💡 **Built as a mini-project in under 3 hours.** This system monitors a student's live webcam feed and uses computer vision to determine — frame by frame — whether they're engaged, distracted, or drowsy. No cloud ML APIs. No subscriptions. Just math and code.

---

## 📸 Live Preview

<p align="center">
  <img src="assets/mockup.png" alt="App UI Preview" width="90%"/>
</p>

> The app auto-calibrates to your natural head position in the first 2 seconds, then begins monitoring. A diagnostic HUD at the bottom shows you exactly what the AI is measuring in real-time.

---

## 🚀 Try It Live

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/abinanthan-cg/student-engagement-monitoring/main/app.py)

> Click above to open the live app directly in your browser. No setup needed.

---

## 🧠 The Science

This project sits at the intersection of geometry, linear algebra, and computer vision. Here's exactly how it works under the hood.

---

### 👁️ Eye Aspect Ratio (EAR) — Drowsiness Detection

The EAR is a simple but surprisingly effective formula introduced in the original drowsiness detection research by Soukupová & Čech (2016).

**The idea**: When your eyes are open, the ratio of vertical-to-horizontal eye distance is relatively constant. When you close your eyes, it drops to near zero.

```
          p2 ●      ● p3
         /                \
p1 ●                        ● p4
         \                /
          p6 ●      ● p5

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

| Eyes State           | Typical EAR | Status         |
| -------------------- | ----------- | -------------- |
| Wide open            | ~0.30–0.45  | Normal         |
| Squinting            | ~0.20–0.30  | Normal (tired) |
| ≤ 0.20 for 25 frames | < 0.20      | **😴 Drowsy**  |

We calculate EAR independently for both eyes using MediaPipe's 478 facial landmarks, then average the two values through a **6-frame rolling mean** to smooth out natural blinks.

---

### 🗺️ Head Pose Estimation — Distraction Detection

We use OpenCV's `solvePnP` — a classical computer vision algorithm — to figure out where your head is pointing in 3D space from a 2D image.

**How solvePnP works:**

1. We pick **10 stable facial landmarks** from MediaPipe (nose tip, chin, eye corners, mouth corners, brow points).
2. We know the corresponding 3D coordinates of these points in a generic face model (in millimetres).
3. `solvePnP` solves the geometric equation to find the 3D rotation that maps the model onto your face in the image.
4. We convert the resulting rotation vector to a **rotation matrix** using `cv2.Rodrigues`.
5. From this matrix, we extract **Yaw, Pitch, and Roll** using direct `atan2` decomposition — **not** `decomposeProjectionMatrix`, which is notoriously unreliable for this use case.

```
     Yaw (left/right)        Pitch (up/down)        Roll (tilt)
      ←  ●  →                  ↑  ●  ↓                  ↺  ●
       (Y-axis)               (X-axis)                (Z-axis)
```

**The critical calibration step**: Because people naturally sit at different angles, we measure and record your resting head pose over the first 40 frames. All future measurements are _relative to this personal baseline_ — so a naturally tilted head won't constantly trigger "Distracted".

---

### ⚖️ The Decision Engine

A raw per-frame signal would be extremely noisy — you blink constantly, and nobody holds their head perfectly still. So we add **temporal hysteresis**:

```
                   ┌─────────────────────────────────────────────┐
                   │       Hysteresis State Machine               │
                   │                                              │
     Eyes closed   │  drowsy_count ≥ 25  ──────────► 😴 Drowsy  │
     (EAR < thresh)│                                              │
                   │                                              │
     Head away     │  distract_count ≥ 15 ─────────► 😵 Distracted│
     (|yaw|>thresh)│                                              │
                   │                                              │
     Looks OK      │  recover_count ≥ 5  ──────────► ✅ Engaged  │
                   └─────────────────────────────────────────────┘
```

This means a single bad frame, a blink, or a momentary glance never changes your status. The system requires sustained evidence before drawing any conclusions.

---

## ✨ Feature Highlights

| Feature                      | Details                                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| 🎯 **Auto-Calibration**      | 40-frame startup calibration adapts to your personal resting head angle — no manual setup |
| 📊 **Temporal Smoothing**    | EAR averaged over 6 frames; head pose median-filtered over 10 frames                      |
| ⚖️ **Hysteresis Logic**      | State only changes after sustained evidence (15–25 frames), not individual noisy ones     |
| 🔢 **Diagnostic HUD**        | Live EAR value, calibrated yaw/pitch angles, and frame counters shown on video            |
| 🔄 **Reset Button**          | One-click calibration reset if you move seats or change posture                           |
| 🌐 **WebRTC Streaming**      | No file uploads, no RTSP — direct browser-to-server video with ICE/STUN                   |
| 🎛️ **Adjustable Thresholds** | EAR, Yaw, and Pitch thresholds all adjustable via sidebar sliders                         |
| 📈 **Live Scoring Chart**    | Real-time engagement score plotted over the last 120 frames                               |

---

## 🛠️ Local Setup

> **Prerequisites**: Python 3.10, pip, a working webcam.

### 1. Clone the repository

```bash
git clone https://github.com/Abinanthan-CG/student-engagement-monitoring.git
cd student-engagement-monitoring
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📦 Dependencies

```txt
streamlit
streamlit-webrtc
opencv-python-headless
mediapipe==0.10.14
numpy<2.0.0
av
```

> **Why `mediapipe==0.10.14`?** Newer versions restructured the API and removed `mp.solutions`. We pin to `0.10.14` for stability with the `solutions.face_mesh` interface.

> **Why `numpy<2.0.0`?** MediaPipe 0.10.x has a hard dependency on NumPy 1.x.

---

## ☁️ Streamlit Cloud Deployment

The repo is fully configured for one-click deployment on **Streamlit Community Cloud**.

| File               | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `requirements.txt` | Python packages                                            |
| `packages.txt`     | System-level `libgl1` and `libglib2.0` for OpenCV on Linux |

> **Important**: During deployment setup, go to **Advanced Settings** and select **Python 3.10**. Python 3.12 is incompatible with MediaPipe 0.10.x.

---

## 📁 Project Structure

```
student-engagement-monitoring/
│
├── app.py                  # Main application (single-file)
├── requirements.txt        # Python dependencies
├── packages.txt            # Linux system dependencies (for Streamlit Cloud)
├── assets/
│   ├── header.png          # Project banner image
│   └── mockup.png          # UI preview image
└── README.md               # You're reading it!
```

---

## 🔬 Known Limitations & Future Work

- **Single face only**: The system monitors one face at a time. Multi-student support would need extra logic.
- **Fixed 3D face model**: We use a generic average face for `solvePnP`. An identity-specific model would improve pose accuracy.
- **No audio analysis**: A drowsy student might be talking. Multi-modal fusion (audio + video) would reduce false positives.
- **Lighting sensitivity**: Very dim or high-contrast lighting affects MediaPipe's tracking confidence.

**Future roadmap ideas:**

- [ ] Gaze estimation using iris landmarks (already available via `refine_landmarks=True`)
- [ ] Session reports with timestamped engagement breakdowns
- [ ] Email/webhook alerts for prolonged disengagement
- [ ] Teacher dashboard for multi-student monitoring

---

## 🧑‍💻 About the Author

Built by **Abinanthan** — a computer science student with a passion for real-world applications of computer vision and machine learning.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and share.

---

<p align="center">
  Made with ❤️, a lot of coffee ☕, and just a little bit of linear algebra. 
  <br/>
  <strong>If you found this useful, drop a ⭐ on the repo!</strong>
</p>
