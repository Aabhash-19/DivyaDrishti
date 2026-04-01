# 🔍 DivyaDrishti
### AI-powered Real-Time Object Detection for the Visually Impaired

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Auto-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**DivyaDrishti** (Sanskrit: *दिव्यदृष्टि* — "Divine Vision") is a real-time, AI-powered object detection and voice-feedback system built to assist visually impaired individuals. It streams live video from a mobile phone camera, performs deep-learning-based object detection on a laptop/server, and announces detected objects using text-to-speech — all through a modern, accessible web interface.

</div>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [AI Models](#-ai-models)
- [Project Structure](#-project-structure)
- [How It Works — Full Pipeline](#-how-it-works--full-pipeline)
- [API Reference](#-api-reference)
- [Frontend UI](#-frontend-ui)
- [TTS System](#-text-to-speech-tts-system)
- [Detection Pipeline Details](#-detection-pipeline-details)
- [Currency Detection](#-currency-detection)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Configuration Reference](#-configuration-reference)
- [Dependencies](#-dependencies)

---

## 🎯 Problem Statement

Visually impaired individuals face significant challenges in navigating everyday environments and handling physical currency. Existing assistive technologies are either prohibitively expensive, require specialized hardware, or lack the accuracy needed for practical use.

**DivyaDrishti** solves this by leveraging a smartphone camera (already owned by the user), a laptop as a processing server, and open-source AI models — making a sophisticated assistive vision system accessible at near-zero additional cost.

---

## 💡 Solution Overview

DivyaDrishti operates as a **client–server** system:

- The **mobile phone** (Android, using IP Webcam app or QPython) acts as a portable camera streaming JPEG frames over Wi-Fi.
- The **laptop/server** receives frames, runs YOLOv8 object detection in real time, and serves a web dashboard showing live annotated video.
- A **text-to-speech engine** on the server announces detected objects verbally, giving auditory feedback to the visually impaired user.
- Users can switch between a **general object detection model** and a **custom-trained Indian currency detection model** through the web UI.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          MOBILE DEVICE                           │
│                                                                  │
│   ┌────────────────┐     client.py / IP Webcam App              │
│   │  Phone Camera  │ ──► Captures JPEG frames (20 FPS)          │
│   └────────────────┘     POST /upload ──────────────────────►   │
└──────────────────────────────────────────────────────────────────┘
                                    │  HTTP (Wi-Fi LAN)
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                       LAPTOP / SERVER                            │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     server.py (Flask)                   │    │
│  │                                                         │    │
│  │  ┌────────────┐   ┌────────────────┐  ┌─────────────┐  │    │
│  │  │ /upload    │   │  detect_objects │  │  generate   │  │    │
│  │  │ (receives  │──►│  (YOLOv8n /    │─►│ _frames()   │  │    │
│  │  │  frames)   │   │  currency_best) │  │  (MJPEG)    │  │    │
│  │  └────────────┘   └────────────────┘  └─────────────┘  │    │
│  │                          │                    │          │    │
│  │                          ▼                    ▼          │    │
│  │                  ┌──────────────┐    ┌──────────────┐   │    │
│  │                  │ Persistence  │    │ Web Browser  │   │    │
│  │                  │  Tracker     │    │ Dashboard    │   │    │
│  │                  │ (3s delay)   │    │ /video_feed  │   │    │
│  │                  └──────────────┘    └──────────────┘   │    │
│  │                          │                              │    │
│  │                          ▼                              │    │
│  │                  ┌──────────────┐                       │    │
│  │                  │ TTSManager   │                       │    │
│  │                  │  (pyttsx3)   │                       │    │
│  │                  │  Voice Output│                       │    │
│  │                  └──────────────┘                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core Features
| Feature | Description |
|---|---|
| 📡 **Real-Time Video Streaming** | MJPEG stream from phone to server at ~20 FPS |
| 🤖 **YOLOv8 Object Detection** | Detects 80 COCO classes (people, cars, bottles, etc.) |
| 💰 **Currency Detection** | Custom-trained model for Indian Rupee notes & coins |
| 🔊 **Text-to-Speech Announcements** | Verbally announces confirmed detections |
| ⏱️ **Persistence-Based Confirmation** | Objects must be visible for 3 seconds to be announced |
| 🔄 **Model Toggle** | Switch between General and Currency models on-demand |
| 📊 **Live Dashboard** | Real-time stats: detected count, tracking count, FPS, stream status |
| 🌙 **Dark Mode** | Toggle-able dark/light theme with smooth transitions |
| 📱 **Responsive UI** | Works on desktop and mobile browsers |

### Smart Detection Features
| Feature | Description |
|---|---|
| ⏳ **3-Second Confirmation Delay** | Prevents false positives from brief observations |
| 🔄 **4-Second Announcement Cooldown** | Avoids repetitive voice announcements |
| 📊 **Progress Bar Visualization** | Visual timer bar showing how long an object has been tracked |
| 🎯 **ROI Cropping for Currency** | Crops center 60% of frame for accurate currency classification |
| 🧹 **Auto-History Cleanup** | Removes disappeared objects from tracking history after 2 seconds |
| ✅ **Confidence Thresholds** | 0.50 for general detection, 0.85 for currency (higher precision) |

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Role |
|---|---|---|
| **Python** | 3.10+ | Core language |
| **Flask** | 3.0.0 | Web framework, REST API, MJPEG streaming |
| **Ultralytics (YOLOv8)** | 8.1.0 | Object detection & classification |
| **OpenCV** | 4.8.1.78 | Frame decoding, annotation, JPEG encoding |
| **PyTorch** | Auto (via Ultralytics) | Deep learning inference engine |
| **NumPy** | 1.24.3 | Frame buffer manipulation |
| **pyttsx3** | 2.90 | Cross-platform text-to-speech |
| **threading** | stdlib | Concurrent TTS worker, frame locking |
| **queue** | stdlib | Thread-safe TTS announcement queue |
| **collections** | stdlib | `defaultdict`, `deque` for tracking history |

### Frontend
| Technology | Role |
|---|---|
| **HTML5** | Semantic page structure |
| **Vanilla CSS** (4 files) | Glassmorphism design system, animations, dark mode |
| **Vanilla JavaScript** (3 files) | Dynamic UI updates, fetch API, TTS controls, theme toggle |
| **Google Fonts (Inter)** | Typography |
| **MJPEG `<img>` tag** | Live video display (no WebSocket needed) |

### Client (Phone-Side)
| Technology | Role |
|---|---|
| **Python + Requests** | `client.py` — fetches frames from IP Webcam, POSTs to server |
| **IP Webcam (Android App)** | Turns phone into a Wi-Fi camera server |
| **QPython3 (Android)** | Runs `client.py` directly on the phone |

### Infrastructure
| Tool | Role |
|---|---|
| **Conda (myenv)** | Environment management |
| **Bash scripts** | `run_server.sh`, `run_client.sh` — one-command launchers |

---

## 🤖 AI Models

### 1. General Detection Model — `yolov8n.pt`

| Property | Value |
|---|---|
| **Model** | YOLOv8 Nano (YOLOv8n) |
| **Source** | Ultralytics (pre-trained on COCO dataset) |
| **Type** | Object Detection |
| **Classes** | 80 COCO classes |
| **Examples** | person, car, bicycle, bottle, cup, chair, dog, cat, laptop, phone, etc. |
| **File Size** | ~6.5 MB |
| **Confidence Threshold** | ≥ 0.50 |
| **Inference** | Bounding boxes with class labels |
| **Annotation Color** | Green `(0, 255, 0)` for confirmed detections |

**COCO Classes include:** person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### 2. Currency Detection Model — `currency_best.pt`

| Property | Value |
|---|---|
| **Model** | YOLOv8 (custom trained) |
| **Type** | Image Classification |
| **Domain** | Indian Rupee (INR) notes and coins |
| **File Size** | ~2.9 MB |
| **Confidence Threshold** | ≥ 0.85 (strict — classification models always output a prediction) |
| **Input** | Center 60% ROI of the camera frame |
| **Annotation Color** | Gold/Yellow `(0, 215, 255)` for confirmed detections |
| **Training Data** | Custom `currency_split` dataset |

> **Why such a high confidence threshold for currency?**  
> Classification models always predict exactly one class (they cannot say "nothing"). Without a strict gate, the model would always output a denomination even when no currency is in view. The 0.85 threshold acts as an "is there actually a note here?" filter.

#### How Currency Detection Works
1. User switches to **Currency Mode** via the UI toggle
2. Server crops the **center 60%** of each frame (the Region of Interest where the user naturally points)
3. The cropped ROI is passed to `currency_best.pt`
4. A white guide rectangle is drawn on screen — **"Align note here"**
5. If confidence ≥ 0.85, the denomination is confirmed and spoken aloud
6. Confirmed detections are shown with a gold bounding box

#### Model Training Details
- **Dataset:** `currency_split/` directory (train/val/test split)
- **Evaluation artifacts:** `confusion_matrix.png`, `f1_per_class.png` (included in project)
- **Training notebook:** `test.ipynb`

---

## 📁 Project Structure

```
DivyaDrishti/
│
├── server.py                    # 🧠 Main Flask server — detection, TTS, streaming
├── client.py                    # 📱 Phone-side client — fetches & uploads frames
│
├── yolov8n.pt                   # General YOLOv8 Nano model (80 COCO classes)
├── currency_best.pt             # Custom currency classification model (INR)
│
├── run_server.sh                # 🚀 One-command server launcher (conda + Python)
├── run_client.sh                # 📡 One-command client launcher (conda + Python)
│
├── templates/
│   └── index.html               # Flask Jinja2 HTML template (main dashboard)
│
├── static/
│   ├── css/
│   │   ├── main.css             # Design system, CSS variables, glassmorphism
│   │   ├── components.css       # Card, button, badge, slider components
│   │   ├── animations.css       # Keyframe animations, page transitions
│   │   └── dark-mode.css        # Dark theme overrides
│   ├── js/
│   │   ├── app.js               # Main app logic, polling, detection rendering
│   │   ├── tts-controls.js      # TTS toggle, volume/rate sliders, test button
│   │   └── theme-toggle.js      # Dark/light mode toggle (localStorage)
│   ├── assets/                  # Static assets directory
│   └── uploads/                 # Uploaded frame staging (if applicable)
│
├── packages/
│   ├── requirements.txt         # Python dependencies (server-side)
│   ├── server side              # Notes on server environment
│   └── client side              # Notes on client environment
│
├── currency_split/              # Training dataset for currency model
│   └── (train / val / test splits)
│
├── confusion_matrix.png         # Currency model evaluation — confusion matrix
├── f1_per_class.png             # Currency model evaluation — F1 scores per class
├── Model Workflow.png           # System workflow diagram
│
├── detect_images.py             # Utility: run detection on static images
├── test.ipynb                   # Jupyter notebook for model training/testing
│
└── LICENSE                      # MIT License
```

---

## ⚙️ How It Works — Full Pipeline

### Step 1: Frame Capture (Client Side)
```
Phone Camera
    └─► IP Webcam App serves JPEG at http://<phone-ip>:8080/shot.jpg
        └─► client.py polls every 50ms (~20 FPS)
            └─► POST /upload with raw JPEG bytes to http://<server-ip>:5001/upload
```

### Step 2: Frame Reception (Server)
```python
# /upload endpoint in server.py
frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
with frame_lock:
    latest_frame = frame        # Thread-safe write
    is_streaming = True
```

### Step 3: Object Detection
```
generate_frames() loop (runs at ~30 FPS)
    └─► safely copies latest_frame
        └─► detect_objects(frame)
            ├─► model = models['general'] OR models['currency']  (based on active toggle)
            ├─► results = model(frame)
            │
            ├─── GENERAL MODEL (detection):
            │     results[0].boxes → iterate boxes
            │     filter by confidence ≥ 0.50
            │     collect: class, confidence, bbox
            │
            └─── CURRENCY MODEL (classification):
                  crop center 60% ROI
                  results[0].probs.top1 → top class
                  filter by confidence ≥ 0.85
                  collect: denomination, confidence, ROI bbox
```

### Step 4: Persistence Tracking
```
Object History maintained per class_name:
    {first_seen: timestamp, last_seen: timestamp, count: detections}

For each detected object:
    if not in history → add with first_seen = now
    if in history     → update last_seen

Confirmation logic:
    if (now - first_seen) >= 3.0 seconds AND object still visible:
        → Object is CONFIRMED ✅
        → Eligible for announcement and overlay

Cleanup:
    if (now - last_seen) > 2.0 seconds:
        → Remove from history (object disappeared)
```

### Step 5: TTS Announcement
```
Announcement filter (anti-spam):
    if class NOT in announced_objects → announce immediately
    if class in announced_objects AND cooldown (4s) expired → re-announce

TTSManager (separate thread):
    announcement_queue (Queue) ──► worker thread
                                        └─► pyttsx3 engine.say(text)
                                            engine.runAndWait()
Voice: Samantha (macOS) / system female voice
Rate: 200 WPM | Volume: 1.0 (max)
```

### Step 6: Video Streaming to Browser
```
generate_frames() yields MJPEG frames:
    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + processed_jpeg + b'\r\n'

Browser <img src="/video_feed"> receives multipart/x-mixed-replace stream
    → Displays live annotated video at ~30 FPS
```

### Step 7: UI Polling (JavaScript)
```javascript
// Every 1 second:
fetch('/status')     → updates stream badge, tracking count, active model
fetch('/detections') → renders Confirmed Detections list with confidence %
```

---

## 🌐 API Reference

All endpoints run on `http://<server-ip>:5001`

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Renders the main web dashboard |
| `GET` | `/video_feed` | MJPEG live video stream |
| `POST` | `/upload` | Receives camera frame (multipart/form-data, field: `frame`) |
| `POST` | `/detect` | Single image detection (multipart/form-data, field: `image`) |
| `GET` | `/detections` | Returns current confirmed detections as JSON |
| `GET` | `/status` | Server + stream status |
| `GET` | `/model_info` | Info about loaded models and settings |
| `GET` | `/reset_tracking` | Clears object tracking history |
| `POST` | `/set_model` | Switch active model (body: `{"model": "general" \| "currency"}`) |

### TTS Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/tts/toggle` | Enable/disable TTS (body: `{"enabled": true \| false}`) |
| `GET` | `/tts/settings` | Get current TTS settings (enabled, rate, volume, cooldown) |
| `POST` | `/tts/settings` | Update TTS settings (body: `{"rate": 200, "volume": 1.0}`) |
| `POST` | `/tts/test` | Announce custom text (body: `{"text": "Hello World"}`) |

### Sample API Responses

**`GET /status`**
```json
{
  "streaming": true,
  "detections_count": 2,
  "general_model_loaded": true,
  "currency_model_loaded": true,
  "active_model": "general",
  "objects_tracking": 3,
  "detection_delay": 3.0
}
```

**`GET /detections`**
```json
{
  "success": true,
  "count": 2,
  "timestamp": 1711234567.89,
  "detections": [
    {
      "class": "bottle",
      "confidence": 0.87,
      "bbox": [120.0, 50.0, 280.0, 350.0],
      "visible_for": 4.2,
      "model_source": "general"
    }
  ],
  "settings": {
    "detection_delay": 3.0,
    "min_confidence": 0.5
  }
}
```

**`POST /set_model`**
```json
// Request body:
{ "model": "currency" }

// Response:
{ "success": true, "active_model": "currency" }
```

---

## 🖥️ Frontend UI

The dashboard is a single-page application served by Flask at `/`.

### Design System
- **Glassmorphism** cards with `backdrop-filter: blur(20px)` and semi-transparent backgrounds
- **Gradient palette:** Purple-blue (`#667eea → #764ba2`), Cyan (`#00d2ff → #3a7bd5`), Pink (`#f093fb → #f5576c`)
- **Typography:** Inter (Google Fonts), weights 300/400/600/700
- **Dark mode:** Full CSS variable swap (`[data-theme="dark"]`), persisted in `localStorage`
- **Responsive:** Fluid grid collapses gracefully below 768px

### UI Sections

| Section | Description |
|---|---|
| **Header** | App title, TTS status badge, Active Model badge, Dark mode toggle |
| **Stats Grid** | 4 cards: Detected Objects, Tracking, FPS, Stream Status |
| **Live Video Feed** | MJPEG stream with LIVE badge and FPS counter overlay |
| **Confirmed Detections Panel** | List of all objects held steady for 3+ seconds with confidence % |
| **TTS Control Panel** | Enable/disable toggle, Volume slider (0–100%), Speech Rate slider (50–300 WPM), Test Voice button |
| **Model Toggle** | Switch between 🌍 General Model and 💰 Currency Model |
| **Controls** | Reset Tracking, Refresh Page |

### JavaScript Modules

| File | Responsibility |
|---|---|
| `app.js` | Core class `DivyaDrishtiApp` — polls `/status` & `/detections` every second, animates counters, renders detection cards, handles model switching |
| `tts-controls.js` | Volume/rate sliders with real-time POST to `/tts/settings`, toggle switch, test button |
| `theme-toggle.js` | Dark/light mode toggle, saves preference to `localStorage` |

---

## 🔊 Text-to-Speech (TTS) System

The TTS system is implemented as a **dedicated background thread** to prevent blocking the Flask request-handling thread.

### TTSManager Class

```
TTSManager
├── __init__()        → starts worker thread
├── _start_worker()   → spawns daemon thread
├── _announcement_worker()
│       ├── pyttsx3.init()           (thread-local engine)
│       ├── setProperty('rate', 200) (words per minute)
│       ├── setProperty('volume', 1.0)
│       ├── Voice selection priority: Samantha → Victoria → Fiona → Alice → voices[1]
│       └── loop: poll Queue → engine.say(text) → engine.runAndWait()
├── announce(text)    → puts text in Queue (non-blocking)
├── toggle(enabled)   → enable/disable flag (thread-safe with Lock)
├── set_rate(rate)    → updates engine property
├── set_volume(vol)   → clamped to [0.0, 1.0]
└── shutdown()        → stops worker thread
```

### TTS Configuration

| Parameter | Default | Description |
|---|---|---|
| `TTS_ENABLED` | `True` | Global on/off switch |
| `TTS_RATE` | `200 WPM` | Speech speed (adjustable 50–300) |
| `TTS_VOLUME` | `1.0` | Volume level (max) |
| `ANNOUNCEMENT_COOLDOWN` | `4.0s` | Same object won't be re-announced within this window |

### Announcement Format

The system generates grammatically natural announcements:
- 1 object: `"bottle"`
- 2 objects: `"bottle and person"`
- 3+ objects: `"bottle, person, and chair"`

---

## 🔬 Detection Pipeline Details

### Confidence & Timing Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `MIN_CONFIDENCE` | `0.50` | Minimum score for general object to enter tracking |
| `CURRENCY_MIN_CONFIDENCE` | `0.85` | Minimum score for currency to be confirmed |
| `DETECTION_DELAY` | `3.0s` | Object must be visible for 3 seconds before confirming |
| `ANNOUNCEMENT_COOLDOWN` | `4.0s` | Minimum time between repeat announcements of the same object |
| `processing_interval` | `0.5s` | Tracking state update frequency |
| `last_seen_timeout` | `2.0s` | Time before removing disappeared object from history |
| `Stream FPS` | `~30 FPS` | `generate_frames()` runs at 33ms/frame |

### Visual Overlays on Video Feed

| Overlay | Description |
|---|---|
| **YOLO bounding boxes** | Default YOLOv8 annotations (class label + confidence) |
| **Green box** | Confirmed general objects (`(0, 255, 0)`, line weight 3) |
| **Gold box** | Confirmed currency detections (`(0, 215, 255)`, line weight 3) |
| **Progress bar** | Below unconfirmed objects — fills red→green over 3 seconds |
| **HUD top-left** | Current mode (General / Currency 💰) + tracking count |
| **Currency guide** | White rectangle — "Align note here" for currency mode |

---

## 💰 Currency Detection

### Switching Modes

1. Open the dashboard in a browser at `http://<server-ip>:5001`
2. Scroll to the **Detection Model** section
3. Click **💰 Currency Model** button
4. The server calls `POST /set_model {"model": "currency"}`
5. Object tracking history is cleared to avoid stale data
6. A white alignment guide appears on the video feed

### Best Practices for Currency Detection

- **Hold the note steady** — the 3-second confirmation delay requires stability
- **Align the note within the white guide box** (center 60% of frame)
- **Good lighting** significantly improves accuracy
- **Fill the frame** — bring the note close to the camera
- The strict 0.85 confidence threshold means uncertain readings are silently ignored

### Currency Model Training

- Dataset stored in `currency_split/` (train/val/test structure)
- Training explored in `test.ipynb`
- Evaluation: `confusion_matrix.png` and `f1_per_class.png` show per-class performance

---

## 🚀 Installation & Setup

### Prerequisites

- macOS / Linux (Windows should work with minor path adjustments)
- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Android phone with [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app
- Both devices on the same **Wi-Fi network**

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/DivyaDrishti.git
cd DivyaDrishti
```

### 2. Create the Conda Environment

```bash
conda create -n myenv python=3.10 -y
conda activate myenv
```

### 3. Install Dependencies

```bash
pip install -r packages/requirements.txt
```

> **Note:** PyTorch, TorchVision, and Pillow are automatically installed as dependencies of `ultralytics`.

### 4. Verify Models Are Present

```bash
ls *.pt
# Should show: yolov8n.pt  currency_best.pt
```

If `yolov8n.pt` is missing, it will be auto-downloaded on first run by Ultralytics.

### 5. Configure Network IP

Edit `server.py` line 648:
```python
ip = "10.167.7.240"  # ← Replace with YOUR laptop's IP address
```

Find your IP with:
```bash
# macOS/Linux:
ifconfig | grep "inet "
# or:
hostname -I
```

Edit `client.py` line 9:
```python
LAPTOP_URL = "http://10.90.19.240:5001/upload"  # ← Your laptop IP
```

And line 8:
```python
IP_WEBCAM_URL = "http://192.168.1.10:8080/shot.jpg"  # ← Your phone IP (shown in IP Webcam app)
```

---

## ▶️ Running the Project

### Start the Server (Laptop)

**Option A — Shell Script (recommended):**
```bash
chmod +x run_server.sh
./run_server.sh
```

**Option B — Manual:**
```bash
conda activate myenv
python server.py
```

You should see:
```
============================================================
🚀 YOLOv8 OBJECT DETECTION STREAM SERVER
============================================================
📍 Local access:  http://localhost:5001
🌐 Network access: http://10.167.7.240:5001
============================================================
✅ YOLOv8 general model loaded. Classes: 80
✅ YOLOv8 currency model loaded. Classes: <N>
🔊 Using voice: Samantha
🔊 TTS engine initialized
```

### Start the Client (Phone or Laptop)

1. Open the **IP Webcam** app on your Android phone
2. Start the server inside the app (note the IP shown, e.g. `192.168.1.10:8080`)
3. On the laptop, run:

```bash
chmod +x run_client.sh
./run_client.sh
```

Or directly:
```bash
conda activate myenv
python client.py
```

### Open the Dashboard

Navigate to `http://<your-laptop-ip>:5001` in any browser (phone or laptop).

---

## ⚙️ Configuration Reference

All runtime parameters are defined as constants at the top of `server.py`:

| Constant | Default | Description |
|---|---|---|
| `active_model_name` | `'general'` | Starting model (`'general'` or `'currency'`) |
| `DETECTION_DELAY` | `3.0` | Seconds an object must be visible before confirming |
| `MIN_CONFIDENCE` | `0.50` | General model minimum confidence score |
| `CURRENCY_MIN_CONFIDENCE` | `0.85` | Currency model minimum confidence score |
| `processing_interval` | `0.5` | Seconds between tracking state updates |
| `TTS_ENABLED` | `True` | Enable TTS on startup |
| `TTS_RATE` | `200` | Words per minute for speech synthesis |
| `TTS_VOLUME` | `1.0` | Speaker volume (0.0–1.0) |
| `ANNOUNCEMENT_COOLDOWN` | `4.0` | Repeat announcement cooldown in seconds |
| `port` | `5001` | Flask server port |

---

## 📦 Dependencies

### Server-Side (`packages/requirements.txt`)

```
flask==3.0.0
opencv-python==4.8.1.78
ultralytics==8.1.0
numpy==1.24.3
requests==2.31.0
pyttsx3==2.90
```

**Auto-installed transitive dependencies:**
- `torch` + `torchvision` (deep learning, via ultralytics)
- `pillow` (image handling, via OpenCV/ultralytics)
- `pyyaml` (configuration, via ultralytics)

### Client-Side (Phone / QPython3)

```
requests
```
*(The client only fetches JPEG frames and HTTP POSTs them; no ML libraries needed on phone)*

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — for the state-of-the-art object detection framework
- [IP Webcam (Android)](https://play.google.com/store/apps/details?id=com.pas.webcam) — for turning any Android phone into a network camera
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) — for offline, cross-platform text-to-speech
- [Flask](https://flask.palletsprojects.com/) — for the lightweight yet powerful web framework

---

<div align="center">

**DivyaDrishti** — Giving sight to the unseen. 🔍

*Built with ❤️ for accessibility and inclusion.*

</div>
