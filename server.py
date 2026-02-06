from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import socket
import base64
from ultralytics import YOLO
import json
from collections import defaultdict, deque
import pyttsx3
from queue import Queue

app = Flask(__name__)

# Global variables for frame sharing
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
is_streaming = False
last_frame_time = 0

# NEW: Persistence tracking variables
object_history = defaultdict(list)  # Track when objects were first seen
DETECTION_DELAY = 3.0  # 3 seconds delay before confirming object
MIN_CONFIDENCE = 0.5   # Minimum confidence threshold
last_processing_time = 0
processing_interval = 0.5  # Process detections every 0.5 seconds

# TTS Configuration
TTS_ENABLED = True  # Enable/disable TTS globally
TTS_RATE = 200  # Speech rate (words per minute) - higher for sweeter, higher pitch
TTS_VOLUME = 1.0  # Volume level (0.0 to 1.0) - maximum volume
ANNOUNCEMENT_COOLDOWN = 4.0  # Seconds before re-announcing same object
announced_objects = {}  # Track when objects were last announced

# TTS Manager Class
class TTSManager:
    def __init__(self):
        self.enabled = TTS_ENABLED
        self.announcement_queue = Queue()
        self.engine = None
        self.worker_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Initialize TTS engine in separate thread
        self._start_worker()
    
    def _start_worker(self):
        """Start the TTS worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._announcement_worker, daemon=True)
        self.worker_thread.start()
    
    def _announcement_worker(self):
        """Worker thread that processes announcement queue"""
        try:
            # Initialize engine in this thread (pyttsx3 requirement)
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', TTS_RATE)
            self.engine.setProperty('volume', TTS_VOLUME)
            
            # Try to set a better voice if available
            voices = self.engine.getProperty('voices')
            if voices:
                # Try to find the sweetest female voice
                # On macOS: 0=Alex(male), 1=Alice, 17=Samantha, 35=Victoria
                # Alice is usually sweeter than Samantha
                preferred_voices = ['Samantha', 'Victoria', 'Fiona', 'Alice']
                selected_voice = None
                
                # Try to find preferred voice by name
                for pref in preferred_voices:
                    for voice in voices:
                        if pref.lower() in voice.name.lower():
                            selected_voice = voice
                            break
                    if selected_voice:
                        break
                
                # Fallback to female voice (index 1)
                if not selected_voice and len(voices) > 1:
                    selected_voice = voices[1]
                elif not selected_voice:
                    selected_voice = voices[0]
                
                self.engine.setProperty('voice', selected_voice.id)
                print(f"🔊 Using voice: {selected_voice.name}")
            
            print("🔊 TTS engine initialized")
            
            while self.running:
                try:
                    # Wait for announcement with timeout
                    if not self.announcement_queue.empty():
                        text = self.announcement_queue.get(timeout=0.1)
                        if text and self.enabled:
                            print(f"🔊 Announcing: {text}")
                            self.engine.say(text)
                            self.engine.runAndWait()
                    else:
                        time.sleep(0.1)
                except Exception as e:
                    print(f"TTS worker error: {e}")
                    time.sleep(0.1)
        except Exception as e:
            print(f"❌ Failed to initialize TTS engine: {e}")
            self.enabled = False
    
    def announce(self, text):
        """Add announcement to queue"""
        if self.enabled and text:
            self.announcement_queue.put(text)
    
    def toggle(self, enabled=None):
        """Enable or disable TTS"""
        with self.lock:
            if enabled is None:
                self.enabled = not self.enabled
            else:
                self.enabled = enabled
            return self.enabled
    
    def set_rate(self, rate):
        """Set speech rate"""
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """Set volume (0.0 to 1.0)"""
        if self.engine:
            self.engine.setProperty('volume', max(0.0, min(1.0, volume)))
    
    def shutdown(self):
        """Shutdown TTS engine"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)

# Initialize TTS Manager
tts_manager = TTSManager()

# Load YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
    print(f"✅ YOLOv8 model loaded. Classes: {len(model.names)}")
    print(f"⏰ Detection delay: {DETECTION_DELAY} seconds")
except Exception as e:
    print(f"❌ Failed to load YOLOv8 model: {e}")
    model = None

def generate_announcement(confirmed_detections, current_time):
    """Generate natural language announcement for confirmed detections"""
    global announced_objects
    
    if not confirmed_detections:
        return None
    
    # Filter objects that need announcement (not recently announced)
    objects_to_announce = []
    for det in confirmed_detections:
        class_name = det['class']
        
        # Check if this object should be announced
        if class_name not in announced_objects:
            # First time seeing this object
            objects_to_announce.append(class_name)
            announced_objects[class_name] = current_time
        elif current_time - announced_objects[class_name] > ANNOUNCEMENT_COOLDOWN:
            # Cooldown period expired, re-announce
            objects_to_announce.append(class_name)
            announced_objects[class_name] = current_time
    
    # Clean up announced_objects for objects no longer detected
    detected_classes = {det['class'] for det in confirmed_detections}
    for class_name in list(announced_objects.keys()):
        if class_name not in detected_classes:
            # Object disappeared, remove from history
            if current_time - announced_objects[class_name] > 5.0:
                del announced_objects[class_name]
    
    if not objects_to_announce:
        return None
    
    # Count objects by class
    object_counts = {}
    for cls in objects_to_announce:
        object_counts[cls] = object_counts.get(cls, 0) + 1
    
    # Generate natural language
    parts = []
    for cls, count in object_counts.items():
        if count == 1:
            parts.append(cls)
        elif count == 2:
            parts.append(f"two {cls}s")
        elif count == 3:
            parts.append(f"three {cls}s")
        else:
            parts.append(f"{count} {cls}s")
    
    # Construct announcement (without 'detected')
    if len(parts) == 0:
        return None
    elif len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{', '.join(parts[:-1])}, and {parts[-1]}"

def detect_objects(frame):
    """Run YOLO object detection on frame with 3-second delay"""
    global object_history, last_processing_time
    
    if model is None:
        return frame, []
    
    try:
        # Run detection
        results = model(frame)
        
        # Extract raw detections
        raw_detections = []
        annotated_frame = frame.copy()
        current_time = time.time()
        
        if results and results[0].boxes is not None:
            # Draw all detections (for visualization)
            annotated_frame = results[0].plot()
            
            # Extract detection data
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = model.names[cls_id]
                
                if conf >= MIN_CONFIDENCE:
                    raw_detections.append({
                        'class': cls_name,
                        'confidence': round(conf, 3),
                        'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        'coords': (int(x1), int(y1), int(x2), int(y2))
                    })
        
        # Process persistence tracking every processing_interval seconds
        confirmed_detections = []
        if current_time - last_processing_time >= processing_interval:
            # Update object history with current detections
            for det in raw_detections:
                class_name = det['class']
                if class_name not in object_history:
                    object_history[class_name] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'count': 1
                    }
                else:
                    object_history[class_name]['last_seen'] = current_time
                    object_history[class_name]['count'] += 1
            
            # Check which objects have been visible for DETECTION_DELAY seconds
            confirmed_classes = []
            for class_name, data in list(object_history.items()):
                visibility_time = current_time - data['first_seen']
                
                # Remove if not seen in last 2 seconds
                if current_time - data['last_seen'] > 2.0:
                    del object_history[class_name]
                    continue
                
                # Check if visible for required delay
                if visibility_time >= DETECTION_DELAY:
                    confirmed_classes.append(class_name)
                    
                    # Find corresponding bbox
                    matching_det = next((d for d in raw_detections if d['class'] == class_name), None)
                    if matching_det:
                        confirmed_detections.append({
                            'class': class_name,
                            'confidence': matching_det['confidence'],
                            'bbox': matching_det['bbox'],
                            'visible_for': round(visibility_time, 1)
                        })
            
            last_processing_time = current_time
            
            # Generate and trigger TTS announcement
            announcement = generate_announcement(confirmed_detections, current_time)
            if announcement:
                print(f"📢 Generated announcement: {announcement}")
                tts_manager.announce(announcement)
            elif confirmed_detections:
                print(f"⚠️ Confirmed detections but no announcement: {[d['class'] for d in confirmed_detections]}")
        
        # Draw confirmed objects with different style
        for det in confirmed_detections:
            x1, y1, x2, y2 = det['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw thick green box for confirmed objects
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Add confirmation label
            label = f"{det['class']} ✓ ({det['visible_for']}s)"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add timer display on frame
        for class_name, data in object_history.items():
            if class_name not in [d['class'] for d in confirmed_detections]:
                # Find bbox for timing display
                matching_det = next((d for d in raw_detections if d['class'] == class_name), None)
                if matching_det:
                    x1, y1, x2, y2 = matching_det['coords']
                    elapsed = current_time - data['first_seen']
                    
                    # Draw timing indicator
                    progress = min(elapsed / DETECTION_DELAY, 1.0)
                    bar_width = int(100 * progress)
                    
                    # Draw progress bar
                    cv2.rectangle(annotated_frame, (x1, y2+5), (x1+bar_width, y2+10), 
                                 (0, int(255*progress), int(255*(1-progress))), -1)
                    
                    # Draw time text
                    time_text = f"{elapsed:.1f}s/{DETECTION_DELAY}s"
                    cv2.putText(annotated_frame, time_text, (x1, y2+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add info overlay
        cv2.putText(annotated_frame, f"Delay: {DETECTION_DELAY}s | Hold object steady", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Objects tracking: {len(object_history)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        
        return annotated_frame, confirmed_detections
        
    except Exception as e:
        print(f"Detection error: {e}")
        return frame, []

def generate_frames():
    """Generate video frames for streaming with object detection"""
    global latest_frame, latest_detections, object_history
    
    while True:
        with frame_lock:
            if latest_frame is not None:
                # Run object detection on the frame
                processed_frame, detections = detect_objects(latest_frame.copy())
                latest_detections = detections  # Store for API access
                
                # Encode the frame
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
            else:
                # Black frame when no stream - reset tracking
                object_history.clear()
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "Waiting for video stream...", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                frame_bytes = buffer.tobytes()
                latest_detections = []
                
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_frame():
    global latest_frame, is_streaming, last_frame_time
    try:
        if 'frame' not in request.files:
            return 'No frame', 400
        
        file = request.files['frame']
        img_data = file.read()
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            with frame_lock:
                latest_frame = frame
                is_streaming = True
                last_frame_time = time.time()
            return 'OK', 200
        return 'Decode failed', 400
        
    except Exception as e:
        print(f"Upload error: {e}")
        return 'Error', 500

@app.route('/detect', methods=['POST'])
def detect_single_image():
    """API endpoint for single image detection"""
    if model is None:
        return jsonify({'error': 'Model not loaded', 'success': False}), 500
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'success': False}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Read image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image', 'success': False}), 400
        
        # Run detection (bypass delay for single images)
        annotated_img, detections = detect_objects(img)
        
        # Convert to base64
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', annotated_img_rgb)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        return jsonify({
            'success': True,
            'detections': detections,
            'count': len(detections),
            'annotated_image': f"data:image/jpeg;base64,{img_base64}",
            'image_size': {'height': img.shape[0], 'width': img.shape[1]},
            'note': 'Single image detection (no delay filter)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/detections')
def get_detections():
    """Get latest detections from video stream"""
    with frame_lock:
        detections_copy = latest_detections.copy()
    
    return jsonify({
        'success': True,
        'detections': detections_copy,
        'count': len(detections_copy),
        'timestamp': time.time(),
        'settings': {
            'detection_delay': DETECTION_DELAY,
            'min_confidence': MIN_CONFIDENCE
        }
    })

@app.route('/model_info')
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded', 'success': False}), 500
    
    return jsonify({
        'success': True,
        'model': 'YOLOv8n',
        'classes_count': len(model.names),
        'classes': model.names,
        'detection_settings': {
            'delay_seconds': DETECTION_DELAY,
            'min_confidence': MIN_CONFIDENCE,
            'description': f'Objects must be visible for {DETECTION_DELAY} seconds'
        }
    })

@app.route('/status')
def status():
    global is_streaming, last_frame_time, object_history
    # If no frame in 5 seconds, consider disconnected
    if time.time() - last_frame_time > 5:
        is_streaming = False
        object_history.clear()
    
    return jsonify({
        'streaming': is_streaming,
        'detections_count': len(latest_detections),
        'model_loaded': model is not None,
        'objects_tracking': len(object_history),
        'detection_delay': DETECTION_DELAY
    })

@app.route('/reset_tracking')
def reset_tracking():
    """Reset object tracking history"""
    global object_history
    object_history.clear()
    return jsonify({
        'success': True,
        'message': 'Object tracking reset',
        'tracking_count': 0
    })

@app.route('/tts/toggle', methods=['POST'])
def tts_toggle():
    """Toggle TTS on/off"""
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled')
        
        new_state = tts_manager.toggle(enabled)
        
        return jsonify({
            'success': True,
            'tts_enabled': new_state,
            'message': f"TTS {'enabled' if new_state else 'disabled'}"
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/tts/settings', methods=['GET', 'POST'])
def tts_settings():
    """Get or update TTS settings"""
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'settings': {
                'enabled': tts_manager.enabled,
                'rate': TTS_RATE,
                'volume': TTS_VOLUME,
                'announcement_cooldown': ANNOUNCEMENT_COOLDOWN
            }
        })
    else:
        try:
            data = request.get_json() or {}
            
            if 'rate' in data:
                tts_manager.set_rate(data['rate'])
            if 'volume' in data:
                tts_manager.set_volume(data['volume'])
            
            return jsonify({
                'success': True,
                'message': 'TTS settings updated',
                'settings': {
                    'enabled': tts_manager.enabled,
                    'rate': data.get('rate', TTS_RATE),
                    'volume': data.get('volume', TTS_VOLUME)
                }
            })
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500

@app.route('/tts/test', methods=['POST'])
def tts_test():
    """Test TTS with custom text"""
    try:
        data = request.get_json() or {}
        text = data.get('text', 'Text to speech is working correctly')
        
        tts_manager.announce(text)
        
        return jsonify({
            'success': True,
            'message': f"Announced: {text}"
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    # YOUR SPECIFIC IP ADDRESS
    ip = "10.90.19.240"
    port = 5001
    
    print("=" * 60)
    print("🚀 YOLOv8 OBJECT DETECTION STREAM SERVER")
    print("=" * 60)
    print(f"📍 Local access:  http://localhost:{port}")
    print(f"🌐 Network access: http://{ip}:{port}")
    print("=" * 60)
    print("⏰ DELAYED DETECTION ENABLED")
    print(f"   Objects need {DETECTION_DELAY} seconds of visibility")
    print(f"   Hold objects steady for detection")
    print("=" * 60)
    print("🔊 TEXT-TO-SPEECH ENABLED")
    print(f"   TTS Status: {'ON' if TTS_ENABLED else 'OFF'}")
    print(f"   Announcement cooldown: {ANNOUNCEMENT_COOLDOWN}s")
    print("=" * 60)
    print("📡 Available endpoints:")
    print(f"   Home:           http://{ip}:{port}/")
    print(f"   Video stream:   http://{ip}:{port}/video_feed")
    print(f"   Single image:   http://{ip}:{port}/detect (POST)")
    print(f"   Live detections: http://{ip}:{port}/detections")
    print(f"   Model info:     http://{ip}:{port}/model_info")
    print(f"   Reset tracking: http://{ip}:{port}/reset_tracking")
    print(f"   Status:         http://{ip}:{port}/status")
    print(f"   TTS Toggle:     http://{ip}:{port}/tts/toggle (POST)")
    print(f"   TTS Settings:   http://{ip}:{port}/tts/settings (GET/POST)")
    print(f"   TTS Test:       http://{ip}:{port}/tts/test (POST)")
    print("=" * 60)
    print("📱 Use this URL in your phone client:")
    print(f"   http://{ip}:{port}")
    print("=" * 60)
    print(f"💡 TIP: Hold objects steady for {DETECTION_DELAY} seconds")
    print("=" * 60)
    
    # Run server on your specific IP address
    app.run(host=ip, port=port, debug=False, threaded=True)