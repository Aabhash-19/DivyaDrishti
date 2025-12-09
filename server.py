from flask import Flask, Response, render_template, request, jsonify
import cv2
import numpy as np
import threading
import time
import socket
import base64
from ultralytics import YOLO
import json

app = Flask(__name__)

# Global variables for frame sharing
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
is_streaming = False
last_frame_time = 0

# Load YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
    print(f"✅ YOLOv8 model loaded. Classes: {len(model.names)}")
except Exception as e:
    print(f"❌ Failed to load YOLOv8 model: {e}")
    model = None

def detect_objects(frame):
    """Run YOLO object detection on frame"""
    if model is None:
        return frame, []
    
    try:
        # Run detection
        results = model(frame)
        
        # Extract detections
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names[cls_id]
                    
                    detections.append({
                        'class': cls_name,
                        'confidence': round(conf, 3),
                        'bbox': [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                    })
        
        # Draw bounding boxes on frame
        annotated_frame = results[0].plot()
        return annotated_frame, detections
    
    except Exception as e:
        print(f"Detection error: {e}")
        return frame, []

def generate_frames():
    """Generate video frames for streaming with object detection"""
    global latest_frame, latest_detections
    
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
                # Black frame when no stream
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
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
        
        # Run detection
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
            'image_size': {'height': img.shape[0], 'width': img.shape[1]}
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
        'timestamp': time.time()
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
        'classes': model.names
    })

@app.route('/status')
def status():
    global is_streaming, last_frame_time
    # If no frame in 5 seconds, consider disconnected
    if time.time() - last_frame_time > 5:
        is_streaming = False
    
    return jsonify({
        'streaming': is_streaming,
        'detections_count': len(latest_detections),
        'model_loaded': model is not None
    })

def get_ip_address():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

if __name__ == '__main__':
    ip = get_ip_address()
    port = 5001
    
    print("=" * 60)
    print("🚀 YOLOv8 OBJECT DETECTION STREAM SERVER")
    print("=" * 60)
    print(f"📍 Local access:  http://localhost:{port}")
    print(f"🌐 Network access: http://{ip}:{port}")
    print("=" * 60)
    print("📡 Available endpoints:")
    print(f"   Home:           http://{ip}:{port}/")
    print(f"   Video stream:   http://{ip}:{port}/video_feed")
    print(f"   Single image:   http://{ip}:{port}/detect (POST)")
    print(f"   Live detections: http://{ip}:{port}/detections")
    print(f"   Model info:     http://{ip}:{port}/model_info")
    print(f"   Status:         http://{ip}:{port}/status")
    print("=" * 60)
    print("📱 Use this URL in your phone client:")
    print(f"   http://{ip}:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)