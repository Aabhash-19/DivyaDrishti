from flask import Flask, Response, render_template, request
import cv2
import numpy as np
import threading
import time
import socket

app = Flask(__name__)

# Global variables for frame sharing
latest_frame = None
frame_lock = threading.Lock()
is_streaming = False
last_frame_time = 0

def generate_frames():
    """Generate video frames for streaming"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
            else:
                # Black frame when no stream
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                frame_bytes = buffer.tobytes()
                
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

@app.route('/status')
def status():
    global is_streaming, last_frame_time
    # If no frame in 5 seconds, consider disconnected
    if time.time() - last_frame_time > 5:
        is_streaming = False
    return {'streaming': is_streaming}

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
    port = 5001  # ← Using port 5001 instead of 5000
    
    print("=" * 60)
    print("🚀 PHONE CAMERA STREAM SERVER STARTED")
    print("=" * 60)
    print(f"📍 Local access:  http://localhost:{port}")
    print(f"🌐 Network access: http://{ip}:{port}")
    print("=" * 60)
    print("📱 Use this URL in your phone client:")
    print(f"   http://{ip}:{port}")
    print("=" * 60)
    print("Waiting for phone connection...")
    app.run(host='0.0.0.0', port=port, debug=False)