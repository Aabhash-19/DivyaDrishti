import requests
import time

print("🎥 IP WEBCAM LIVE VIDEO STREAM")
print("==================================")


IP_WEBCAM_URL = "http://192.168.1.10:8080/shot.jpg"
LAPTOP_URL = "http://192.168.1.23:5001/upload"

print(f"📱 IP Webcam: {IP_WEBCAM_URL}")
print(f"💻 Laptop: {LAPTOP_URL}")
print("==================================")

def test_connections():
    """Test if IP Webcam and laptop are accessible"""
    print("Testing connections...")
    
    
    try:
        response = requests.get(IP_WEBCAM_URL, timeout=10)
        if response.status_code == 200:
            print("✅ IP Webcam connected - LIVE feed available")
            return True
        else:
            print(f"❌ IP Webcam error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to IP Webcam: {e}")
        return False
    
    
    try:
        response = requests.get("http://192.168.1.23:5001", timeout=10)
        if response.status_code == 200:
            print("✅ Laptop server connected")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to laptop server: {e}")
        return False

def start_live_stream():
    """Start true live video streaming"""
    frame_count = 0
    success_count = 0
    start_time = time.time()
    
    print(f"\n🎬 STARTING LIVE VIDEO STREAM")
    print("📹 Streaming real-time camera feed")
    print("⚡ Smooth video transmission")
    print("🛑 Press STOP in QPython to end stream")
    print("==================================")
    
    try:
        while True:
            try:
                
                response = requests.get(IP_WEBCAM_URL, timeout=5)
                
                if response.status_code == 200:
                    
                    files = {'frame': (f'live_{frame_count}.jpg', response.content, 'image/jpeg')}
                    server_response = requests.post(LAPTOP_URL, files=files, timeout=3)
                    
                    if server_response.status_code == 200:
                        success_count += 1
                        status = "✅ LIVE"
                    else:
                        status = "❌ SERVER ERROR"
                else:
                    status = "❌ WEBCAM ERROR"
                    
            except requests.exceptions.Timeout:
                status = "⏰ TIMEOUT"
            except requests.exceptions.ConnectionError:
                status = "🔌 CONNECTION LOST"
            except Exception as e:
                status = f"⚠️ ERROR: {str(e)[:20]}"
            
            
            frame_count += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= 2.0:
                fps = success_count / elapsed_time
                print(f"📊 FPS: {fps:.1f} | Frames: {frame_count} | Status: {status}")
                frame_count = 0
                success_count = 0
                start_time = current_time
            
            
            time.sleep(0.05)  
            
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped by user")
    except Exception as e:
        print(f"\n❌ Stream error: {e}")


if test_connections():
    print("\n" + "="*50)
    print("🎉 ALL SYSTEMS GO! Starting LIVE video...")
    print("="*50)
    start_live_stream()
else:
    print("\n❌ Cannot start stream - fix connections first")
    print("\nTROUBLESHOOTING:")
    print("1. Make sure IP Webcam server is RUNNING")
    print("2. You should see camera video in IP Webcam app")
    print("3. Check laptop server is running: python server.py")