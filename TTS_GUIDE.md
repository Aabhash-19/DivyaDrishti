# Text-to-Speech Feature - User Guide

## Overview
The TTS system announces detected objects after they've been visible for 3 seconds, helping visually impaired users understand their environment.

## Features
- **Offline TTS** - Works without internet using pyttsx3
- **Smart announcements** - Only speaks when objects are confirmed (3s delay)
- **Anti-spam** - Won't repeat announcements for 10 seconds
- **Natural language** - "Person and laptop detected" instead of robotic lists
- **API control** - Enable/disable, adjust settings via HTTP

## Quick Start

### 1. Server is Running
The server automatically starts with TTS enabled. You should see:
```
🔊 TTS engine initialized
🔊 TEXT-TO-SPEECH ENABLED
   TTS Status: ON
   Announcement cooldown: 10.0s
```

### 2. Test TTS
```bash
curl -X POST http://10.90.19.240:5001/tts/test \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test"}'
```

### 3. How It Works
1. Point camera at object
2. Hold steady for 3 seconds
3. Hear announcement: "Person detected"
4. Same object won't be announced again for 10 seconds

## API Endpoints

### Toggle TTS On/Off
```bash
# Disable TTS
curl -X POST http://10.90.19.240:5001/tts/toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Enable TTS
curl -X POST http://10.90.19.240:5001/tts/toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'
```

### Get TTS Settings
```bash
curl http://10.90.19.240:5001/tts/settings
```

### Update TTS Settings
```bash
# Adjust speech rate (50-300 words per minute)
curl -X POST http://10.90.19.240:5001/tts/settings \
  -H "Content-Type: application/json" \
  -d '{"rate": 180}'

# Adjust volume (0.0 to 1.0)
curl -X POST http://10.90.19.240:5001/tts/settings \
  -H "Content-Type: application/json" \
  -d '{"volume": 0.8}'
```

## Configuration

Edit `server.py` to change defaults:

```python
TTS_ENABLED = True              # Enable/disable on startup
TTS_RATE = 150                  # Speech rate (words/min)
TTS_VOLUME = 0.9                # Volume (0.0 to 1.0)
ANNOUNCEMENT_COOLDOWN = 10.0    # Seconds before re-announcing
```

## Announcement Examples

| Detected Objects | Announcement |
|-----------------|--------------|
| 1 person | "Person detected" |
| 2 persons | "Two persons detected" |
| 1 person, 1 laptop | "Person and laptop detected" |
| 2 persons, 1 laptop, 1 book | "Two persons, laptop, and book detected" |

## Troubleshooting

### No Audio Output
- **macOS**: TTS uses built-in voices (should work automatically)
- **Linux**: Install espeak: `sudo apt-get install espeak`
- **Windows**: Uses SAPI5 (should work automatically)

### TTS Not Announcing
1. Check TTS is enabled: `curl http://10.90.19.240:5001/tts/settings`
2. Verify objects are held for 3+ seconds
3. Check 10-second cooldown hasn't blocked announcement
4. Look for "🔊 Announcing:" in server logs

### Adjust Voice
Edit `server.py` line ~66:
```python
# Change voice index (0 = male, 1 = female on macOS)
self.engine.setProperty('voice', voices[1].id)
```

## Advanced Usage

### Disable TTS for Specific Sessions
```python
# In your code
import requests
requests.post('http://10.90.19.240:5001/tts/toggle', 
              json={'enabled': False})
```

### Custom Announcements
The TTS system can be extended to announce custom messages by calling:
```python
tts_manager.announce("Your custom message here")
```
