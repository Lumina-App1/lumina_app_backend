"""
FASTAPI BACKEND - Faster than Flask
Same functionality, less delay
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from model.object_detection import detect_all_objects, get_no_detection_message
from model.target_search import detect_target_object, reset_target_search

app = FastAPI()

# Enable CORS for Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(data: dict):
    """Live object detection endpoint"""
    try:
        # Get image from request
        image_data = base64.b64decode(data['image'])
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Invalid image"}
        
        # Resize for faster processing (320x240)
        frame = cv2.resize(frame, (320, 240))
        
        # Detect objects
        detections = detect_all_objects(frame)
        
        # No objects found
        if not detections or len(detections) == 0:
            no_detection_msg = get_no_detection_message()
            return {
                "success": True,
                "detection_found": False,
                "message": no_detection_msg["message"] if no_detection_msg else "No objects detected"
            }
        
        # Return best detection
        best = detections[0]
        return {
            "success": True,
            "detection_found": True,
            "primary_detection": best,
            "all_detections": detections
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/search")
async def search(data: dict):
    """Target object search endpoint"""
    try:
        image_data = base64.b64decode(data['image'])
        target_name = data.get('target', '')
        
        if not target_name:
            return {"success": False, "error": "No target specified"}
        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"success": False, "error": "Invalid image"}
        
        # Resize for speed
        frame = cv2.resize(frame, (320, 240))
        
        # Search for target
        result = detect_target_object(frame, target_name)
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/reset_search")
async def reset_search_endpoint():
    """Reset target search state"""
    reset_target_search()
    return {"success": True, "message": "Search reset"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Vision Aid Backend is running (FastAPI)",
        "server": "fastapi"
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🤖 AI Vision Aid Backend (FASTAPI - Faster!)")
    print("=" * 60)
    print("\n✅ Server starting on http://0.0.0.0:5000")
    print("\n📡 Available Endpoints:")
    print("   POST /detect        - Live object detection")
    print("   POST /search        - Target object search")
    print("   POST /reset_search  - Reset search state")
    print("   GET  /health        - Health check")
    print("\n⚡ FASTER than Flask!")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)