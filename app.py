from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import traceback
from model.object_detection import detect_all_objects, get_no_detection_message
from model.target_search import detect_target_object, reset_target_search

app = Flask(__name__)
CORS(app)

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint for live object detection"""
    try:
        print("\n" + "=" * 50)
        print("/detect endpoint called")
        
        # Get image data
        data = request.json
        if not data or 'image' not in data:
            print("No image in request")
            return jsonify({"success": False, "error": "No image data"}), 400
        
        print(f"Image data length: {len(data['image'])} chars")
        
        # Decode base64
        try:
            image_data = base64.b64decode(data['image'])
            print(f"Base64 decoded: {len(image_data)} bytes")
        except Exception as e:
            print(f"Base64 decode failed: {e}")
            return jsonify({"success": False, "error": f"Base64 decode: {e}"}), 400
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        print(f"Numpy array: {len(nparr)} bytes")
        
        # Try to decode as image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("cv2.imdecode FAILED! Trying IMREAD_UNCHANGED...")
            frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
        if frame is None:
            print("ALL decode attempts FAILED!")
            return jsonify({"success": False, "error": "Cannot decode image - invalid format"}), 400
        
        print(f"Frame decoded: shape={frame.shape}")
        
        # Resize for faster processing
        frame = cv2.resize(frame, (320, 240))
        print("Running object detection...")
        
        # Detect objects
        detections = detect_all_objects(frame)
        
        print(f"Detection result: {len(detections) if detections else 0} objects found")
        
        # No objects found
        if not detections or len(detections) == 0:
            no_detection_msg = get_no_detection_message()
            msg = no_detection_msg["message"] if no_detection_msg else "No objects detected"
            print(f"{msg}")
            return jsonify({
                "success": True,
                "detection_found": False,
                "message": msg
            })
        
        # Return best detection
        best = detections[0]
        print(f"BEST: {best['label']} | {best['direction']} | {best['meters']}m")
        print(f"Voice: {best['voice_message'][:80]}...")
        print("=" * 50)
        
        return jsonify({
            "success": True,
            "detection_found": True,
            "primary_detection": best,
            "all_detections": detections
        })
        
    except Exception as e:
        print(f"EXCEPTION: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """Endpoint for target object search"""
    try:
        print("\n" + "=" * 50)
        print("🔍 /search endpoint called")
        
        data = request.json
        if not data:
            print("No data")
            return jsonify({"success": False, "error": "No data"}), 400
            
        image_data = base64.b64decode(data['image'])
        target_name = data.get('target', '')
        
        print(f"Target: {target_name}")
        print(f"Image size: {len(image_data)} bytes")
        
        if not target_name:
            return jsonify({"success": False, "error": "No target specified"}), 400
        
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Frame decode failed")
            return jsonify({"success": False, "error": "Invalid image"}), 400
        
        print(f"Frame decoded: {frame.shape}")
        frame = cv2.resize(frame, (320, 240))
        
        result = detect_target_object(frame, target_name)
        
        print(f"Result: {result.get('status', 'unknown')}")
        if result.get('status') == 'found':
            print(f"🎯 Found: {result.get('label')} at {result.get('meters')}m")
        print("=" * 50)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/reset_search', methods=['POST'])
def reset_search():
    """Reset target search state"""
    reset_target_search()
    return jsonify({"success": True, "message": "Search reset"})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "AI Vision Aid Backend is running"})


if __name__ == '__main__':
    print("=" * 50)
    print("AI Vision Aid Backend Server (Flask)")
    print("=" * 50)
    print("Starting server on http://0.0.0.0:5000")
    print("Endpoints:")
    print("   POST /detect - Live object detection")
    print("   POST /search - Target object search")
    print("   POST /reset_search - Reset search")
    print("   GET  /health - Health check")
    print("=" * 50)
    print("")
    print(" DEBUG MODE: ON - Check this terminal for logs")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)