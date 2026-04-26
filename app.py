from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from model.object_detection import detect_all_objects, get_no_detection_message
from model.target_search import detect_target_object, reset_target_search

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

@app.route('/detect', methods=['POST'])
def detect():
    """Endpoint for live object detection"""
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400
        
        # Detect objects
        detections = detect_all_objects(frame)
        
        # No objects found
        if not detections or len(detections) == 0:
            no_detection_msg = get_no_detection_message()
            return jsonify({
                "success": True,
                "detection_found": False,
                "message": no_detection_msg["message"] if no_detection_msg else "No objects detected"
            })
        
        # Return best detection
        best = detections[0]
        return jsonify({
            "success": True,
            "detection_found": True,
            "primary_detection": best,
            "all_detections": detections
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """Endpoint for target object search"""
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        target_name = data.get('target', '')
        
        if not target_name:
            return jsonify({"success": False, "error": "No target specified"}), 400
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400
        
        # Search for target
        result = detect_target_object(frame, target_name)
        
        return jsonify(result)
        
    except Exception as e:
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
    print("AI Vision Aid Backend Server")
    print("=" * 50)
    print("Starting server on http://0.0.0.0:5000")
    print("Endpoints:")
    print("  POST /detect - Live object detection")
    print("  POST /search - Target object search")
    print("  GET  /health - Health check")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)