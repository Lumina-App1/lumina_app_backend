from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import traceback
from model.object_detection import detect_all_objects, get_no_detection_message
from model.target_search import detect_target_object, reset_target_search
from distance_direction.utils import reset_guidance_tracker

app = Flask(__name__)
CORS(app)


def decode_image(b64_string):
    """
    Decode a base64 image string to a cv2 frame.
    Returns (frame, error_message).
    frame is None if decoding failed.
    """
    try:
        image_data = base64.b64decode(b64_string)
    except Exception as e:
        return None, f"Base64 decode failed: {e}"

    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        frame = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if frame is None:
        return None, "Cannot decode image — invalid format or empty data"

    return frame, None


@app.route('/detect', methods=['POST'])
def detect():
    """Live object detection endpoint."""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No image data"}), 400

        frame, err = decode_image(data['image'])
        if frame is None:
            return jsonify({"success": False, "error": err}), 400

        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))

        detections = detect_all_objects(frame)

        # No speakable detections
        if not detections:
            no_detection_msg = get_no_detection_message()
            msg = no_detection_msg["message"] if no_detection_msg else "No new objects to announce"
            return jsonify({
                "success": True,
                "detection_found": False,
                "message": msg
            })

        best = detections[0]

        return jsonify({
            "success": True,
            "detection_found": True,
            "primary_detection": best,
            "all_detections": detections
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    """Target object search endpoint."""
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "No data"}), 400

        target_name = data.get('target', '').strip()
        if not target_name:
            return jsonify({"success": False, "error": "No target specified"}), 400

        frame, err = decode_image(data['image'])
        if frame is None:
            return jsonify({"success": False, "error": err}), 400

        frame = cv2.resize(frame, (640, 480))

        result = detect_target_object(frame, target_name)
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/reset_search', methods=['POST'])
def reset_search():
    """Reset target search state."""
    reset_target_search()
    reset_guidance_tracker()  # Also reset guidance cooldowns
    return jsonify({"success": True, "message": "Search reset"})


@app.route('/reset_guidance', methods=['POST'])
def reset_guidance():
    """Reset only the guidance/repetition tracker."""
    reset_guidance_tracker()
    return jsonify({"success": True, "message": "Guidance tracker reset"})


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "AI Vision Aid Backend is running"
    })


if __name__ == '__main__':
    
    import os
    print("=" * 50)
    print("AI Vision Aid Backend (Flask)")
    print("=" * 50)
    print("Endpoints:")
    print("  POST /detect          - Live object detection")
    print("  POST /search          - Target object search")
    print("  POST /reset_search    - Reset search + guidance")
    print("  POST /reset_guidance  - Reset guidance tracker only")
    print("  GET  /health          - Health check")
    print("=" * 50)

    # app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        threaded=True
    )