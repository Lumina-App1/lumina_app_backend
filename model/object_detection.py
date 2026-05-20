import cv2
import time
from ultralytics import YOLO
from distance_direction.utils import get_full_guidance

# Load YOLO model once at startup
model = YOLO("yolov8m.pt")

# Confidence threshold — objects below this are ignored
THRESHOLD = 0.40

# Track last detection time for no-detection messages
last_detection_time = time.time()
NO_DETECTION_TIMEOUT = 2  # seconds


def detect_all_objects(img):
    """
    Detect all objects in frame.
    Returns list of detections sorted by distance (closest first).
    Returns empty list if nothing found.
    Returns None only on frame skip (not used currently).
    """
    global last_detection_time

    # Get frame dimensions
    height, width = img.shape[:2]

    # Run YOLO
    results = model(img, verbose=False)

    detections = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            # Skip low confidence detections
            if conf < THRESHOLD:
                continue

            # Get guidance (direction + distance + voice)
            guidance = get_full_guidance(coords, width, height, label, conf)

            # guidance is None = repetition cooldown active, skip voice
            # but still count as a detection for last_detection_time
            if guidance is None:
                # Still a valid detection, just dont announce it
                # Add minimal entry so detection_found stays True
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": coords,
                    "direction": "unknown",
                    "short_guidance": "",
                    "guidance": "",
                    "distance": "",
                    "meters": 999,
                    "warning": "",
                    "instruction": "",
                    "action": "",
                    "action_message": "",
                    "voice_message": "",  # Empty = no announcement needed
                    "normalized_pos": [0.5, 0.5],
                    "silent": True  # Flag: detected but dont speak
                })
                continue

            detections.append({
                "label": label,
                "confidence": conf,
                "box": coords,
                "direction": guidance["direction"],
                "short_guidance": guidance["short_guidance"],
                "guidance": guidance["guidance"],
                "distance": guidance["distance"],
                "meters": guidance["meters"],
                "warning": guidance["warning"],
                "instruction": guidance["instruction"],
                "action": guidance["action"],
                "action_message": guidance["action_message"],
                "voice_message": guidance["voice_message"],
                "normalized_pos": guidance["normalized_pos"],
                "silent": False
            })

    # Update last detection time if any objects found (silent or not)
    if detections:
        last_detection_time = time.time()

    # Filter to only non-silent for sorting and returning top results
    speakable = [d for d in detections if not d.get("silent", False)]

    # Sort by distance (closest first), then confidence
    speakable.sort(key=lambda x: (x["meters"], -x["confidence"]))

    # Return top 3 speakable detections
    return speakable[:3] if speakable else []


def get_no_detection_message():
    """
    Returns a message when no objects detected for a while.
    Returns None if objects were recently detected.
    """
    global last_detection_time

    time_since_last = time.time() - last_detection_time

    if time_since_last > NO_DETECTION_TIMEOUT:
        return {
            "status": "no_objects",
            "message": "No objects detected. Try moving the camera slowly."
        }
    return None