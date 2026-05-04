import cv2
import time
from ultralytics import YOLO
from distance_direction.utils import get_full_guidance

# Load YOLO model once
model = YOLO("yolov8s.pt")

THRESHOLD = 0.45  # Confidence threshold (45%)

# Frame throttling - process every 3rd frame only
frame_counter = 0
PROCESS_EVERY_N_FRAMES = 1

# Track last detection time
last_detection_time = time.time()
NO_DETECTION_TIMEOUT = 2  # seconds




def detect_all_objects(img):
    """
    Optimized detection with frame throttling and non-repetition
    """
    global frame_counter, last_detection_time
    
    frame_counter += 1
    
    # SKIP frames for performance (don't process every frame)
    if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
        return None  # Skip this frame
    
    # Get frame dimensions
    height, width = img.shape[:2]
    
    # Run YOLO detection (no resize - keep original for better accuracy)
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
            
            # Filter low confidence
            if conf < THRESHOLD:
                continue
            
            # Get full guidance (direction + distance + voice message)
            guidance = get_full_guidance(coords, width, height, label, conf)
            
            # Skip if this is a repetition (guidance returns None)
            if guidance is None:
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
                "normalized_pos": guidance["normalized_pos"]
            })
    
    # Update last detection time if objects found
    if detections:
        last_detection_time = time.time()
    
    # Sort by priority (closest objects first, then highest confidence)
    detections.sort(key=lambda x: (x["meters"], -x["confidence"]))
    
    # Return top 3 objects (limit for performance)
    return detections[:3] if detections else []


def get_no_detection_message():
    """Returns message when no objects detected for a while"""
    global last_detection_time
    
    time_since_last = time.time() - last_detection_time
    
    if time_since_last > NO_DETECTION_TIMEOUT:
        return {
            "status": "no_objects",
            "message": "No objects detected. Try moving the camera slowly."
        }
    return None