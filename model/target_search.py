import cv2
import time
from ultralytics import YOLO
from distance_direction.utils import get_full_guidance

model = YOLO("yolov8s.pt")
THRESHOLD = 0.45

# Target search state
searching = False
current_target = None
last_search_announcement = time.time()


def detect_target_object(img, target_name):
    """
    Search for specific target object with guidance
    """
    global searching, current_target, last_search_announcement
    
    searching = True
    current_target = target_name
    
    height, width = img.shape[:2]
    
    # Run detection
    results = model(img, verbose=False)
    
    best_match = None
    max_conf = 0
    
    for r in results:
        if r.boxes is None:
            continue
            
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            
            # Check if matches target
            if conf > THRESHOLD and label.lower() == target_name.lower():
                if conf > max_conf:
                    max_conf = conf
                    best_match = {
                        "label": label,
                        "confidence": conf,
                        "box": coords
                    }
    
    # Target not found
    if best_match is None:
        current_time = time.time()
        # Don't spam the same message every frame
        if current_time - last_search_announcement > 3:
            last_search_announcement = current_time
            return {
                "status": "not_found",
                "voice_message": f"{target_name} not found. Slowly move the camera left and right."
            }
        return {
            "status": "not_found",
            "voice_message": None
        }
    
    # Target found - get guidance
    box = best_match["box"]
    guidance = get_full_guidance(box, width, height, best_match["label"], best_match["confidence"])
    
    if guidance is None:
        return {
            "status": "found",
            "voice_message": None
        }
    
    # Enhanced found message for target
    if guidance["meters"] < 0.8:
        final_message = f"Target found! {guidance['voice_message']} You can reach out now."
    else:
        final_message = f"Target found! {guidance['voice_message']}"
    
    return {
        "status": "found",
        "label": best_match["label"],
        "confidence": best_match["confidence"],
        "direction": guidance["direction"],
        "guidance": guidance["guidance"],
        "distance": guidance["distance"],
        "meters": guidance["meters"],
        "warning": guidance["warning"],
        "voice_message": final_message
    }


def reset_target_search():
    """Reset target search state"""
    global searching, current_target
    searching = False
    current_target = None