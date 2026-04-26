import math

def estimate_distance(box, frame_width, object_label=""):
    """
    Estimate distance using bounding box size (No MiDaS - much faster!)
    
    Returns distance in meters with natural language guidance
    """
    x1, y1, x2, y2 = box
    bbox_width = x2 - x1
    
    # Calculate what percentage of frame the object takes
    bbox_percentage = (bbox_width / frame_width) * 100
    
    # Realistic distance estimation for assistive navigation
    # These values work well for most common objects
    if bbox_percentage > 35:
        meters = 0.5
        label = "very close"
        warning = "Stop! Object is right in front of you"
        instruction = "Reach out carefully"
    elif bbox_percentage > 20:
        meters = 1.0
        label = "close"
        warning = "Object is about 1 meter away"
        instruction = "Move forward slowly"
    elif bbox_percentage > 12:
        meters = 2.0
        label = "medium"
        warning = "Object is about 2 meters away"
        instruction = "Continue moving forward"
    elif bbox_percentage > 7:
        meters = 3.5
        label = "far"
        warning = "Object is about 3-4 meters away"
        instruction = "Walk straight ahead"
    elif bbox_percentage > 4:
        meters = 5.0
        label = "very far"
        warning = "Object is about 5 meters away"
        instruction = "Keep walking straight"
    else:
        meters = 8.0
        label = "distant"
        warning = "Object is far away, about 8+ meters"
        instruction = "Move forward to get closer"
    
    return {
        "distance_label": label,
        "meters": meters,
        "warning": warning,
        "instruction": instruction,
        "bbox_percentage": bbox_percentage
    }