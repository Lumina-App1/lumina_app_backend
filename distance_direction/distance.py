import math

# Known real-world widths in meters for common objects
KNOWN_WIDTHS = {
    "person": 0.5,
    "car": 1.8,
    "bottle": 0.07,
    "cup": 0.08,
    "laptop": 0.35,
    "cell phone": 0.07,
    "chair": 0.5,
    "dog": 0.3,
    "cat": 0.25,
    "bicycle": 0.6,
    "motorcycle": 0.8,
    "bus": 2.5,
    "truck": 2.5,
    "dining table": 1.0,
    "couch": 1.5,
    "bed": 1.4,
    "tv": 0.9,
    "backpack": 0.3,
    "umbrella": 0.9,
    "handbag": 0.3,
    "suitcase": 0.5,
    "bench": 1.2,
}

FOCAL_LENGTH = 600  # Approximate focal length for mobile cameras

def estimate_distance(box, frame_width, object_label=""):
    """
    Estimate distance using known object widths and focal length formula.
    Returns distance in meters with natural language guidance.
    """
    x1, y1, x2, y2 = box
    bbox_width = x2 - x1

    # Get known width for object, default 0.4m if unknown
    known_width = KNOWN_WIDTHS.get(object_label.lower(), 0.4)

    if bbox_width > 0:
        meters = (known_width * FOCAL_LENGTH) / bbox_width
        meters = round(max(0.3, min(meters, 15.0)), 1)  # clamp between 0.3-15m
    else:
        meters = 5.0

    # Labels based on calculated meters
    if meters < 0.8:
        label = "very close"
        warning = f"Stop! {object_label} is right in front of you"
        instruction = "Reach out carefully"
    elif meters < 1.5:
        label = "close"
        warning = f"{object_label} is about 1 meter away"
        instruction = "Move forward slowly"
    elif meters < 3.0:
        label = "medium"
        warning = f"{object_label} is about 2 meters away"
        instruction = "Continue moving forward"
    elif meters < 5.0:
        label = "far"
        warning = f"{object_label} is about 3-4 meters away"
        instruction = "Walk straight ahead"
    elif meters < 8.0:
        label = "very far"
        warning = f"{object_label} is about 5 meters away"
        instruction = "Keep walking straight"
    else:
        label = "distant"
        warning = f"{object_label} is far away, about 8+ meters"
        instruction = "Move forward to get closer"

    return {
        "distance_label": label,
        "meters": meters,
        "warning": warning,
        "instruction": instruction,
        "bbox_percentage": (bbox_width / frame_width) * 100
    }