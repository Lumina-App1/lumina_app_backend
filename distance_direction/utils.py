from distance_direction.direction import estimate_direction
from distance_direction.distance import estimate_distance

# Track last announcement to prevent repetition
last_announcement = {
    "object": "",
    "direction": "",
    "distance_label": "",
    "frame_count": 0
}

def get_full_guidance(box, img_width, img_height, label, confidence):
    """
    Complete guidance system for visually impaired users
    
    Returns natural voice instructions for navigation
    """
    global last_announcement
    
    # Calculate direction (5-level)
    direction_data = estimate_direction(box, img_width)
    
    # Calculate distance (fast bounding box method)
    distance_data = estimate_distance(box, img_width, label)
    
    # Determine if user needs to STOP (very close)
    if distance_data["meters"] < 0.8:
        action = "STOP"
        action_message = f"Stop. {label} is very close to you."
    elif distance_data["meters"] < 1.5:
        action = "SLOW"
        action_message = f"Move forward slowly, {label} is nearby."
    else:
        action = "MOVE"
        action_message = "Keep moving forward."
    
    # Build natural voice message for visually impaired user
    if direction_data["direction"] == "center":
        voice_message = (
            f"{label} detected straight ahead, "
            f"{distance_data['warning'].lower().replace('object', label)}. "
            f"{action_message}"
        )
    else:
        voice_message = (
            f"{label} detected on your {direction_data['direction']}, "
            f"{distance_data['warning'].lower().replace('object', label)}. "
            f"{direction_data['guidance']}. "
            f"{action_message}"
        )
    
    # Check for repetition (don't repeat same thing every frame)
    is_repetition = (
        last_announcement["object"] == label and 
        last_announcement["direction"] == direction_data["direction"] and
        last_announcement["distance_label"] == distance_data["distance_label"] and
        last_announcement["frame_count"] < 15  # Wait 15 frames before repeating
    )
    
    if not is_repetition:
        last_announcement = {
            "object": label,
            "direction": direction_data["direction"],
            "distance_label": distance_data["distance_label"],
            "frame_count": 0
        }
    else:
        last_announcement["frame_count"] += 1
        # Return None to indicate no announcement needed
        return None
    
    return {
        "direction": direction_data["direction"],
        "guidance": direction_data["guidance"],
        "short_guidance": direction_data["short_guidance"],
        "distance": distance_data["distance_label"],
        "meters": distance_data["meters"],
        "warning": distance_data["warning"],
        "instruction": distance_data["instruction"],
        "action": action,
        "action_message": action_message,
        "voice_message": voice_message,
        "normalized_pos": direction_data["normalized_pos"]
    }