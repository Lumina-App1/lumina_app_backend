from distance_direction.direction import estimate_direction
from distance_direction.distance import estimate_distance
import time

# Track last announcement per object label to prevent repetition
# Key: label, Value: {direction, distance_label, last_time}
last_announcements = {}

# Cooldown in seconds — same object + same direction + same distance
# will be skipped if announced within this many seconds
REPEAT_COOLDOWN_SECONDS = 2.5

# If object MOVES (direction/distance changes), announce immediately
# regardless of cooldown
MOVEMENT_COOLDOWN_SECONDS = 0.8


def get_full_guidance(box, img_width, img_height, label, confidence):
    """
    Complete guidance system for visually impaired users.
    Returns natural voice instructions for navigation.
    Uses TIME-based cooldown instead of frame-count (more reliable).
    """
    global last_announcements

    # Calculate direction (5-level)
    direction_data = estimate_direction(box, img_width)

    # Calculate distance (known width method)
    distance_data = estimate_distance(box, img_width, label)

    # Determine action based on distance
    meters = distance_data["meters"]
    if meters < 0.8:
        action = "STOP"
        action_message = f"Stop. {label} is very close to you."
    elif meters < 1.5:
        action = "SLOW"
        action_message = f"Move forward slowly, {label} is nearby."
    else:
        action = "MOVE"
        action_message = "Keep moving forward."

    # Build natural voice message
    direction = direction_data["direction"]
    warning_text = distance_data["warning"].lower().replace("object", label)

    if direction == "center":
        voice_message = (
            f"{label} detected straight ahead, "
            f"{warning_text}. "
            f"{action_message}"
        )
        message = (
            f"{label} detected straight ahead, "
            f"{warning_text}. "
        )
    else:
        voice_message = (
            f"{label} detected on your {direction}, "
            f"{warning_text}. "
            f"{direction_data['guidance']}. "
            f"{action_message}"
        )
        message = (
            f"{label} detected on your {direction}, "
            f"{warning_text}. "
            f"{direction_data['guidance']}. "
        )

    # --- TIME-BASED REPETITION FILTER ---
    current_time = time.time()
    prev = last_announcements.get(label)

    if prev is not None:
        same_position = (
            prev["direction"] == direction and
            prev["distance_label"] == distance_data["distance_label"]
        )

        if same_position:
            # Same position — apply full cooldown
            time_since = current_time - prev["last_time"]
            if time_since < REPEAT_COOLDOWN_SECONDS:
                return None  # Skip, too soon
        else:
            # Object moved — apply shorter cooldown
            time_since = current_time - prev["last_time"]
            if time_since < MOVEMENT_COOLDOWN_SECONDS:
                return None  # Still a tiny bit too soon

    # Update tracking
    last_announcements[label] = {
        "direction": direction,
        "distance_label": distance_data["distance_label"],
        "last_time": current_time
    }

    return {
        "direction": direction,
        "guidance": direction_data["guidance"],
        "short_guidance": direction_data["short_guidance"],
        "distance": distance_data["distance_label"],
        "meters": meters,
        "warning": distance_data["warning"],
        "instruction": distance_data["instruction"],
        "action": action,
        "action_message": action_message,
        "voice_message": voice_message,
        "message": message,
        "normalized_pos": direction_data["normalized_pos"]
    }


def reset_guidance_tracker():
    """Call this when switching modes or resetting the app."""
    global last_announcements
    last_announcements = {}