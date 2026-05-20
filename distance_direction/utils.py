import time
from distance_direction.direction import estimate_direction
from distance_direction.distance import estimate_distance

# Track last announcement per label to prevent repetition
# Key: label, Value: {direction, distance_label, last_time}
last_announcements = {}

REPEAT_COOLDOWN_SECONDS = 2.5   # same object, same position — skip
MOVEMENT_COOLDOWN_SECONDS = 0.8  # object moved — shorter cooldown


def get_full_guidance(box, img_width, img_height, label, confidence):
    """
    Complete guidance system for visually impaired users.

    voice_message structure (clean, no contradictions):
        "{label} is {direction_phrase}, {distance_phrase}. {action}."

    Examples:
        "Laptop is on your right, about 1 meter away. Stop."
        "Person is straight ahead, about 2 meters away. Move forward slowly."
        "Car is on your far left, about 3 to 4 meters away. Keep moving."
    """
    global last_announcements

    direction_data = estimate_direction(box, img_width)
    distance_data  = estimate_distance(box, img_width, label)

    direction = direction_data["direction"]          # e.g. "left"
    guidance  = direction_data["guidance"]           # e.g. "on your left"
    meters    = distance_data["meters"]
    dist_desc = _distance_phrase(meters)             # e.g. "about 1 meter away"

    # ── Action based on distance only (not mixed with direction) ──────
    if meters < 0.8:
        action = "STOP"
        action_message = "Stop."
    elif meters < 1.5:
        action = "SLOW"
        action_message = "Move forward slowly."
    elif meters < 3.0:
        action = "MOVE"
        action_message = "Continue moving."
    elif meters < 5.0:
        action = "MOVE"
        action_message = "Keep moving."
    else:
        action = "MOVE"
        action_message = "Keep moving."

    # ── Clean voice message: object + where it IS + how far + what to do ──
    # No navigation commands ("move left/right") mixed in — just description.
    voice_message = f"{label} is {guidance}, {dist_desc}. {action_message}"

    # Example outputs:
    #   "Laptop is on your right, about 1 meter away. Move forward slowly."
    #   "Person is straight ahead, about 2 meters away. Continue moving."
    #   "Car is on your far left, about 5 meters away. Keep moving."
    #   "Bottle is straight ahead, right in front of you. Stop."

    # ── Repetition filter ─────────────────────────────────────────────
    current_time = time.time()
    prev = last_announcements.get(label)

    if prev is not None:
        same_position = (
            prev["direction"]      == direction and
            prev["distance_label"] == distance_data["distance_label"]
        )
        time_since = current_time - prev["last_time"]

        if same_position and time_since < REPEAT_COOLDOWN_SECONDS:
            return None   # Too soon, same position
        if not same_position and time_since < MOVEMENT_COOLDOWN_SECONDS:
            return None   # Moved but still too soon

    # Update tracker
    last_announcements[label] = {
        "direction":      direction,
        "distance_label": distance_data["distance_label"],
        "last_time":      current_time,
    }

    return {
        "direction":      direction,
        "guidance":       guidance,
        "short_guidance": direction_data["short_guidance"],
        "distance":       distance_data["distance_label"],
        "meters":         meters,
        "warning":        distance_data["warning"],
        "instruction":    distance_data["instruction"],
        "action":         action,
        "action_message": action_message,
        "voice_message":  voice_message,
        "normalized_pos": direction_data["normalized_pos"],
    }


def _distance_phrase(meters: float) -> str:
    """Convert meters to a natural spoken phrase."""
    if meters < 0.8:
        return "right in front of you"
    elif meters < 1.5:
        return "about 1 meter away"
    elif meters < 3.0:
        return "about 2 meters away"
    elif meters < 5.0:
        return "about 3 to 4 meters away"
    elif meters < 8.0:
        return "about 5 meters away"
    else:
        return "about 8 or more meters away"


def reset_guidance_tracker():
    """Call this when switching modes or resetting the app."""
    global last_announcements
    last_announcements = {}