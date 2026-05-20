def estimate_direction(box, img_width):
    """
    Estimate direction with 5 levels: far_left, left, center, right, far_right

    box = [x1, y1, x2, y2]
    img_width = width of frame

    Returns direction description only — no navigation commands.
    Navigation is handled by utils.py based on distance.
    """
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2

    # Normalized position: -1 = far left, 0 = center, +1 = far right
    normalized_pos = (x_center - (img_width / 2)) / (img_width / 2)

    if normalized_pos < -0.6:
        return {
            "direction": "far left",
            "guidance": "on your far left",       # describes position only
            "short_guidance": "far left",
            "normalized_pos": normalized_pos
        }
    elif normalized_pos < -0.25:
        return {
            "direction": "left",
            "guidance": "on your left",            # describes position only
            "short_guidance": "left",
            "normalized_pos": normalized_pos
        }
    elif normalized_pos < 0.25:
        return {
            "direction": "center",
            "guidance": "straight ahead",          # describes position only
            "short_guidance": "center",
            "normalized_pos": normalized_pos
        }
    elif normalized_pos < 0.6:
        return {
            "direction": "right",
            "guidance": "on your right",           # describes position only
            "short_guidance": "right",
            "normalized_pos": normalized_pos
        }
    else:
        return {
            "direction": "far right",
            "guidance": "on your far right",       # describes position only
            "short_guidance": "far right",
            "normalized_pos": normalized_pos
        }