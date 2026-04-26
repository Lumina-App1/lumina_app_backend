def estimate_direction(box, img_width):
    """
    Estimate direction with 5 levels: far_left, left, center, right, far_right
    
    box = [x1, y1, x2, y2]
    img_width = width of frame
    
    Returns detailed guidance for visually impaired users
    """
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    
    # Calculate normalized position (-1 to 1)
    # -1 = far left, 0 = center, 1 = far right
    normalized_pos = (x_center - (img_width / 2)) / (img_width / 2)
    
    # 5-level direction with proper guidance
    if normalized_pos < -0.6:
        return {
            "direction": "far left",
            "guidance": "Move significantly to your left",
            "short_guidance": "far left",
            "normalized_pos": normalized_pos
        }
    elif normalized_pos < -0.25:
        return {
            "direction": "left",
            "guidance": "Move left",
            "short_guidance": "left",
            "normalized_pos": normalized_pos
        }
    elif normalized_pos < 0.25:
        return {
            "direction": "center",
            "guidance": "Straight ahead, move forward carefully",
            "short_guidance": "center",
            "normalized_pos": normalized_pos
        }
    elif normalized_pos < 0.6:
        return {
            "direction": "right",
            "guidance": "Move right",
            "short_guidance": "right",
            "normalized_pos": normalized_pos
        }
    else:
        return {
            "direction": "far right",
            "guidance": "Move significantly to your right",
            "short_guidance": "far right",
            "normalized_pos": normalized_pos
        }