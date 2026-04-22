from ultralytics import YOLO
import cv2

# Load model once
model = YOLO("yolov8n.pt")

THRESHOLD = 0.4


def get_direction(box, img_width):
    x_center = (box[0] + box[2]) / 2

    if x_center < img_width / 3:
        return "left"
    elif x_center < 2 * img_width / 3:
        return "center"
    else:
        return "right"


def get_distance(box):
    width = box[2] - box[0]

    if width > 200:
        return "near"
    else:
        return "far"


def detect_target_object(img, target_name):
    # Resize image for faster processing
    img = cv2.resize(img, (640, 480))
    height, width, _ = img.shape

    # Run YOLO detection
    results = model(img)

    best_match = None
    max_conf = 0

    # Loop through detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            # Check if this object matches target
            if conf > THRESHOLD and label.lower() == target_name.lower():
                if conf > max_conf:
                    max_conf = conf
                    best_match = {
                        "label": label,
                        "confidence": conf,
                        "box": coords
                    }

    # If object NOT found
    if best_match is None:
        return {
            "status": "not_found",
            "message": f"{target_name} not visible. Move left or right slowly."
        }

    # If object found → calculate direction & distance
    box = best_match["box"]

    direction = get_direction(box, width)
    distance = get_distance(box)

    return {
        "status": "found",
        "label": best_match["label"],
        "confidence": best_match["confidence"],
        "direction": direction,
        "distance": distance
    }