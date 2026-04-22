from ultralytics import YOLO
import cv2

# Load model once
model = YOLO("yolov8n.pt")

THRESHOLD = 0.4

def detect_all_objects(img):
    # Resize for performance
    img = cv2.resize(img, (640, 480))

    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            coords = box.xyxy[0].tolist()

            if conf > THRESHOLD:
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": coords
                })

    # Sort by importance
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)

    # Limit to top 5 objects (better for audio output)
    return detections[:5]