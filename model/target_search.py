import cv2
import time
from ultralytics import YOLO
from distance_direction.utils import get_full_guidance

def is_match(label, target):

    label = label.lower()
    target = target.lower()

    synonyms = {
        "laptop": ["laptop", "notebook", "computer"],
        # "mouse": ["mouse", "computer mouse"],
        # "keyboard": ["keyboard"],
        "phone": ["phone", "cell phone", "mobile"],
        "bottle": ["bottle", "water bottle"],
        "cup": ["cup", "glass"],
        "person": ["person", "human"]
    }

    allowed = synonyms.get(target, [target])

    return any(a in label for a in allowed)

# Better model (optional but recommended)
model = YOLO("yolov8s.pt")
THRESHOLD = 0.35

searching = False
current_target = None
last_search_announcement = 0


#############################################
# Normalize spoken target names
#############################################

def normalize_target(target):

    target = target.lower().strip()

    aliases = {

        # Laptop aliases
        "my laptop":"laptop",
        "computer":"laptop",
        "notebook":"laptop",
        "laptop computer":"laptop",

        # Phones
        "phone":"cell phone",
        "mobile":"cell phone",
        "mobile phone":"cell phone",

        # Common objects
        "water bottle":"bottle",
        "glass":"cup",
        "human":"person",

        # Add more if needed
    }

    return aliases.get(target, target)


#############################################
# Target Search Detection
#############################################

def detect_target_object(img, target_name):

    global searching
    global current_target
    global last_search_announcement

    searching = True

    target_name = normalize_target(target_name)
    current_target = target_name

    height, width = img.shape[:2]

    results = model(img, verbose=False)

    best_match = None
    max_conf = 0


    #########################################
    # Find best matching target
    #########################################

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])
            label = model.names[cls].lower()

            conf = float(box.conf[0])

            # Debug print in terminal
            print("Detected:", label, conf)

            if (
            conf > THRESHOLD and
            is_match(label, target_name) and
            conf > max_conf
            ):
                max_conf = conf

                best_match = {
                    "label": label,
                    "confidence": conf,
                    "box": box.xyxy[0].tolist()
                }


    #########################################
    # If target NOT found
    #########################################

    if best_match is None:

        now = time.time()

        if now - last_search_announcement > 2:
            last_search_announcement = now

            return {
                "status":"not_found",

                "voice_message":
                    f"{target_name} not found. "
                    "Move camera slowly left and right."
            }

        else:
            return {
                "status":"searching",
                "voice_message":None
            }



    #########################################
    # If target FOUND
    #########################################

    guidance = get_full_guidance(
        best_match["box"],
        width,
        height,
        best_match["label"],
        best_match["confidence"]
    )


    if guidance is None:
        return {
            "status":"found",
            "voice_message":"Target found ahead"
        }


    if guidance["meters"] < 0.8:

        voice_message = (
            f"Target found. "
            f"{guidance['voice_message']} "
            f"You can reach out now."
        )

    else:

        voice_message = (
            f"Target found. "
            f"{guidance['voice_message']}"
        )


    return {

        "status":"found",

        "label":best_match["label"],

        "confidence":best_match["confidence"],

        "direction":guidance["direction"],

        "guidance":guidance["guidance"],

        "distance":guidance["distance"],

        "meters":guidance["meters"],

        "warning":guidance["warning"],

        "voice_message":voice_message
    }



#############################################
# Reset Search
#############################################

def reset_target_search():

    global searching
    global current_target

    searching = False
    current_target = None