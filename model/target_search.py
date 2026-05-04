import cv2
import time
import re
from ultralytics import YOLO
from distance_direction.utils import get_full_guidance


def is_match(label, target):
    label = label.lower().strip()
    target = target.lower().strip()

    synonyms = {
        "person": ["person", "human", "man", "woman", "people"],

        "bicycle": ["bicycle", "bike", "cycle"],
        "car": ["car", "automobile", "vehicle"],
        "motorcycle": ["motorcycle", "bike", "motorbike"],
        "airplane": ["airplane", "plane", "aircraft"],
        "bus": ["bus", "coach"],
        "train": ["train", "locomotive"],
        "truck": ["truck", "lorry"],
        "boat": ["boat", "ship", "vessel"],

        "traffic light": ["traffic light", "signal", "traffic signal"],
        "fire hydrant": ["fire hydrant"],
        "stop sign": ["stop sign"],
        "parking meter": ["parking meter"],
        "bench": ["bench", "seat"],

        "bird": ["bird"],
        "cat": ["cat", "kitten"],
        "dog": ["dog", "puppy"],
        "horse": ["horse", "stallion"],
        "sheep": ["sheep"],
        "cow": ["cow", "cattle"],
        "elephant": ["elephant"],
        "bear": ["bear"],
        "zebra": ["zebra"],
        "giraffe": ["giraffe"],

        "backpack": ["backpack", "bag", "rucksack"],
        "umbrella": ["umbrella"],
        "handbag": ["handbag", "purse", "bag"],
        "tie": ["tie", "necktie"],
        "suitcase": ["suitcase", "luggage", "bag"],

        "frisbee": ["frisbee"],
        "skis": ["skis"],
        "snowboard": ["snowboard"],
        "sports ball": ["sports ball", "ball", "football", "soccer ball", "basketball"],
        "kite": ["kite"],
        "baseball bat": ["baseball bat", "bat"],
        "baseball glove": ["baseball glove", "glove"],
        "skateboard": ["skateboard"],
        "surfboard": ["surfboard"],
        "tennis racket": ["tennis racket", "racket"],

        "bottle": ["bottle", "water bottle"],
        "wine glass": ["wine glass", "glass"],
        "cup": ["cup", "mug", "glass"],
        "fork": ["fork"],
        "knife": ["knife", "blade"],
        "spoon": ["spoon"],
        "bowl": ["bowl", "dish"],

        "banana": ["banana"],
        "apple": ["apple"],
        "sandwich": ["sandwich"],
        "orange": ["orange", "fruit"],
        "broccoli": ["broccoli"],
        "carrot": ["carrot"],
        "hot dog": ["hot dog"],
        "pizza": ["pizza"],
        "donut": ["donut", "doughnut"],
        "cake": ["cake"],

        "chair": ["chair", "seat"],
        "couch": ["couch", "sofa", "settee"],
        "potted plant": ["potted plant", "plant"],
        "bed": ["bed", "mattress"],
        "dining table": ["dining table", "table", "desk"],
        "toilet": ["toilet", "wc"],

        "tv": ["tv", "television", "monitor", "screen"],
        "laptop": ["laptop", "notebook", "computer"],
        "mouse": ["mouse", "computer mouse"],
        "remote": ["remote", "remote control"],
        "keyboard": ["keyboard"],
        "cell phone": ["cell phone", "phone", "mobile", "smartphone"],

        "microwave": ["microwave", "microwave oven"],
        "oven": ["oven"],
        "toaster": ["toaster"],
        "sink": ["sink", "basin"],
        "refrigerator": ["refrigerator", "fridge"],

        "book": ["book"],
        "clock": ["clock", "watch"],
        "vase": ["vase"],
        "scissors": ["scissors", "scissor"],
        "teddy bear": ["teddy bear", "teddy"],
        "hair drier": ["hair drier", "hair dryer", "dryer"],
        "toothbrush": ["toothbrush", "brush"]
    }

    allowed = synonyms.get(target, [target])

    # ✅ Flexible matching (FIXED)
    for word in allowed:
        if word in label or label in word:
            return True

    return False


# Model
model = YOLO("yolov8s.pt")
THRESHOLD = 0.35

searching = False
current_target = None
last_search_announcement = 0


#############################################
# Normalize spoken target names
#############################################

def normalize_target(target):
    #  clean unwanted characters
    target = re.sub(r'[^a-zA-Z\s]', '', target)
    target = target.lower().strip()

    fillers = ["my", "the", "a", "an", "find", "search", "for", "look", "located"]
    words = target.split()

    cleaned_target = " ".join([w for w in words if w not in fillers])

    aliases = {
       # --- PERSON & ACCESSORIES ---
        "person": "person", "human": "person", "man": "person", "woman": "person", "people": "person", "guy": "person", "girl": "person", "boy": "person",
        "bicycle": "bicycle", "bike": "bicycle", "cycle": "bicycle", "cycle": "bicycle",
        "car": "car", "automobile": "car", "vehicle": "car", "sedan": "car", "suv": "car",
        "motorcycle": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
        "airplane": "airplane", "plane": "airplane", "aircraft": "airplane", "jet": "airplane",
        "bus": "bus", "coach": "bus", "shuttle": "bus",
        "train": "train", "locomotive": "train", "subway": "train", "metro": "train",
        "truck": "truck", "lorry": "truck", "semi": "truck", "pickup": "truck",
        "boat": "boat", "ship": "boat", "vessel": "boat", "yacht": "boat", "canoe": "boat",
        
        # --- OUTDOOR/TRAFFIC ---
        "traffic light": "traffic light", "signal": "traffic light", "stoplight": "traffic light",
        "fire hydrant": "fire hydrant", "hydrant": "fire hydrant",
        "stop sign": "stop sign",
        "parking meter": "parking meter",
        "bench": "bench", "seat": "bench", "pew": "bench",

        # --- ANIMALS ---
        "bird": "bird", "sparrow": "bird", "pigeon": "bird", "parrot": "bird",
        "cat": "cat", "kitten": "cat", "kitty": "cat", "feline": "cat",
        "dog": "dog", "puppy": "dog", "pooch": "dog", "canine": "dog",
        "horse": "horse", "pony": "horse", "stallion": "horse",
        "sheep": "sheep", "lamb": "sheep", "ram": "sheep",
        "cow": "cow", "cattle": "cow", "bull": "cow", "ox": "cow",
        "elephant": "elephant",
        "bear": "bear",
        "zebra": "zebra",
        "giraffe": "giraffe",

        # --- PERSONAL ITEMS ---
        "backpack": "backpack", "bag": "backpack", "rucksack": "backpack", "knapsack": "backpack",
        "umbrella": "umbrella", "parasol": "umbrella",
        "handbag": "handbag", "purse": "handbag", "pocketbook": "handbag",
        "tie": "tie", "necktie": "tie",
        "suitcase": "suitcase", "luggage": "suitcase", "baggage": "suitcase", "trunk": "suitcase",

        # --- SPORTS ---
        "frisbee": "frisbee", "flying disc": "frisbee",
        "skis": "skis", "ski": "skis",
        "snowboard": "snowboard",
        "sports ball": "sports ball", "ball": "sports ball", "football": "sports ball", "soccer ball": "sports ball", "basketball": "sports ball",
        "kite": "kite",
        "baseball bat": "baseball bat", "bat": "baseball bat",
        "baseball glove": "baseball glove", "mitt": "baseball glove",
        "skateboard": "skateboard",
        "surfboard": "surfboard",
        "tennis racket": "tennis racket", "racket": "tennis racket", "racquet": "tennis racket",

        # --- KITCHEN & DINING ---
        "bottle": "bottle", "water bottle": "bottle", "container": "bottle", "flask": "bottle",
        "wine glass": "wine glass", "stemware": "wine glass",
        "cup": "cup", "mug": "cup", "tea cup": "cup", "coffee cup": "cup",
        "fork": "fork",
        "knife": "knife", "blade": "knife", "steak knife": "knife",
        "spoon": "spoon", "teaspoon": "spoon",
        "bowl": "bowl", "dish": "bowl", "basin": "bowl",
        "banana": "banana",
        "apple": "apple",
        "sandwich": "sandwich", "sub": "sandwich", "burger": "sandwich",
        "orange": "orange", "citrus": "orange",
        "broccoli": "broccoli",
        "carrot": "carrot",
        "hot dog": "hot dog",
        "pizza": "pizza",
        "donut": "donut", "doughnut": "donut",
        "cake": "cake", "pastry": "cake",

        # --- INDOOR/FURNITURE ---
        "chair": "chair", "stool": "chair", "armchair": "chair",
        "couch": "couch", "sofa": "couch", "loveseat": "couch", "settee": "couch",
        "potted plant": "potted plant", "plant": "potted plant", "flower": "potted plant",
        "bed": "bed", "mattress": "bed", "cot": "bed",
        "dining table": "dining table", "table": "dining table", "desk": "dining table", "workbench": "dining table",
        "toilet": "toilet", "commode": "toilet", "loo": "toilet", "john": "toilet",

        # --- ELECTRONICS ---
        "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv", "display": "tv",
        "laptop": "laptop", "computer": "laptop", "notebook": "laptop", "macbook": "laptop",
        "mouse": "mouse", "computer mouse": "mouse",
        "remote": "remote", "remote control": "remote", "clicker": "remote",
        "keyboard": "keyboard", "keypad": "keyboard",
        "cell phone": "cell phone", "phone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone", "iphone": "cell phone",

        # --- APPLIANCES ---
        "microwave": "microwave", "microwave oven": "microwave",
        "oven": "oven", "stove": "oven", "range": "oven",
        "toaster": "toaster",
        "sink": "sink", "washbasin": "sink", "faucet": "sink",
        "refrigerator": "refrigerator", "fridge": "refrigerator", "freezer": "refrigerator",

        # --- MISC ---
        "book": "book", "novel": "book", "textbook": "book",
        "clock": "clock", "watch": "clock", "timer": "clock", "wall clock": "clock",
        "vase": "vase", "flowerpot": "vase",
        "scissors": "scissors", "shears": "scissors", "clipper": "scissors",
        "teddy bear": "teddy bear", "stuffed animal": "teddy bear", "teddy": "teddy bear",
        "hair drier": "hair drier", "hair dryer": "hair drier", "blow dryer": "hair drier",
        "toothbrush": "toothbrush"
    }

    #  Try full phrase
    if cleaned_target in aliases:
        return aliases[cleaned_target]

    #  Try word fallback
    for word in cleaned_target.split():
        if word in aliases:
            return aliases[word]

    return cleaned_target


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

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])

            #  FIX: normalize YOLO label
            label = model.names[cls].lower()
            label = normalize_target(label)

            conf = float(box.conf[0])

            #  Debug
            print("TARGET:", target_name)
            print("YOLO:", label, conf)

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

    if best_match is None:

        now = time.time()

        if now - last_search_announcement > 2:
            last_search_announcement = now

            return {
                "status": "not_found",
                "voice_message":
                # f"{target_name} not found. Move camera slowly left and right."
                f"{target_name} not found."
            }

        else:
            return {
                "status": "searching",
                "voice_message": None
            }

    guidance = get_full_guidance(
        best_match["box"],
        width,
        height,
        best_match["label"],
        best_match["confidence"]
    )

    if guidance is None:
        return {
            "status": "found",
            "voice_message": f"{target_name} found ahead"
        }

    if guidance["meters"] < 0.8:
        voice_message = (
            f"{target_name} found. {guidance['message']} You can reach out now."
        )
    else:
        voice_message = (
            f"{target_name} found. {guidance['message']}"
        )

    return {
        "status": "found",
        "label": best_match["label"],
        "confidence": best_match["confidence"],
        "direction": guidance["direction"],
        "guidance": guidance["guidance"],
        "distance": guidance["distance"],
        "meters": guidance["meters"],
        "warning": guidance["warning"],
        "voice message": guidance["message"]
    }


#############################################
# Reset Search
#############################################

def reset_target_search():
    global searching
    global current_target

    searching = False
    current_target = None