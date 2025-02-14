import mediapipe as mp
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Pinch thresholds
pinch_threshold = 0.05

def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

def is_pinch(lm1, lm2, landmarks):
    # Calculate distance between two landmarks
    distance = math.hypot(
        landmarks[lm1].x - landmarks[lm2].x,
        landmarks[lm1].y - landmarks[lm2].y
    )
    return distance < pinch_threshold

# Thinking on using a fist as stop gesture but idk if I will ever use it
def is_fist(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    palm_base = landmarks.landmark[mp_hands.HandLandmark.WRIST]

    threshold = 0.3
    
    return (
        calculate_distance(thumb_tip, palm_base) < threshold and
        calculate_distance(index_tip, palm_base) < threshold and
        calculate_distance(middle_tip, palm_base) < threshold and
        calculate_distance(ring_tip, palm_base) < threshold and
        calculate_distance(pinky_tip, palm_base) < threshold
    )