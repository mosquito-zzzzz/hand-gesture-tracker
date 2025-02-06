import cv2
import mediapipe as mp
import time
import gestures as gt
import warnings
warnings.filterwarnings("ignore", message=".NORM_RECT without IMAGE_DIMENSIONS.")

# Optimize the performance
cv2.setUseOptimized(True)
cv2.setNumThreads(12)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Colors (BRG Format)
COLOR_BACKGROUND = (20, 20, 40)  # Dark blue
COLOR_WHITE = (255, 255, 255)  # White

prev_time = 0

# Open a video file or capture from webcam
video_path = 0  # Replace with your video path or use 0 for webcam
cap = cv2.VideoCapture(video_path)

cap.set(cv2.CAP_PROP_FPS, 30)

def draw_debug_panel(frame, hand_landmarks):
    """
    Draws a debug panel with landmark coordinates.
    """
    h, w, _ = frame.shape
    panel_width = 300
    cv2.rectangle(frame, (w - panel_width, 0), (w, h), COLOR_BACKGROUND, -1)

    # Header.
    cv2.putText(frame, "DEBUG PANEL", (w - panel_width + 10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_WHITE, 1, cv2.LINE_AA)

    # Landmark coordinates.
    y_offset = 70
    for idx, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.putText(frame, f"Landmark {idx}: ({cx}, {cy})",
                    (w - panel_width + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
        y_offset += 20


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(image_rgb)

    # Calculate FPS
    current_time = time.time()
    delta_time = current_time - prev_time
    fps = 1 / delta_time if delta_time > 0 else 0
    prev_time = current_time

    # Display the FPS
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            draw_debug_panel(frame, hand_landmarks)

            # Detect gestures
            if gt.is_fist(hand_landmarks):
                cv2.putText(frame, "Fist", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gt.is_thumbs_up(hand_landmarks):
                cv2.putText(frame, "Thumbs Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif gt.is_peace_sign(hand_landmarks):
                cv2.putText(frame, "Peace Sign", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Hand Gesture Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()