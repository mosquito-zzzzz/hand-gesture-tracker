import cv2
import mediapipe as mp
import numpy as np
from pyautogui import scroll, size
from pynput.mouse import Button, Controller
import gestures as gt
import warnings 
warnings.filterwarnings("ignore", message=".NORM_RECT without IMAGE_DIMENSIONS.")

cv2.setUseOptimized(True)
cv2.setNumThreads(12)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Screen dimensions
screen_w, screen_h = size()

mouse = Controller()
click_performed = False  # Flag to prevent multiple clicks

prev_x = None
prev_y = None

# Smoothing parameters (to reduce mouse jitter)
smoothing_factor = 30
sensitivity = 9

cap = cv2.VideoCapture(0)  # Initialize camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for natural movement
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Thumb + Index Pinch: Move Mouse
            if gt.is_pinch(mp_hands.HandLandmark.THUMB_TIP,
               mp_hands.HandLandmark.INDEX_FINGER_TIP, landmarks):
                # Get the normalized index fingertip coordinates (range 0 to 1)
                current_x = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                current_y = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                # If this is the first frame of the pinch, initialize prev_x and prev_y
                if prev_x is None or prev_y is None:
                    prev_x = current_x
                    prev_y = current_y
                else:
                    # Calculate the relative movement (delta) since the last frame
                    delta_x = current_x - prev_x
                    delta_y = current_y - prev_y

                    # Invert the x movement if desired (to reverse the direction)
                    delta_x = -delta_x

                    # Convert the relative movement to screen pixels
                    # (Assuming screen_w and screen_h are your screen dimensions)
                    move_x = delta_x * screen_w * sensitivity
                    move_y = delta_y * screen_h * sensitivity

                    # Apply smoothing by dividing the movement (optional)
                    move_x = move_x / smoothing_factor
                    move_y = move_y / smoothing_factor

                    # Move the mouse relatively; Controller.move() adds the offset to the current position
                    mouse.move(move_x, move_y)

                    # Update previous pinch coordinates for the next frame
                    prev_x = current_x
                    prev_y = current_y
            else:
                # If pinch is not detected, reset the previous coordinates so the next pinch starts fresh
                prev_x = None
                prev_y = None

            # Thumb + Middle Pinch: Left Click
            if gt.is_pinch(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, landmarks):
                if not click_performed:  # Click only if it hasn't been done yet
                    mouse.click(Button.left)
                    click_performed = True  # Set flag to prevent further clicks
                else:
                    click_performed = False  # Reset flag when pinch is released

            # Thumb + Ring Pinch: Right Click
            if gt.is_pinch(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, landmarks):
                if not click_performed:  # Click only if it hasn't been done yet
                    mouse.click(Button.right)
                    click_performed = True  # Set flag to prevent further clicks
                else:
                    click_performed = False  # Reset flag when pinch is released

            # Thumb + Pinky Pinch: Scroll
            if gt.is_pinch(mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.PINKY_TIP, landmarks):
                pinky_y = landmarks[mp_hands.HandLandmark.PINKY_TIP].y
                scroll_amount = np.interp(pinky_y, [0, 1], [100, -100])  # Adjust scroll sensitivity
                scroll(int(scroll_amount))

    # Exit on 'q' key press (works even without cv2.imshow)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()