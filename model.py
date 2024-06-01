import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

prev_y = 0

def detect_gesture(hand_landmarks):
    global prev_y

    index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    
    # Determine direction of hand movement => kamera u kompjuteru je invertirana
    direction = None
    if index_tip_y > prev_y:
        direction = "down"
    elif index_tip_y < prev_y:
        direction = "up"
    
    prev_y = index_tip_y
    
    return direction

def perform_action(direction):
    if direction == "up":
        pyautogui.press('volumeup')
    elif direction == "down":
        pyautogui.press('volumedown')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            direction = detect_gesture(hand_landmarks)
            if direction:
                perform_action(direction)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
