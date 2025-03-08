import cv2
import mediapipe as mp
import joblib
import time
import os
from collections import deque
from features_extractor import HandFeatureExtractor

# ========================== Parameters ==========================
MODEL_PATH = "./gesture_model_SVM.pkl"
GESTURE_LABELS = {
    0: '5', 1: '2', 2: '3', 3: 'pause', 4: 'wrist', 5: 'rock',
    6: '1', 7: '4', 8: 'ok', 9: 'go home', 10: 'end'
}
CAM_ID = 0
WINDOW_SIZE = 5
# ================================================================

if __name__ == "__main__":
    # TODO 1: Initialization
    clf = joblib.load(MODEL_PATH)
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils
    LandmarkSpec = mpDraw.DrawingSpec((0, 0, 255), 2, 2)
    ConnectionSpec = mpDraw.DrawingSpec((0, 255, 0), 2, 2)
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print(f"==> Camera {CAM_ID} open failed.")
        cap.release()
        exit()

    # TODO 2: Start Recognition
    current_time = time.time()
    gesture_window = deque(maxlen=WINDOW_SIZE)
    confirmed_gesture = "Unknown"
    while True:
        ret, image = cap.read()
        if not ret:
            print(f"==> Cannot read current frame, camera closed.")
            break
        imageHeight, imageWidth = image.shape[0], image.shape[1]
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(imageRGB)
        if results.multi_hand_landmarks:
            for hand_idx, hand_lmks in enumerate(results.multi_hand_landmarks):
                mpDraw.draw_landmarks(image, hand_lmks, mpHands.HAND_CONNECTIONS, LandmarkSpec, ConnectionSpec)
                hfe = HandFeatureExtractor(hand_lmks.landmark, 25, 22, 1)
                binary_features = hfe.extract().reshape(1, -1)
                prediction = clf.predict(binary_features)[0]
                gesture_name = GESTURE_LABELS.get(prediction, "Unknown")
                gesture_window.append(gesture_name)
                if len(gesture_window) == WINDOW_SIZE and all(g == gesture_window[0] for g in gesture_window):
                    confirmed_gesture = gesture_window[0]
                print(f"Confirmed: {confirmed_gesture}, Predicted: {gesture_name}")

        last_time = current_time
        current_time = time.time()
        fps = int(1.0 / (current_time - last_time))
        cv2.putText(image, f"FPS:{fps}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Gesture: {confirmed_gesture}", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Hands Recognition", image)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("==> Camera closed.")
            break
        if key & 0xFF == ord('s'):
            os.makedirs('./results', exist_ok=True)
            cv2.imwrite(f'./results/{str(current_time)}.jpg', image)
            print("==> Image saved in ./results/ successfully.")

    # TODO 3: Exit and Release All Resources
    cap.release()
    cv2.destroyAllWindows()
