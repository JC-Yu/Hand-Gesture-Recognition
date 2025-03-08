import os
import numpy as np
import mediapipe as mp
import cv2
import re
import pandas as pd
from features_extractor import HandFeatureExtractor

# ========================== Parameters ==========================
DATASET_PATH = './hand_gestures'
CSV_PATH = './gesture_features.csv'
# ================================================================

if __name__ == "__main__":
    # TODO 1: MediaPipe Initialization
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.5
    )

    # TODO 2: Load and Preprocess Dataset
    subject_dirs = sorted(
        os.listdir(DATASET_PATH),
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    gesture_names = sorted(
        os.listdir(os.path.join(DATASET_PATH, subject_dirs[0])),
        key=lambda x: int(re.search(r'\d+', x).group())
    )
    gesture_labels = {gesture: idx for idx, gesture in enumerate(gesture_names)}
    print(f"subject dirs: {subject_dirs}\n"
          f"gesture names: {gesture_names}\n"
          f"gesture labels: {gesture_labels}")

    columns = [f"f{i}" for i in range(14)] + ["label", "gesture", "subject"]
    features = []

    for subject in subject_dirs:
        subject_path = os.path.join(DATASET_PATH, subject)

        for gesture, label in gesture_labels.items():
            gesture_path = os.path.join(subject_path, gesture)
            if not os.path.exists(gesture_path):
                continue

            image_lists = os.listdir(gesture_path)
            for image_name in image_lists:
                image_path = os.path.join(gesture_path, image_name)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # extract binary features
                        hfe = HandFeatureExtractor(hand_landmarks.landmark, 25, 22, 1)
                        binary_features = hfe.extract()
                        features.append(np.append(binary_features, [label, gesture, subject]))
                        print(f"{binary_features}, {label}, {gesture}, {subject}")

    df = pd.DataFrame(features, columns=columns)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n==> Features had been saved in {CSV_PATH}")
