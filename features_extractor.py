import numpy as np
import math


class HandFeatureExtractor:
    def __init__(self, landmarks, bending_threshold=25, spanning_threshold=22, distance_threshold=1):
        self.landmarks = landmarks
        self.bending_threshold = bending_threshold / 180 * math.pi
        self.spanning_threshold = spanning_threshold / 180 * math.pi
        self.distance_threshold = distance_threshold

    def distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y))

    def angle(self, pO, pA, pB):
        inner_product = (pA.x - pO.x) * (pB.x - pO.x) + (pA.y - pO.y) * (pB.y - pO.y)
        return math.acos(inner_product / (self.distance(pO, pA) * self.distance(pO, pB)))

    def distance_ratio(self):
        thumb_ratio = (self.distance(self.landmarks[2], self.landmarks[17]) /
                       self.distance(self.landmarks[4], self.landmarks[17]))
        index_ratio = (self.distance(self.landmarks[6], self.landmarks[0]) /
                       self.distance(self.landmarks[8], self.landmarks[0]))
        middle_ratio = (self.distance(self.landmarks[10], self.landmarks[0]) /
                       self.distance(self.landmarks[12], self.landmarks[0]))
        ring_ratio = (self.distance(self.landmarks[14], self.landmarks[0]) /
                       self.distance(self.landmarks[16], self.landmarks[0]))
        pink_ratio = (self.distance(self.landmarks[18], self.landmarks[0]) /
                       self.distance(self.landmarks[20], self.landmarks[0]))
        return thumb_ratio, index_ratio, middle_ratio, ring_ratio, pink_ratio

    def bending_angle(self):
        thumb_angle = self.angle(self.landmarks[0], self.landmarks[2], self.landmarks[4])
        index_angle = self.angle(self.landmarks[0], self.landmarks[5], self.landmarks[8])
        middle_angle = self.angle(self.landmarks[0], self.landmarks[9], self.landmarks[12])
        ring_angle = self.angle(self.landmarks[0], self.landmarks[13], self.landmarks[16])
        pink_angle = self.angle(self.landmarks[0], self.landmarks[17], self.landmarks[20])
        return thumb_angle, index_angle, middle_angle, ring_angle, pink_angle

    def spanning_angle(self):
        thumb2index_angle = self.angle(self.landmarks[2], self.landmarks[4], self.landmarks[8])
        index2middle_angle = self.angle(self.landmarks[5], self.landmarks[8], self.landmarks[12])
        middle2ring_angle = self.angle(self.landmarks[9], self.landmarks[12], self.landmarks[16])
        ring2pink_angle = self.angle(self.landmarks[13], self.landmarks[16], self.landmarks[20])
        return thumb2index_angle, index2middle_angle, middle2ring_angle, ring2pink_angle

    def extract(self):
        thumb_ratio, index_ratio, middle_ratio, ring_ratio, pink_ratio = self.distance_ratio()
        thumb_angle, index_angle, middle_angle, ring_angle, pink_angle = self.bending_angle()
        thumb2index_angle, index2middle_angle, middle2ring_angle, ring2pink_angle = self.spanning_angle()
        features = np.array([
            thumb_ratio, index_ratio, middle_ratio, ring_ratio, pink_ratio,
            thumb_angle, index_angle, middle_angle, ring_angle, pink_angle,
            thumb2index_angle, index2middle_angle, middle2ring_angle, ring2pink_angle
        ])

        binary_features = np.zeros_like(features, dtype=int)
        binary_features[0:5] = (features[0:5] > self.distance_threshold).astype(int)
        binary_features[5:10] = (features[5:10] <= self.bending_threshold).astype(int)
        binary_features[10:14] = (features[10:14] > self.spanning_threshold).astype(int)

        return binary_features


if __name__ == "__main__":
    import mediapipe as mp
    import cv2
    import os

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.5
    )
    mpDraw = mp.solutions.drawing_utils
    LandmarkSpec = mpDraw.DrawingSpec((0, 0, 255), 2, 2)
    ConnectionSpec = mpDraw.DrawingSpec((0, 255, 0), 2, 2)

    image_list = os.listdir('./test')
    for name in image_list:
        image_path = os.path.join('./test', name)
        image = cv2.imread(image_path)
        imageHeight, imageWidth = image.shape[0], image.shape[1]
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:
            for hand_idx, hand_lmks in enumerate(results.multi_hand_landmarks):
                mpDraw.draw_landmarks(image, hand_lmks, mpHands.HAND_CONNECTIONS, LandmarkSpec, ConnectionSpec)

                hfe = HandFeatureExtractor(hand_lmks.landmark)
                binary_features = hfe.extract()
                print(f"==> image: {name}, binary features = {binary_features}")

                for idx, lmk in enumerate(hand_lmks.landmark):
                    xPos, yPos = int(lmk.x * imageWidth), int(lmk.y * imageHeight)
                    cv2.putText(image, str(idx), (xPos + 5, yPos - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            cv2.imshow("Hands Recognition", image)
            cv2.waitKey()
    cv2.destroyAllWindows()
