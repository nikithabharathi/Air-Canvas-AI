import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, track_conf=0.7, model_path=None):
        self.model_path = model_path or os.path.join("models", "hand_landmarker.task")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Missing MediaPipe model: {self.model_path}\n"
                "Put the Hand Landmarker .task file in the models folder."
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=track_conf,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self._last_timestamp_ms = 0

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now_ms = int(time.time() * 1000)
        timestamp_ms = now_ms if now_ms > self._last_timestamp_ms else self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        return self.results

    def get_landmarks(self, frame):
        lm_list = []

        if self.results and self.results.hand_landmarks:
            hand_landmarks = self.results.hand_landmarks[0]
            h, w, _ = frame.shape

            for idx, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((idx, cx, cy))

        return lm_list

    def draw_hands(self, frame):
        if not self.results or not self.results.hand_landmarks:
            return

        h, w, _ = frame.shape

        for hand_landmarks in self.results.hand_landmarks:
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            for p1, p2 in HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks[p1].x * w), int(hand_landmarks[p1].y * h)
                x2, y2 = int(hand_landmarks[p2].x * w), int(hand_landmarks[p2].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)