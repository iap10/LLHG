import cv2
import numpy as np
import time

class VisualFeedbackModule:
    def __init__(self):
        self.prev_state = None
        self.last_state_change_time = time.time()
        self.alert_duration = {
            "drowsy": 2.0,
            "distracted": 1.0
        }

    def update(self, state):
        now = time.time()

        if state != self.prev_state:
            self.last_state_change_time = now
            self.prev_state = state

        duration = now - self.last_state_change_time

        if state == "drowsy" and duration >= self.alert_duration["drowsy"]:
            self.display_message("Drowsy! Please wake up!", (0, 0, 255))  # Red

        elif state == "distracted" and duration >= self.alert_duration["distracted"]:
            self.display_message("Distracted! Please focus.", (0, 165, 255))  # Orange

        elif state == "focus":
            self.display_message("Focused. Keep it up!", (0, 180, 0))  # Green

        elif state == "unresponsive":
            self.display_message("No user detected.", (128, 128, 128))  # Gray

        else:
            self.display_message(f"State: {state}", (100, 100, 100))

    def display_message(self, text, color=(0, 0, 0)):
        img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
        cv2.putText(img, text, (80, 380), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
        cv2.namedWindow("Attention Status", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Attention Status", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Attention Status", img)
        cv2.waitKey(1)
