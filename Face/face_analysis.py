import cv2
import numpy as np
import mediapipe as mp
import time
import json

# extract pixel coordinates (normalized)
def extract_points(landmarks, indices, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])

# Face Analysis
def run_face_analysis():
    # create object
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
    # video frame
    cap = cv2.VideoCapture(0)

    # landmark index mapping
    INDEX_MAP = {
        "LEFT_EYE": [33, 160, 158, 133, 153, 144],
        "RIGHT_EYE": [362, 385, 387, 263, 373, 380],
        "NOSE": 1,
        "LEFT_EYE_CEN": 33,
        "RIGHT_EYE_CEN": 263,
        "LEFT_EAR": 234,
        "RIGHT_EAR": 454
    }

    EAR_THRESHOLD = 0.2
    CLOSED_DURATION = 2.0

    blink_count = 0
    eye_closed = False
    closed_start = None

    # real-time video analyzing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame processing
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb)

        output = {
            "yaw": None,
            "pitch": None,
            "roll": None,
            "ear": None,
            "eye_closed": False,
            "blink_count": blink_count,
            "face_detected": False
        }

        # face detected successfully
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            output["face_detected"] = True

            points = {}
            for key, idx in INDEX_MAP.items(): 
                if isinstance(idx, list):   # for eyes (multiple index)
                    points[key] = extract_points(landmarks, idx, w, h)
                else:                       # for the others
                    pt = landmarks[idx]
                    points[key] = np.array([pt.x * w, pt.y * h])

            # EAR calculation
            left_eye = points["LEFT_EYE"]
            right_eye = points["RIGHT_EYE"]
            A = np.linalg.norm(left_eye[1] - left_eye[5])
            B = np.linalg.norm(left_eye[2] - left_eye[4])
            C = np.linalg.norm(left_eye[0] - left_eye[3])
            left_ear_val = (A + B) / (2.0 * C)

            A = np.linalg.norm(right_eye[1] - right_eye[5])
            B = np.linalg.norm(right_eye[2] - right_eye[4])
            C = np.linalg.norm(right_eye[0] - right_eye[3])
            right_ear_val = (A + B) / (2.0 * C)

            ear = (left_ear_val + right_ear_val) / 2.0
            output["ear"] = round(ear, 3)

            # determine eye closure and blink
            now = time.time()
            if ear < EAR_THRESHOLD:
                if not eye_closed:
                    closed_start = now
                    eye_closed = True
                elif now - closed_start > CLOSED_DURATION:
                    output["eye_closed"] = True
            else:
                if eye_closed and closed_start and now - closed_start < 1.0:
                    blink_count += 1
                eye_closed = False
                closed_start = None
            output["blink_count"] = blink_count

            # estimate head pose
            nose = points["NOSE"]
            left_ear = points["LEFT_EAR"]
            right_ear = points["RIGHT_EAR"]
            left_eye_cen = points["LEFT_EYE_CEN"]
            right_eye_cen = points["RIGHT_EYE_CEN"]

            # yaw (left and right): nose position relative to both ears
            yaw = (np.linalg.norm(nose - left_ear) - np.linalg.norm(nose - right_ear)) / w * 100
            
            # pitch (up and down): nose position relative to eyes
            eye_mid_y = (left_eye_cen[1] + right_eye_cen[1]) / 2
            pitch = (eye_mid_y - nose[1]) / h * 100

            # roll (tilt): slop of horizontal line connecting two eyes
            delta = right_eye_cen - left_eye_cen
            roll = np.degrees(np.arctan2(delta[1], delta[0]))

            output["yaw"] = round(yaw, 2)
            output["pitch"] = round(pitch, 2)
            output["roll"] = round(roll, 2)

        # json output file
        print(json.dumps(output, indent=2))
        cv2.imshow("MediaPipe Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_analysis()
