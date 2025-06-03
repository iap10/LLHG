import cv2
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import mediapipe as mp
import json

# Hopenet TensorRT Inference
class HopenetTRT:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.INFO)
        # load trt engine
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read()) 
        # create inference context
        self.context = self.engine.create_execution_context()

        # allocate memory 
        self.input_shape = (1, 3, 224, 224)
        self.output_shapes = [(1, 66)] * 3  # yaw, pitch, roll

        self.host_inputs = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.cuda_inputs = cuda.mem_alloc(self.host_inputs.nbytes)
        self.host_outputs = [cuda.pagelocked_empty(trt.volume(s), dtype=np.float32) for s in self.output_shapes]
        self.cuda_outputs = [cuda.mem_alloc(o.nbytes) for o in self.host_outputs]

        self.bindings = [int(self.cuda_inputs)] + [int(o) for o in self.cuda_outputs]   # list containing input & output address
        self.stream = cuda.Stream()     # stream for asynchronous inference 

    def preprocess(self, frame):
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0   # RGB normalization
        img = img.transpose(2, 0, 1).reshape(1, 3, 224, 224)    # CHW format
        return img.astype(np.float32)

    def infer(self, frame):
        input_img = self.preprocess(frame)
        # on GPU
        np.copyto(self.host_inputs, input_img.ravel())      # flatten
        cuda.memcpy_htod_async(self.cuda_inputs, self.host_inputs, self.stream)

        # asynchronous inference 
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for i in range(3):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.cuda_outputs[i], self.stream)
        
        # wait for all completion 
        self.stream.synchronize()

        # select argmax of prediction
        yaw = float(np.argmax(self.host_outputs[0]) - 33)
        pitch = float(np.argmax(self.host_outputs[1]) - 33)
        roll = float(np.argmax(self.host_outputs[2]) - 33)
        return yaw, pitch, roll

# EAR Calculation
def calc_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])     # vertical 1 
    B = np.linalg.norm(eye[2] - eye[4])     # vertical 2
    C = np.linalg.norm(eye[0] - eye[3])     # horizontal 
    return (A + B) / (2.0 * C)              # formula for EAR

def get_eye_points(landmarks, indexes, w, h):
    return np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indexes])    # landmark -> pixel coordinate

# Face Analysis
def run_face_analysis(hopenet_engine_path):
    hopenet = HopenetTRT(hopenet_engine_path)   # load Hopenet engine 
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)     # track one face
    cap = cv2.VideoCapture(0)

    # landmark index
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    EAR_THRESHOLD = 0.2
    CLOSED_DURATION = 0.2

    blink_count = 0
    eye_closed = False
    closed_start = None

    # real-time video analyzing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # extract face and eyes
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

        # face detection completed
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            output["face_detected"] = True

            # calculate EAR
            left_eye = get_eye_points(landmarks, LEFT_EYE, w, h)
            right_eye = get_eye_points(landmarks, RIGHT_EYE, w, h)
            ear = (calc_ear(left_eye) + calc_ear(right_eye)) / 2.0
            output["ear"] = round(ear, 3)

            # 1. eye closure and blink 
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

            # 2. head pose
            yaw, pitch, roll = hopenet.infer(frame)
            output["yaw"] = round(yaw, 2)
            output["pitch"] = round(pitch, 2)
            output["roll"] = round(roll, 2)

        # output
        print(json.dumps(output, indent=2))
        cv2.imshow("Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_face_analysis("hopenet.engine")
