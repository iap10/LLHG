import cv2
import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

class YoloTRT:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.INFO)

        # Load TensorRT engine
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.batch_size = 1

        # Allocate memory
        input_shape = (1, 3, 640, 640)
        output_shape = (1, 25200 * 7)

        self.host_inputs = [cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)]
        self.cuda_inputs = [cuda.mem_alloc(self.host_inputs[0].nbytes)]
        self.host_outputs = [cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)]
        self.cuda_outputs = [cuda.mem_alloc(self.host_outputs[0].nbytes)]
        self.bindings = [int(self.cuda_inputs[0]), int(self.cuda_outputs[0])]
        self.stream = cuda.Stream()

        # Configuration
        self.yolo_version = "v5"
        self.CONF_THRESH = 0.3
        self.IOU_THRESHOLD = 0.4
        self.LEN_ONE_RESULT = 7
        self.class_names = ["Bottle", "Toy", "Computer keyboard", "Pen",
                            "Mobile phone", "Computer mouse", "Tablet computer", "Human hand"]
        self.moderate_state = [0,2,5,6,7]
        self.play_state = [1,4]
        self.study_state = [3]

    def PreProcessImg(self, img):
        origin_h, origin_w = img.shape[:2]
        image_resized = cv2.resize(img, (640, 640))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_rgb = image_rgb.transpose(2, 0, 1) / 255.0
        input_image = np.expand_dims(image_rgb, axis=0).astype(np.float32)
        return input_image, img, origin_h, origin_w

    def Inference(self, img):
        input_image, image_raw, origin_h, origin_w = self.PreProcessImg(img)
        np.copyto(self.host_inputs[0], input_image.ravel())
        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)

        t1 = time.time()
        self.context.execute_async(self.batch_size, self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()
        t2 = time.time()

        output = self.host_outputs[0]
        t = t2 - t1
        return output, t

    def PostProcess(self, output, origin_h, origin_w):
        num = int(output[0])
        pred = np.reshape(output[1:], (-1, self.LEN_ONE_RESULT))[:num, :]
        pred = pred[:, :6]

        boxes = self.NonMaxSuppression(pred, origin_h, origin_w,
                                       conf_thres=self.CONF_THRESH,
                                       nms_thres=self.IOU_THRESHOLD)

        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def NonMaxSuppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        boxes = prediction[prediction[:, 4] >= conf_thres]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])

        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)

        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]

        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]

        boxes = np.stack(keep_boxes, 0)
        return boxes

    def xywh2xyxy(self, origin_h, origin_w, boxes):
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes_xyxy

    def bbox_iou(self, box1, box2):
        x1 = np.maximum(box1[:, 0], box2[:, 0])
        y1 = np.maximum(box1[:, 1], box2[:, 1])
        x2 = np.minimum(box1[:, 2], box2[:, 2])
        y2 = np.minimum(box1[:, 3], box2[:, 3])
        inter_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        iou = inter_area / (area1 + area2 - inter_area + 1e-6)
        return iou

    def determine_state(self, obj_ids, held):
        """
        Determine overall state:
        0 = Moderate (default), 1 = Play, 2 = Study
        Based on held class IDs
        """
        found_states = set()

        for cid, is_held in zip(obj_ids, held):
            if is_held:
                if cid in self.play_state:
                    return 1  # Play 우선
                elif cid in self.study_state:
                    found_states.add(2)  # Study
                elif cid in self.moderate_state:
                    found_states.add(0)  # Moderate

        if 0 in found_states:
            return 0
        elif 2 in found_states:
            return 2
        else:
            return 0

    def held_status_vectorized(self, boxes, class_ids, hand_class_id=7, iou_thresh=0.2):
        boxes = np.array(boxes)
        class_ids = np.array(class_ids)

        hand_mask = class_ids == hand_class_id
        hand_boxes = boxes[hand_mask]
        object_boxes = boxes[~hand_mask]
        object_ids = class_ids[~hand_mask]

        held = np.zeros(len(object_boxes), dtype=bool)
        for i, obox in enumerate(object_boxes):
            for hbox in hand_boxes:
                if self.compute_iou_single(obox, hbox) > iou_thresh:
                    held[i] = True
                    break

        return object_boxes, object_ids, held

    def compute_iou_single(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def run_camera_inference(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Can't open Camera")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output, t = self.Inference(frame)
            boxes, scores, class_ids = self.PostProcess(output, frame.shape[0], frame.shape[1])

            objs, obj_ids, held = self.held_status_vectorized(boxes, class_ids)
            state = self.determine_state(obj_ids, held)

            for box, cid, is_held in zip(objs, obj_ids, held):
                x1, y1, x2, y2 = map(int, box)
                label = f"{self.class_names[int(cid)]} {'[Held]' if is_held else '[Free]'}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if is_held else (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

            fps = 1.0 / (t + 1e-6)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("YOLOv5n TensorRT Inference", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = YoloTRT("yolov5n.engine")
    model.run_camera_inference()