from ultralytics import YOLO
import cv2
import atexit

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

def detect_object(target):

    if not cap.isOpened():
        print("카메라 실패")
        return None

    for _ in range(30):
        ret, frame = cap.read()
    if not ret:
        return None

    results = model(frame)

    best = None
    best_conf = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])

            if name != target:
                continue

            if conf > best_conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2

                best = {
                    "center_x": cx,
                    "confidence": conf
                }
                best_conf = conf

    return best

def release():
    cap.release()

atexit.register(lambda: cap.release())
