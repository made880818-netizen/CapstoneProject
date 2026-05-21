from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "YOLO API running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(image)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            detections.append({
                "name": model.names[cls_id],
                "confidence": round(conf, 3),
                "box": [x1, y1, x2, y2]
            })

    return {"detections": detections}