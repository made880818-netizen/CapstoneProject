from ultralytics import YOLO
import cv2

# 찾고 싶은 물건 이름
TARGET_OBJECT = "bottle"

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("카메라를 열 수 없습니다.")
        break

    results = model(frame)
    found = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = model.names[class_id]
            confidence = float(box.conf[0])

            if name == TARGET_OBJECT and confidence >= 0.5:
                found = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                text = f"{name} {confidence:.2f} center=({center_x},{center_y})"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                print(f"찾음: {name}, 신뢰도: {confidence:.2f}, 중심좌표: ({center_x}, {center_y})")

    if not found:
        cv2.putText(frame, f"Finding {TARGET_OBJECT}...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Find Object", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()