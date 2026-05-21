from ultralytics import YOLO
import cv2
import speech_recognition as sr

VOICE_TO_OBJECT = {
    "컵": "cup",
    "컵을": "cup",
    "병": "bottle",
    "물병": "bottle",
    "핸드폰": "cell phone",
    "휴대폰": "cell phone",
    "마우스": "mouse",
    "키보드": "keyboard"
}

def listen_command():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("말하세요. 예: 컵 찾아")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="ko-KR")
        print("인식된 음성:", text)
        return text
    except sr.UnknownValueError:
        print("음성을 이해하지 못했습니다.")
        return None
    except sr.RequestError as e:
        print("음성 인식 서버 오류:", e)
        return None

def extract_target_object(text):
    if text is None:
        return None

    for korean, yolo_name in VOICE_TO_OBJECT.items():
        if korean in text:
            return yolo_name

    return None

command = listen_command()
target_object = extract_target_object(command)

if target_object is None:
    print("찾을 물체를 정하지 못했습니다.")
    exit()

print("찾을 물체:", target_object)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    #linux 실행 시 cv2.VideoCapture(0)로 변경

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 안됨")
        break

    results = model(frame)
    display = frame.copy()

    found = False

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        name = model.names[class_id]
        conf = float(box.conf[0])

        if name != target_object:
            continue

        if conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            display,
            f"{name} {conf*100:.1f}% ({cx}, {cy})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(f"찾음: {name}, 확률: {conf*100:.1f}%, 중심좌표: ({cx}, {cy})")
        found = True

    if not found:
        cv2.putText(
            display,
            f"Finding: {target_object}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Voice Object Detection", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()