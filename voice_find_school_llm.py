from ultralytics import YOLO
import cv2, os, json
import speech_recognition as sr
from openai import OpenAI
from Lidar_Function import get_xy

api_key = os.getenv("SCHOOL_LLM_API_KEY")

if not api_key:
    raise RuntimeError("SCHOOL_LLM_API_KEY 환경변수가 없습니다.")

client = OpenAI(
    api_key=api_key,
    base_url="https://factchat-cloud.mindlogic.ai/v1/gateway",
)

model = YOLO("yolov8n.pt")

def listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("찾을 물체를 말하세요")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="ko-KR")
        print(" 음성:", text)
        return text
    except:
        return None


def parse(text):
    if not text:
        return None

    res = client.chat.completions.create(
        model="claude-sonnet-4-6",
        messages=[
            {
                "role": "system",
                "content": "한국어 명령에서 찾을 물체를 YOLO COCO 클래스명으로 바꾸고 JSON만 출력해라."
            },
            {
                "role": "user",
                "content": f'{text}\n예시: {{"action":"find","object":"cup"}}'
            }
        ],
        temperature=0
    )

    try:
        return json.loads(res.choices[0].message.content.strip())
    except:
        return None
    
def detect(target):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, imgsz=320)[0]

        for box in result.boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if name == target and conf > 0.35:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                img_width = frame.shape[1]

                x, y = get_xy(cx, img_width)
                
                print(f"LIDAR 좌표 -> X={x:.1f}, Y={y:.1f}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

cmd = listen()
data = parse(cmd)

if not data or data.get("action") != "find":
    print("명령 해석 실패")
else:
    target = data["object"]
    print("찾을 물체:", target)
    detect(target)
    
def get_llm_response(user_input):

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    command = listen()

    parsed = parse(command)

    print(parsed)