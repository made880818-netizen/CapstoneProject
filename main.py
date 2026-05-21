from config import IMAGE_WIDTH
from Lidar_Function import (
    init_lidar,
    stop_lidar,
    get_xy
)
from llm_bridge import parse_command
from yolo_detect import detect_object   
from voice_find import cap


def main():

    init_lidar()

    try:
        while True:

            user_input = input("\n명령 입력: ")

            if user_input == "exit":
                break

            command = parse_command(user_input)

            if command is None:
                print("명령 해석 실패")
                continue

            action = command.get("action")
            target_object = command.get("object")

            if action != "find" or target_object is None:
                print("잘못된 명령")
                continue

            print(f"[대상] {target_object}")

            detection = detect_object(target_object)

            if detection is None:
                print("[YOLO] 못 찾음")
                continue

            cx = detection["center_x"]

            x, y = get_xy(cx, IMAGE_WIDTH)

            if x is None:
                print("[LiDAR] 실패")
                continue

            print(f"{target_object} 위치 → x={x:.1f}, y={y:.1f}")
    except KeyboardInterrupt:
        print("사용자 종료")

    finally:
        stop_lidar()
        cap.release()

if __name__ == "__main__":
    main()
