import math
from rplidar import RPLidar

from config import (
    LIDAR_PORT,
    CAMERA_FOV,
    ANGLE_OFFSET
)

lidar = None


def init_lidar():
    global lidar

    if lidar is None:
        lidar = RPLidar(LIDAR_PORT)
        print("[LiDAR] 연결 완료")


def stop_lidar():
    global lidar

    if lidar:
        lidar.stop()
        lidar.disconnect()
        print("[LiDAR] 종료")


def pixel_to_angle(x, img_width):
    return (x / img_width - 0.5) * CAMERA_FOV


def get_distance_at_angle(scan, target_angle):
    closest = None
    min_diff = 999

    for (_, angle, distance) in scan:

        diff = min(
            abs(angle - target_angle),
            360 - abs(angle - target_angle)
        )

        if diff < min_diff:
            min_diff = diff
            closest = distance

    return closest


def get_scan():
    try:
        return next(lidar.iter_scans())
    except:
        return None


def get_xy(center_x, img_width):

    scan = get_scan()
    if scan is None:
        return None, None

    angle = pixel_to_angle(center_x, img_width)
    angle = (angle + ANGLE_OFFSET) % 360

    distance = get_distance_at_angle(scan, angle)

    if distance is None or distance == 0:
        return None, None

    x = distance * math.cos(math.radians(angle))
    y = distance * math.sin(math.radians(angle))

    return x, y
