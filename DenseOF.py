import numpy as np
import cv2
import time
import sys

"""
176 x 144 пикселей Quarter CIF
320 x 144 пикселей
320 x 240 пикселей QVGA
352 x 288 пикселей CIF
320 x 240 пикселей QVGA
640 x 360 пикселей
640 x 480 пикселей VGA
854 x 480 пикселей WVGA
1024 x 768 пикселей XGA
1280 x 720 пикселей HD ready
1920 x 1080 пикселей Full HD

У нас пока есть:

Кадров 1828 Ширина 1280.0 Высота 720.0 36 градусов (лол)
Кадров 2941 Ширина 1920.0 Высота 1080.0 155 градусов
"""


def draw_flow(img, flow, step=8):  # step=16
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    # точки н-нада?
    # for (x1, y1), (_x2, _y2) in lines:
    #    cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    cv2.rectangle(img_bgr, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 0, 0), 1)
    cv2.putText(img_bgr, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_bgr, f"{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} кадр", (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img_bgr, f"{cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.3f} секунд", (20, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                2)



    return img_bgr


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


match 1:  # TODO настройка
    case 0:
        cap = cv2.VideoCapture(0)
        start_frame = 100  #
        viewing_angle = 361
    case 1:
        cap = cv2.VideoCapture("./PERFECT_BANDO_FPV_FREESTYLE.mp4")
        start_frame = 0  #
        viewing_angle = 155
    case 2:
        cap = cv2.VideoCapture("./Полёт д1.mp4")
        start_frame = 0  #
        viewing_angle = 36
    case _:
        viewing_angle = 361
        print("\\(-_-)/")
        sys.exit()
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
viewing_angle_req = 60
if viewing_angle > viewing_angle_req:
    width_res = round(width * viewing_angle_req/viewing_angle)
    height_res = round(height * viewing_angle_req/viewing_angle)
    rect_x0 = round((width - width_res)/2)
    rect_y0 = round((height - height_res)/2)
    rect_x1 = rect_x0 + width_res
    rect_y1 = rect_y0 + height_res
else:
    rect_x0 = -1
    rect_y0 = -1
    rect_x1 = width
    rect_y1 = height

print('Кадров', length, 'Ширина', width, 'Высота', height, 'FPS', video_fps)
print('Запуск с', start_frame)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # frame_number

suc, prev = cap.read()
if not suc:
    print('Картинки нет')
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:

    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    # Сравнение кадров?
    if not (np.bitwise_xor(prevgray, gray).any()):
        print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} кадр")

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for current frame detection
    fps = 1 / (end - start)
    # print(f"{fps:.2f} FPS")

    cv2.imshow('flow', draw_flow(gray, flow))
    # cv2.imshow('flow HSV', draw_hsv(flow))

    key = cv2.waitKey(5)
    if (key == ord('q')) or (key == ord('й')):
        break

cap.release()
cv2.destroyAllWindows()