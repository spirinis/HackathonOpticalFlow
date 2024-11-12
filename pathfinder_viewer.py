import numpy as np
import cv2
import time
import sys

# МЕНЮ НАСТРОЙКИ
add_sparse_flow = True  # Нарисует вектора прошедших фильтрацию точек
add_sparse_lamps = True  # Подсветит опасные точки
show_lamps = False  # Откроет дополнительное окно только с опасными точками
draw_bad_flow = True  # Отобразит отсеянные фильтром точки и вектора
start_frame = 0  # можно посмотреть конкретный момент
step = 30  # Установка плотности точек. Расстояние между точками. Меняется в зависимости от производительности железа

viewing_angle = 155
viewing_angle_req = 60


def draw_flow(img_shape, flow, step=14):  # step=16
    """Возвращает слой с векторами движения пикселей"""
    h, w = img_shape
    img_bgr = np.zeros((h, w, 3), np.uint8)
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    global lines
    # сборка массива по столбцам и строкам, на выходе размер (999, 2, 2)
    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)  # математическое округление

    cv2.polylines(img_bgr, lines, False, (0, 255, 0))

    # точки н-нада?
    if 1:
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(img_bgr, center=(x1, y1), radius=1, color=(0, 255, 0),  thickness=-1)

    return img_bgr


def draw_grid(img_shape, step=20, colored_cross=False,
              viewing_angle_rect=False, cross=False, grid=False, blinds=False) -> np.ndarray:
    """Возвращает слой с сеткой"""
    h, w = img_shape
    img_bgr = np.zeros((h, w, 3), np.uint8)

    if grid:
        # сетка пикселей, а то я не вижу, куда что рисовать BGR
        x_lines = np.int32([[[i, 0], [i, h]] for i in range(step, w, step)])
        y_lines = np.int32([[[0, i], [w, i]] for i in range(step, h, step)])
        cv2.polylines(img_bgr, x_lines, False, (0, 0, 100), 1)
        cv2.polylines(img_bgr, y_lines, False, (0, 0, 100), 1)
    if cross:
        # Центр кадра
        cv2.polylines(img_bgr, [np.int32([[half_width, 0], [half_width, height]])], False, (0, 0, 255), 1)
        cv2.polylines(img_bgr, [np.int32([[0, half_height], [width, half_height]])], False, (0, 0, 255), 1)
    if colored_cross:
        # Цветной крест
        cv2.line(img_bgr, (0, half_height), (15, half_height), (0, 0, 255), 5)  # Красный
        cv2.line(img_bgr, (half_width, 0), (half_width, 15), (0, 255, 0), 5)  # Зелёный
        cv2.line(img_bgr, (width, half_height), (width-15, half_height), (255, 200, 170), 5)  # Голубой
        cv2.line(img_bgr, (half_width, height), (half_width, height-15), (255, 100, 100), 5)  # Синий
    if blinds:
        # blinds Нужно ?закрыть? зону, где видны подвижные винты
        cv2.rectangle(img_bgr, (620, height-200), (-1, height), (0, 0, 255,), 1)
        cv2.rectangle(img_bgr, (width-620, height-200), (width, height), (0, 0, 255), 1)
        # cv2.ellipse(img_bgr, (width, height-70), (670, 150), 0, 160, 270, (0, 0, 255), 3)
        # cv2.ellipse(img_bgr, (0, height-70), (670, 150), 0, 270, 380, (0, 255, 0), 3)
    if viewing_angle_rect:
        # считаем синий прямоугольник угла обзора в 60 градусов
        width_res = round(width * viewing_angle_req / viewing_angle)
        height_res = round(height * viewing_angle_req / viewing_angle)
        rect_x0 = round((width - width_res) / 2)
        rect_y0 = round((height - height_res) / 2)
        rect_x1 = rect_x0 + width_res
        rect_y1 = rect_y0 + height_res
        cv2.rectangle(img_bgr, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 0, 0), 3)

    return img_bgr


def draw_hsv(flow_):
    """Возвращает слой с радугой в цветовом пространстве hsv,
    где цвет - направление вектора, интенсивность - длинна вектора"""
    h, w = flow_.shape[:2]
    fx, fy = flow_[:, :, 0], flow_[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def get_flow_lk(img1, img2, points_):
    frame_layer = np.zeros((height, width, 3), np.uint8)
    win_size = (45, 45)
    # (-1, 2), (-1, 1), (-1, 1)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(img2, img1, points_, None, winSize=win_size, maxLevel=2,
                                                     criteria=(
                                                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    flow_ = next_pts - points_
    fx, fy = flow_[:, 0], flow_[:, 1]
    x, y = points_[:, 0], points_[:, 1]
    ang = np.arctan2(fy, fx)  # 4.71238898038469 = np.radians(270)
    ang_middle = (np.arctan2(half_height - y, half_width - x))
    modulus = np.sqrt(fx * fx + fy * fy)
    modulus_middle = np.sqrt((half_width - x)**2 + (half_height - y)**2)

    modulus = modulus / (5+np.sqrt(modulus_middle)) * 30  # (modulus_middle/((half_width + half_height-2*step))) / 3
    fx = modulus * np.cos(ang)
    fy = modulus * np.sin(ang)
    next_pts = np.vstack([x+fx, y+fy]).T
    def fang(x, y):
        ar = np.arctan2(half_height - y, half_width - x) + np.pi
        print(ar)
        print(ar * 180 / np.pi)
    next_pts = np.int32(next_pts + 0.5)  # математическое округление
    points_ = np.int32(points_ + 0.5)
    # mask = np.greater(modulus, np.median(modulus) * 1.2)
    mask = (np.median(modulus) * 1.0 < modulus) & (modulus < np.percentile(modulus, 99))
    # mask = (np.percentile(modulus, 50) < modulus) & (modulus < np.percentile(modulus, 99))
    # op.open_images([img, draw_sparse_lamps(img.shape[:2], flow_m, points)], 'img flow')

    # mask = np.not_equal(points, next_pts)
    # mask = mask.T  # mask.reshape(2, -1)
    # mask = (mask[0] | mask[1]) #& np.less(abs(ang_middle - ang), 90*np.pi / 180)
    # mask = np.where((modulus > np.mean(modulus)*0.9), mask, np.False_)  # + грубее, - чувствительнее
    # # np.where(modulus < 2 * np.mean(modulus), mask, np.False_)
    mask_inv = ~mask
    points_, points_bad = points_[mask], points_[mask_inv]
    next_pts, nextPts_bad = next_pts[mask], next_pts[mask_inv]
    modulus, modulus_bad = modulus[mask], modulus[mask_inv]
    ang, ang_bad = ang[mask], ang[mask_inv]

    flow = next_pts - points_
    lines = np.concatenate((points_, next_pts), axis=1) #  lines.shape = (-1, 4)
    rlines = lines.reshape(-1, 2, 2)
    _ = cv2.polylines(frame_layer, rlines, False, (0, 0, 255))
    # точки н-нада?
    for x1, y1, _x2, _y2 in lines:
        cv2.circle(frame_layer, center=(x1, y1), radius=1, color=(255, 0, 255), thickness=1)
    # flow = np.concatenate((flow, points), axis=1)

    if draw_bad_flow:
        lines_bad = np.concatenate((points_bad, nextPts_bad), axis=1)
        rlines_bad = lines_bad.reshape(-1, 2, 2)
        _ = cv2.polylines(frame_layer, rlines_bad, False, (255, 255, 0,))
        for x1, y1, _x2, _y2 in lines_bad:
            cv2.circle(frame_layer, center=(x1, y1), radius=1, color=(255, 255, 0), thickness=1)

    return frame_layer, flow, points_


def draw_sparse_hsv(img_shape, flow_, points):
    """Возвращает слой с радугой в цветовом пространстве hsv,
    где цвет - направление вектора, интенсивность - длинна вектора"""
    h, w = img_shape[:2]
    fx, fy = flow_[:, 0], flow_[:, 1]

    ang = np.arctan2(fy, fx) + np.pi
    modulus = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    for (x, y), a, m in zip(points, ang, modulus):
        hsv[y, x, 0] = a * (180 / np.pi / 2)
        hsv[y, x, 1] = 255
        hsv[y, x, 2] = np.minimum(m * 4, 255)
    # hsv[..., 0] = ang * (180 / np.pi / 2)
    # hsv[..., 1] = 255
    # hsv[..., 2] = np.minimum(modulus * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, y, in points:
        cv2.circle(bgr, center=(x, y), radius=6, color=(int(bgr[y, x, 0]), int(bgr[y, x, 1]), int(bgr[y, x, 2])),
                   thickness=-1)

    return bgr


def draw_sparse_lamps(img_shape, flow_, points):
    """Возвращает слой с радугой в цветовом пространстве hsv,
    где цвет - направление вектора, интенсивность - длинна вектора"""

    try:
        fx, fy = flow_[:, 0], flow_[:, 1]
    except TypeError:
        print(type(flow_))
        print('flow_ =', flow_)
        return
    ang = np.arctan2(fy, fx) + np.pi
    modulus = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((height, width, 3), np.uint8)
    for (x, y), a, m in zip(points, ang, modulus):
        hsv[y, x, 0] = 0
        hsv[y, x, 1] = 255
        hsv[y, x, 2] = np.minimum(50 + m * 2, 255)  # np.minimum(modulus * 4, 255)
    # hsv[..., 0] = ang * (180 / np.pi / 2)
    # hsv[..., 1] = 255

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, y, in points:
        cv2.circle(bgr, center=(x, y), radius=6, color=(int(bgr[y, x, 0]), int(bgr[y, x, 1]), int(bgr[y, x, 2])),
                   thickness=-1)
    return bgr


match 5:  # Выбор видео TODO
    case 0:
        cap = cv2.VideoCapture("./Стены_вокруг_куст_стена_на_пути9.mp4")  #
    case 1:
        cap = cv2.VideoCapture("./Тёмный_коридор_арка_куст.mp4")
    case 2:
        cap = cv2.VideoCapture("./Резкий_поворот_кусты_стена_разбился.mp4")
    case 3:
        cap = cv2.VideoCapture("./Тёмный_коридор_колонны .mp4")
    case 4:
        cap = cv2.VideoCapture("./здания_дверь_колонны_перекрытия.mp4")
    case 5:
        cap = cv2.VideoCapture("./FPV_FREESTYLE_очень_сложные_движения.mp4")
    case _:
        print("\\(-_-)/")
        sys.exit()
# расчёт постоянных, вынесенных из цикла
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_width = int(width/2)
half_height = int(height/2)

# инструкция
print('Пробел - пауза, \n'
      '1 - добавить стрелочки, 2 - добавить HSV, 3 - окно HSV, Q/Esc - закрыть окна')
print('Кадров', length, 'Ширина', width, 'Высота', height, 'FPS', video_fps)
print('Запуск с', start_frame)

# Начало кода для прототипа.
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # номер кадра, с которого начнём
suc, prev = cap.read()  # захват кадра
if not suc:
    print('Картинки нет')
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Цикл покадровой итерации
go = True
end_cycle = None
sparse_flow = None
sparse_points = None
prev_sparse_points = None
sparse_flow_img = None

if width // step % 2 == 1:
    indent_w = width % step / 2
else:
    indent_w = (width % step + step) / 2
if height // step % 2 == 1:
    indent_h = height % step / 2
else:
    indent_h = (height % step + step) / 2
points_grid = np.mgrid[indent_w:width:step, indent_h:height:step].astype(int)  # (2, 11, 19)
points = []
for x, y in zip(points_grid[0].flatten(), points_grid[1].flatten()):
    points.append([x, y])
points = np.array(points).astype(np.float32).reshape(-1, 2)

while go:
    # начальное время для расчёта FPS
    start_cycle = time.time_ns()  # time.time()

    suc, img = cap.read()
    if not suc:
        cv2.waitKey(0)
        print("Спасибо за внимание")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    prev_sparse_points = sparse_points
    # считаем поток, только если его нужно отображать
    if add_sparse_flow or add_sparse_lamps:
        sparse_flow_img, sparse_flow, sparse_points = get_flow_lk(prev_gray, gray, points)
        # cv2.imshow('Sparse flow', sparse_flow_img)

    prev_gray = gray
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img_hsv[:, :, 1]
    saturation_hsv = cv2.merge([s, s, s, ])

    output_bgr = img

    # сложение слоёв для формирования картинки
    if add_sparse_flow:
        output_bgr = cv2.add(output_bgr, sparse_flow_img)
    if add_sparse_lamps and type(sparse_flow) is not None:
        output_bgr = cv2.add(output_bgr, draw_sparse_lamps(output_bgr.shape, sparse_flow, sparse_points))
    if show_lamps and type(sparse_flow) is not None:
        cv2.imshow('lamps', draw_sparse_lamps(output_bgr.shape, sparse_flow, sparse_points))

    # Конец кода для прототипа. Конечное время вычислений, большинство из которых планируется делать на устройстве
    # Текстовая информация в углу
    cv2.putText(output_bgr, f"{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} кадр", (20, 70), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(output_bgr, f"{cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.3f} секунд", (20, 110),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # разметка кадра
    output_bgr = cv2.add(output_bgr, draw_grid((height, width), 20,
                                               colored_cross=True, viewing_angle_rect=True, cross=True,
                                               grid=False, blinds=True))

    # Управление с клавиатуры
    key = cv2.waitKey(1)
    if key == ord(' '):
        while True:
            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            if (key == ord('q')) or (key == ord('й')) or (key == 27):
                go = False
                break
    if key == ord('1'):
        add_sparse_flow = not add_sparse_flow
    if key == ord('2'):
        add_sparse_lamps = not add_sparse_lamps
    if key == ord('3'):
        show_lamps = not show_lamps
    if key == ord('4'):
        draw_bad_flow = not draw_bad_flow
    if (key == ord('q')) or (key == ord('й')) or (key == 27):
        break

    end_cycle = time.time_ns()  # time.time()

    # Подсчёт FPS
    end_calculations = time.time_ns()  # time.time()
    fps = 1 / ((end_calculations - start_cycle)/(10**9))
    if end_cycle is None:
        end_cycle = start_cycle + 0.01
    if fps > video_fps:
        calc_time = (end_cycle - start_cycle)/(10**9)
        # time.sleep(1 / (video_fps + calc_time))
        end_calculations = time.time_ns()  # time.time()
        if calc_time != 0:
            fps = 1 / calc_time
        else:
            fps = 99

    # Отображение fps
    # print(f"{fps:.2f} FPS")
    cv2.putText(output_bgr, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 255, 0), 2)
    # Вывод результирующего кадра
    cv2.imshow('flow', output_bgr)

cap.release()
cv2.destroyAllWindows()
