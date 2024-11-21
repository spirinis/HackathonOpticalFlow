import json
import time
import sys
import logging.config
import logging

import numpy as np
import cv2

# МЕНЮ НАСТРОЙКИ
add_sparse_flow = True  # Нарисует вектора прошедших фильтрацию точек
add_sparse_lamps = True  # Подсветит опасные точки
show_lamps = False  # Откроет дополнительное окно только с опасными точками
draw_bad_flow = True  # Отобразит отсеянные фильтром точки и вектора
start_frame = 0  # можно посмотреть конкретный момент
step = 30  # Установка плотности точек. Расстояние между точками. Меняется в зависимости от производительности железа
path_to_video = 'videos/'
match 5:  # Выбор видео
    case 0:
        video_name = "Стены_вокруг_куст_стена_на_пути9"
    case 1:
        video_name = "Тёмный_коридор_арка_куст"
    case 2:
        video_name = "Резкий_поворот_кусты_стена_разбился"
    case 3:
        video_name = "Тёмный_коридор_колонны"
    case 4:
        video_name = "здания_дверь_колонны_перекрытия"
    case 5:
        video_name = "FPV_FREESTYLE_очень_сложные_движения"
    case _:
        print("\\(-_-)/")
        sys.exit()

with open("logging.conf") as file:
    log_config = json.load(file)
logging.config.dictConfig(log_config)
# LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
# stream_handler = logging.StreamHandler()
# stream_handler.setLevel(logging.INFO)
# file_handler = logging.FileHandler(filename="logs/pathfinder_viewer.log", mode="w")
# file_handler.setLevel(logging.INFO)
# logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler], format=LOG_FORMAT)
logging.info('Video file %s%s.mp4' % (path_to_video, video_name))
cap = cv2.VideoCapture(f'{path_to_video}{video_name}.mp4')

viewing_angle = 155
viewing_angle_req = 60


def draw_flow(img_shape: tuple, flow: np.ndarray, step_: int = 14) -> np.ndarray:
    """
    Возвращает слой с векторами движения пикселей
    :param img_shape: (высота, ширина) слоя
    :param flow:
    :param step_: шаг в пикселях между измеряемыми точками в потоке
    :return: bgr изображение
    """
    h, w = img_shape
    img_bgr = np.zeros((h, w, 3), np.uint8)
    y, x = np.mgrid[step_ / 2:h:step_, step_ / 2:w:step_].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # сборка массива по столбцам и строкам, на выходе размер (999, 2, 2)
    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)  # математическое округление

    cv2.polylines(img_bgr, lines, False, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:  # добавление начальных точек векторов
        cv2.circle(img_bgr, center=(x1, y1), radius=1, color=(0, 255, 0),  thickness=-1)

    return img_bgr


def draw_grid(img_shape: tuple, step_: int = 20, colored_cross: bool = False, viewing_angle_rect: bool = False,
              cross: bool = False, grid: bool = False, blinds: bool = False) -> np.ndarray:
    """Возвращает слой с разнообразной разметкой кадра.
    :param img_shape: (высота, ширина) слоя
    :param step_: Шаг в пикселях между измеряемыми точками в потоке
    :param colored_cross: Флаг для креста цветов
    :param viewing_angle_rect: Флаг для прямоугольника угла обзора
    :param cross: Флаг для центрального креста
    :param grid: Флаг для отображения сетки
    :param blinds: Флаг для отображения неинформативных частей кадра
    :return: bgr изображение
    """
    h, w = img_shape
    img_bgr = np.zeros((h, w, 3), np.uint8)

    if grid:
        # BGR сетка пикселей, чтобы понимать, куда что рисовать
        x_lines = np.int32([[[i, 0], [i, h]] for i in range(step_, w, step_)])
        y_lines = np.int32([[[0, i], [w, i]] for i in range(step_, h, step_)])
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
        # выделяет зону, где видны подвижные винты
        cv2.rectangle(img_bgr, (620, height-200), (-1, height), (0, 0, 255,), 1)
        cv2.rectangle(img_bgr, (width-620, height-200), (width, height), (0, 0, 255), 1)
    if viewing_angle_rect:
        # рисует синий прямоугольник угла обзора в 60 градусов
        width_res = round(width * viewing_angle_req / viewing_angle)
        height_res = round(height * viewing_angle_req / viewing_angle)
        rect_x0 = round((width - width_res) / 2)
        rect_y0 = round((height - height_res) / 2)
        rect_x1 = rect_x0 + width_res
        rect_y1 = rect_y0 + height_res
        cv2.rectangle(img_bgr, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 0, 0), 3)

    return img_bgr


def draw_hsv(flow_: np.ndarray) -> np.ndarray:
    """Возвращает слой с векторами в цветовом пространстве hsv,
    где цвет - направление вектора, интенсивность - длинна вектора
    :param flow_: Объект оптического потока.
    :return: Кадр hsv представления векторов."""
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


def get_flow_lk(img1: np.ndarray, img2: np.ndarray, points_: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Рассчитывает оптический поток по методу Лукаса-Канаде
    :param img1: предыдущее изображение
    :param img2: текущее изображение
    :param points_: точки измерений
    :return:
    (frame_layer: изображение с потоком, flow: объект потока, points_: точки измерений, прошедшие фильтрацию)
    """
    frame_layer = np.zeros((height, width, 3), np.uint8)
    win_size = (45, 45)
    # формат next_pts: (-1, 2), (-1, 1), (-1, 1)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(img2, img1, points_, None, winSize=win_size, maxLevel=2,
                                                     criteria=(
                                                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    flow_ = next_pts - points_
    fx, fy = flow_[:, 0], flow_[:, 1]
    x, y = points_[:, 0], points_[:, 1]
    ang = np.arctan2(fy, fx)
    modulus = np.sqrt(fx * fx + fy * fy)
    modulus_middle = np.sqrt((half_width - x)**2 + (half_height - y)**2)
    # выравнивание модулей векторов
    modulus = modulus / (5+np.sqrt(modulus_middle)) * 30
    fx = modulus * np.cos(ang)
    fy = modulus * np.sin(ang)
    next_pts = np.vstack([x+fx, y+fy]).T
    next_pts = np.int32(next_pts + 0.5)  # математическое округление
    points_ = np.int32(points_ + 0.5)
    # фильтрация ошибок
    mask = (np.median(modulus) * 1.0 < modulus) & (modulus < np.percentile(modulus, 99))
    mask_inv = ~mask
    points_, points_bad = points_[mask], points_[mask_inv]
    next_pts, nextPts_bad = next_pts[mask], next_pts[mask_inv]
    # отрисовка
    flow = next_pts - points_
    lines = np.concatenate((points_, next_pts), axis=1)
    rlines = lines.reshape(-1, 2, 2)
    _ = cv2.polylines(frame_layer, rlines, False, (0, 0, 255))
    # начала векторов
    for x1, y1, _x2, _y2 in lines:
        cv2.circle(frame_layer, center=(x1, y1), radius=1, color=(255, 0, 255), thickness=1)

    if draw_bad_flow:
        lines_bad = np.concatenate((points_bad, nextPts_bad), axis=1)
        rlines_bad = lines_bad.reshape(-1, 2, 2)
        _ = cv2.polylines(frame_layer, rlines_bad, False, (255, 255, 0,))
        for x1, y1, _x2, _y2 in lines_bad:
            cv2.circle(frame_layer, center=(x1, y1), radius=1, color=(255, 255, 0), thickness=1)

    return frame_layer, flow, points_


def draw_sparse_lamps(flow_: np.ndarray, points_: np.ndarray) -> np.ndarray:
    """Возвращает слой с препятствиями
    :param flow_: объект потока
    :param points_: точки измерений
    :return: BGR изображение препятствий
    """
    logging.info('Функция %s' % __name__)
    try:
        fx, fy = flow_[:, 0], flow_[:, 1]
    except TypeError as e:
        print(type(flow_))
        print(f'flow_ = {flow_}')
        logging.error('Ошибка %s' % e)
        raise TypeError
    ang = np.arctan2(fy, fx) + np.pi
    modulus = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((height, width, 3), np.uint8)
    for (x, y), a, m in zip(points_, ang, modulus):
        hsv[y, x, 0] = 0
        hsv[y, x, 1] = 255
        hsv[y, x, 2] = np.minimum(50 + m * 2, 255)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, y, in points_:
        cv2.circle(bgr, center=(x, y), radius=6, color=(int(bgr[y, x, 0]), int(bgr[y, x, 1]), int(bgr[y, x, 2])),
                   thickness=-1)
    return bgr


# расчёт постоянных, вынесенных из цикла
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# инструкция
print('Пробел - пауза, \n'
      '1 - отобразить вектора, 2 - отобразить препятствия, 3 - окно с препятствия,'
      '4 - отобразить отфильтрованные вектора, Q/Esc - закрыть окна')
print(f'Кадров {length} Ширина {width} Высота {height} FPS {video_fps}')
print(f'Запуск с {start_frame}')

# Начало кода для прототипа.
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # номер кадра, с которого начнём
suc, prev = cap.read()  # захват кадра
if not suc:
    print('Картинки нет')
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

go = True
end_cycle = None
sparse_flow = None
sparse_points = None
prev_sparse_points = None
sparse_flow_img = None
half_width = int(width/2)
half_height = int(height/2)

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

# Цикл покадровой итерации
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

    prev_gray = gray
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img_hsv[:, :, 1]
    saturation_hsv = cv2.merge([s, s, s, ])

    output_bgr = img

    # сложение слоёв для формирования картинки
    if add_sparse_flow:
        output_bgr = cv2.add(output_bgr, sparse_flow_img)
    if add_sparse_lamps and type(sparse_flow) is not None:
        output_bgr = cv2.add(output_bgr, draw_sparse_lamps(sparse_flow, sparse_points))
    if show_lamps and type(sparse_flow) is not None:
        cv2.imshow('lamps', draw_sparse_lamps(sparse_flow, sparse_points))

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
    if key == ord(' '):  # пауза
        while True:
            key = cv2.waitKey(1)
            if key == ord(' '):
                break
            if (key == ord('q')) or (key == ord('й')) or (key == 27):
                go = False
                break
    if key == ord('1'):  # вектора
        add_sparse_flow = not add_sparse_flow
    if key == ord('2'):  # препятствия
        add_sparse_lamps = not add_sparse_lamps
    if key == ord('3'):  # окно с препятствиями
        if not add_sparse_flow:
            add_sparse_flow = not add_sparse_flow
        show_lamps = not show_lamps
    if key == ord('4'):  # отфильтрованные вектора
        if not add_sparse_flow:
            add_sparse_flow = not add_sparse_flow
        draw_bad_flow = not draw_bad_flow
    if (key == ord('q')) or (key == ord('й')) or (key == 27):  # закрыть окна
        break

    end_cycle = time.time_ns()

    # Подсчёт FPS
    end_calculations = time.time_ns()
    fps = 1 / ((end_calculations - start_cycle)/(10**9))
    if end_cycle is None:
        end_cycle = start_cycle + 0.01
    if fps > video_fps:
        calc_time = (end_cycle - start_cycle)/(10**9)
        end_calculations = time.time_ns()
        if calc_time != 0:
            fps = 1 / calc_time
        else:
            fps = 99

    # Отображение fps
    cv2.putText(output_bgr, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 255, 0), 2)
    # Вывод результирующего кадра
    cv2.imshow('flow', output_bgr)

cap.release()
cv2.destroyAllWindows()
