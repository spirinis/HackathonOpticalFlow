import numpy as np
import cv2
import time
import sys

# МЕНЮ НАСТРОЙКИ
add_flow = False
add_hsv = False
show_hsv = False

viewing_angle_req = 60
frame_queue = []

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
        cv2.polylines(img_bgr, x_lines, False, (100, 100, 100), 1)
        cv2.polylines(img_bgr, y_lines, False, (100, 100, 100), 1)
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
        # считаем синий прямоугольник угла обзора
        if viewing_angle > viewing_angle_req:  # если он нужен
            width_res = round(width * viewing_angle_req / viewing_angle)
            height_res = round(height * viewing_angle_req / viewing_angle)
            rect_x0 = round((width - width_res) / 2)
            rect_y0 = round((height - height_res) / 2)
            rect_x1 = rect_x0 + width_res
            rect_y1 = rect_y0 + height_res
        else:  # прячем за границы, если не нужен
            rect_x0 = -1
            rect_y0 = -1
            rect_x1 = width
            rect_y1 = height
        cv2.rectangle(img_bgr, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 0, 0), 1)  # прямоугольник обзора 60 градусов

    return img_bgr


def draw_hsv(flow):
    """Возвращает слой с радугой в цветовом пространстве hsv,
    где цвет - направление вектора, интенсивность - длинна вектора"""
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


def calculate_optical_flow(prev, next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                           poly_n=5, poly_sigma=1.2, flags=0):
    """
    Вычисляет оптический поток.

    args:
    - prev : Предыдущий кадр в оттенках серого.
    - next : Текущий кадр в оттенках серого.
    - flow (Optional): Массив для сохранения результата потока, или None для его автоматического создания.
    - pyr_scale (float): Коэффициент уменьшения изображения в пирамиде.
    - levels (int): Количество уровней пирамиды.
    - winsize (int): Размер окна для усреднения потока (больше — более грубый результат).
    - iterations (int): Количество итераций на каждом уровне пирамиды.
    - poly_n (int): Размер области для аппроксимации полинома. Чем больше значение, тем грубее сглаживание.
    - poly_sigma (float): Стандартное отклонение Гауссова ядра для аппроксимации полинома.
    - flags (int): Флаги для метода.
    return:
        Двумерный массив (векторы потока) размером `(h, w, 2)`, где h — высота, w — ширина,
        и два канала представляют смещения по осям X и Y соответственно.
    """
    flow = cv2.calcOpticalFlowFarneback(prev=prev,
                                        next=next,
                                        flow=flow,
                                        pyr_scale=pyr_scale,
                                        levels=levels ,
                                        winsize=winsize,
                                        iterations=iterations,
                                        poly_n=poly_n,
                                        poly_sigma=poly_sigma,
                                        flags=flags)
    return flow


def draw_contours(img_grey):
    thresh = 100
    ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    # cv2.imshow('contours', contours_frame)
    contours_, hierarchy_ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_long, contours_short = [], []
    for cont in contours_:
        (contours_long, contours_short)[len(cont) < 10].append(cont)

    # contours1, contours2 = filter(lambda i: len(i) > 6, contours1)
    h, w = img_grey.shape[:2]
    img_contours = np.zeros((h, w, 3), np.uint8)
    cv2.drawContours(img_contours, contours_long, -1, (255, 255, 255), 1)
    cv2.drawContours(img_contours, contours_short, -1, (0, 0, 255), 1)

    return img_contours, contours_, hierarchy_


match 1:  # СЮДА МЕНЯТЬ
    case 0:
        cap = cv2.VideoCapture(0)
        start_frame = 100  #
        viewing_angle = 361
    case 1:
        cap = cv2.VideoCapture("./PERFECT_BANDO_FPV_FREESTYLE.mp4")
        # будка с дыркой 400 620 640 дом 1000 1130 1200 завод 1440 1550 1600 1900 дверь 1990 ангар 2400 2500 крыша 2875
        start_frame = 1300
        viewing_angle = 155
    case 2:
        cap = cv2.VideoCapture("./Полёт д1.mp4")
        start_frame = 0  #
        viewing_angle = 36
    case _:
        viewing_angle = 361
        print("\\(-_-)/")
        sys.exit()
# расчёт постоянных, вынесенных из цикла
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_width = int(width/2)
half_height = int(height/2)

# проецировать векторы на ось из-за кадра к точке направления
# инструкция
print('Пробел - пауза, 1 - добавить стрелочки, 2 - добавить HSV, 3 - окно HSV, Q/Esc - закрыть окна')
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
start_video = time.time()
while go:
    # начальное время для расчёта FPS
    start_cycle = time.time_ns()  # time.time()

    suc, img = cap.read()
    if not suc:
        end_video = time.time()
        cv2.waitKey(0)
        print(int(end_video - start_video), "секунд показа")
        print("Спасибо за внимание")
        break

    # очередь предыдущих кадров
    if not len(frame_queue) > 5:
        frame_queue.append((img, int(cap.get(cv2.CAP_PROP_POS_FRAMES))))
    else:
        frame_queue.pop(0)
        frame_queue.append((img, int(cap.get(cv2.CAP_PROP_POS_FRAMES))))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Сравнение кадров попиксельно
    # print(f"{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} кадр", abs(sum(sum(prev_gray - gray))))

    # считаем поток, только если его нужно отображать (буст)old
    if add_flow or add_hsv or show_hsv:
        flow =calculate_optical_flow(prev = prev_gray, next = gray)

    prev_gray = gray

    output_bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # сложение слоёв для формирования картинки
    if add_flow:
        output_bgr_gray = cv2.add(output_bgr_gray, draw_flow(output_bgr_gray.shape[:2], flow))
    if add_hsv:
        output_bgr_gray = cv2.add(output_bgr_gray, draw_hsv(flow))
    if show_hsv:
        cv2.imshow('flow HSV', cv2.add(draw_hsv(flow), draw_grid((height, width), colored_cross=True, cross=True,)))

    contours_frame, contours, hierarchy = draw_contours(gray)
    cv2.imshow('contours', contours_frame)

    # Конец кода для прототипа. Конечное время вычислений, большинство из которых планируется делать на устройстве
    # Текстовая информация в углу
    cv2.putText(output_bgr_gray, f"{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} кадр", (20, 70), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(output_bgr_gray, f"{cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.3f} секунд", (20, 110),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_bgr_gray, f"{time.time() - start_video:.3f} с время показа", (20, 150),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

    # разметка кадра
    output_bgr_gray = cv2.add(output_bgr_gray, draw_grid((height, width), 20,
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
        add_flow = not add_flow
    if key == ord('2'):
        add_hsv = not add_hsv
    if key == ord('3'):
        show_hsv = not show_hsv
        # print('HSV flow visualization is', ['off', 'on'][show_hsv])
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

    # print(f"{fps:.2f} FPS")
    cv2.putText(output_bgr_gray, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                (0, 255, 0), 2)
    # Вывод результирующего кадра
    cv2.imshow('flow', output_bgr_gray)
    # cv2.resizeWindow('flow', 900, 900)
cap.release()
cv2.destroyAllWindows()
