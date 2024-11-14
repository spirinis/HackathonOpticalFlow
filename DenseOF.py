import numpy as np
import cv2
import time
import sys

# МЕНЮ НАСТРОЙКИ
add_flow = False
add_sparse_flow = True
add_hsv = False
show_hsv = False
show_contours = False
add_sparse_hsv = True
show_lamps = False
show_colored = 1
# будка с дыркой 400 620 640 дом 1000 1130 1200 завод 1440 1550 1600 1900 дверь 1990 ангар 2400 2500 крыша 2875
start_frame = 0  # 1650 колонны дверь

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
"""

# проецировать векторы на ось из-за кадра к точке направления
# серый - минимальная разница между ргб, зелёный - минимальная между г
# делить по цветам на основе цветов в диагоналях 3 и 4 четверти


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
        cv2.rectangle(img_bgr, (rect_x0, rect_y0), (rect_x1, rect_y1), (255, 0, 0), 3)  # прямоугольник обзора 60 градусов

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


def get_flow_lk(img1, img2, step=100):
    # imgs = get_and_open_image((1440, 1441, 1460, 1461))
    # img1 = imgs[0]
    # img2 = imgs[1]
    # step=100
    frame_layer = np.zeros((height, width, 3), np.uint8)
    if width // step % 2 == 1:
        indent_w = width % step / 2
    else:
        indent_w = (width % step + step) / 2
    if height // step % 2 == 1:
        indent_h = height % step / 2
    else:
        indent_h = (height % step + step) / 2
    points_grid = np.mgrid[indent_w:width:step, indent_h:height:step].astype(int)  # (2, 11, 19)
    # points = points_grid.reshape(2, -1)
    # points = points_grid.reshape(-1, 2).astype(np.float32)
    points = []
    for x, y in zip(points_grid[0].flatten(), points_grid[1].flatten()):
        points.append([x, y])
    points = np.array(points).astype(np.float32).reshape(-1, 2)
    winSize = (45, 45)
    # (-1, 2), (-1, 1), (-1, 1)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(img2, img1, points, None, winSize=winSize, maxLevel=2,
                                                    criteria=(
                                                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # nextPtsr, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, winSize=winSize, maxLevel=2,
    #                                                  criteria=(
    #                                                     cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # d = abs(nextPts - nextPtsr).reshape(-1, 2).max(-1)
    # mask = np.less(d, 1)
    # nextPts = nextPts[mask]
    # points = points[mask]
    # good = d < 1
    flow_ = nextPts - points
    fx, fy = flow_[:, 0], flow_[:, 1]
    x, y = points[:, 0], points[:, 1]
    ang = np.arctan2(fy, fx)  # 4.71238898038469 = np.radians(270)
    ang_middle = (np.arctan2(half_height - y, half_width - x))
    modulus = np.sqrt(fx * fx + fy * fy)
    modulus_middle = np.sqrt((half_width - x)**2 + (half_height - y)**2)

    modulus = modulus / (5+np.sqrt(modulus_middle)) * 30  # (modulus_middle/((half_width + half_height-2*step))) / 3
    fx = modulus * np.cos(ang) #+ x modulus *
    fy = modulus * np.sin(ang) #+ y
    nextPts = np.vstack([x+fx, y+fy]).T
    # flow_ = nextPts - points

    # ang += np.pi
    # ang_middle += np.pi #   * 180 / np.pi
    def fang(x, y):
        ar = np.arctan2(half_height - y, half_width - x) + np.pi
        print(ar)
        print(ar * 180 / np.pi)

    # nextPts = nextPts.reshape(points_grid.shape)  # (-1, 2) -> (2, 11, 19)
    # nextPts00 = nextPts.reshape(-1, 2)
    nextPts0 = nextPts
    nextPts = np.int32(nextPts + 0.5)  # математическое округление
    points = np.int32(points + 0.5)
    # fm = []
    # for (x0, y0), (x1, y1) in zip(points, nextPts):
    #     if (x0 != x1) or (y0 != y1):
    #         fm.append(True)
    #     else:
    #         fm.append(False)
    # points = points[fm]
    # nextPts = nextPts[fm]
    mask = np.greater(modulus, np.median(modulus) * 1.2)
    # op.open_images([img, draw_sparse_lamps(img.shape[:2], flow_m, points)], 'img flow')

    # mask = np.not_equal(points, nextPts)
    # mask = mask.T  # mask.reshape(2, -1)
    # mask = (mask[0] | mask[1]) #& np.less(abs(ang_middle - ang), 90*np.pi / 180)
    # mask = np.where((modulus > np.mean(modulus)*0.9), mask, np.False_)  # + грубее, - чувствительнее
    # # np.where(modulus < 2 * np.mean(modulus), mask, np.False_)
    mask_inv = ~mask
    points, points_bad = points[mask], points[mask_inv]
    nextPts, nextPts_bad = nextPts[mask], nextPts[mask_inv]
    modulus, modulus_bad = modulus[mask], modulus[mask_inv]
    ang, ang_bad = ang[mask], ang[mask_inv]

    flow = nextPts - points
    lines = np.concatenate((points, nextPts), axis=1) #  lines.shape = (-1, 4)
    rlines = lines.reshape(-1, 2, 2)
    # flines = []
    # for x0, y0, x1, y1 in lines:
    #     if (x0 != x1) and (y0 != y1):
    #         flines.append([x0, y0, x1, y1])
    # flines = np.array(flines)
    # rflines = flines.reshape(-1, 2, 2)
    _ = cv2.polylines(frame_layer, rlines, False, (0, 0, 255))
    # точки н-нада?
    for x1, y1, _x2, _y2 in lines:
        cv2.circle(frame_layer, center=(x1, y1), radius=1, color=(255, 0, 255), thickness=1)
    # flow = np.concatenate((flow, points), axis=1)

    lines_bad = np.concatenate((points_bad, nextPts_bad), axis=1)
    rlines_bad = lines_bad.reshape(-1, 2, 2)
    _ = cv2.polylines(frame_layer, rlines_bad, False, (255, 255, 0,))
    for x1, y1, _x2, _y2 in lines_bad:
        cv2.circle(frame_layer, center=(x1, y1), radius=1, color=(255, 255, 0), thickness=1)

    return frame_layer, flow, points


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
    h, w = img_shape[:2]
    try:
        fx, fy = flow_[:, 0], flow_[:, 1]
    except TypeError:
        print(type(flow_))
        print('flow_ =', flow_)
    ang = np.arctan2(fy, fx) + np.pi
    modulus = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    for (x, y), a, m in zip(points, ang, modulus):
        hsv[y, x, 0] = 0
        hsv[y, x, 1] = 255
        hsv[y, x, 2] = np.minimum(50 + m * 2, 255)
    # hsv[..., 0] = ang * (180 / np.pi / 2)
    # hsv[..., 1] = 255
    # hsv[..., 2] = np.minimum(modulus * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, y, in points:
        cv2.circle(bgr, center=(x, y), radius=6, color=(int(bgr[y, x, 0]), int(bgr[y, x, 1]), int(bgr[y, x, 2])),
                   thickness=-1)
    # фильтровать по соседям и предидущему кадру с соседями
    return bgr



greys_count = None


def draw_contours(img_grey, contour_length=150):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # прикольно но долго но может пригодиться
    # def kmeans_color_quantization(image, clusters=8, rounds=1):
    #     h, w = image.shape[:2]
    #     samples = np.zeros([h * w, 3], dtype=np.float32)
    #     count = 0
    #
    #     for x in range(h):
    #         for y in range(w):
    #             samples[count] = image[x][y]
    #             count += 1
    #
    #     compactness, labels, centers = cv2.kmeans(samples,
    #                                               clusters,
    #                                               None,
    #                                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
    #                                               rounds,
    #                                               cv2.KMEANS_RANDOM_CENTERS)
    #
    #     centers = np.uint8(centers)
    #     res = centers[labels.flatten()]  # .flatten преобразует массив NumPy в одномерный
    #     return res.reshape(image.shape+(3,))

    # может, имеет смысл размывать мелкие детали
    # img_grey = cv2.GaussianBlur(img_grey, (7, 7), 0)

    # Упрощаем цвета до нескольких. Можно младшие биты цвета отбросить или:
    # 255 // div + 1 цветов на выходе
    # ну ладно...
    def calc_div(colors_count):
        """Посчитает div по colors_count"""
        colors_divs = {}
        for div_ in range(1, 256):
            colors_divs[str((lambda d: 255 // d)(div_))] = div_ + 1
        try:
            return colors_divs[str(colors_count)]  # = div
        except KeyError:
            return 'Математика не сошлась, жёсткая штука. Не делится'

    def greys_counter(div, step=1):
        """Посчитает и покажет цвета результата разбиения и их количество, в зависимости от div"""
        res = []
        for p in range(0, 255, step):
            res.append(p // div * div)
        # max_ = max(res)
        # res = sorted(set(list(map(lambda x: x + (255 - max_) // 2, res))))
        res = sorted(set(res))
        return res, len(res)

    # {'255': 2, '127': 3, '85': 4, '63': 5, '51': 6, '42': 7, '36': 8, '31': 9, '28': 10, '25': 11, '23': 12, '21': 13,
    # '19': 14, '18': 15, '17': 16, '15': 18, '14': 19, '13': 20, '12': 22, '11': 24, '10': 26, '9': 29, '8': 32,
    # '7': 37, '6': 43, '5': 52, '4': 64, '3': 86, '2': 128, '1': 256}
    #
    div = 63  # 64 80 28
    global greys_count
    if greys_count is None:
        greys_count = greys_counter(div)
        print('Оттенки серого', greys_count)
    img_div = img_grey // div * div  # получение кратного div цвета
    # смещение середины интервала цветов с шагом div от 0 в середину цветового пространства
    # step = len(img_div) // 5
    # max_ = max(np.concatenate((img_div[step*1], img_div[step*2], img_div[step*3], img_div[step*4], )))
    # img_div += (255 - max_) // 2  # С грозой
    # h14 = cv2.calcHist([i14], [1], None, [256], [0, 256])
    cv2.imshow('thresh_img', img_div)

    global thresh_img  # отладка
    # thresh_img = kmeans_color_quantization(img_grey)

    # ret, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
    contours_, hierarchy_ = [], []
    for color in greys_count[0]:
        ret, thresh_img = cv2.threshold(img_div, color, 255, cv2.THRESH_BINARY)
        contours_i, hierarchy_i = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_ += contours_i
        hierarchy_ = hierarchy_i
        '''cv2.THRESH_BINARY — это тип порога, который преобразует все пиксели со значением больше или равным порогу в максимальное значение (обычно 255), а остальные пиксели — в 0.
        Это наиболее распространённый тип порога для бинаризации изображений.
        cv2.THRESH_BINARY_INV — инвертированный двоичный порог. Пиксели со значениями меньше или равными порогу становятся максимальными, а остальные — минимальными.
        cv2.THRESH_TRUNC — этот тип порога обрезает значения пикселей до порогового значения. Все пиксели, которые больше или равны порогу, становятся равными этому порогу.
        cv2.THRESH_TOZERO — все пиксели с интенсивностью выше порога становятся нулевыми, а все остальные остаются неизменными.
        cv2.THRESH_TOZERO_INV — инвертирует cv2.THRESH_TOZERO, делая все пиксели ниже порога нулевыми.'''

    # thresh_img = img_div

    # contours_, hierarchy_ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    """RETR_EXTERNAL извлекает только крайние внешние контуры. Он устанавливает hierarchy[i][2]=hierarchy[i][3]=-1
    для всех контуров.
    RETR_LIST извлекает все контуры без установления каких-либо иерархических связей.
    RETR_CCOMP извлекает все контуры и организует их в двухуровневую иерархию. На верхнем уровне указаны внешние границы
    компонентов. На втором уровне указаны границы отверстий. Если внутри отверстия подключенного компонента есть другой
    контур, он по-прежнему размещается на верхнем уровне.
    RETR_TREE извлекает все контуры и восстанавливает полную иерархию вложенных контуров.
    RETR_FLOODFILL ???
    
    CHAIN_APPROX_NONE: выдаст все точки при обходе
    CHAIN_APPROX_SIMPLE: аппроксимирует отрезками
    CHAIN_APPROX_TC89_L1: ???
    CHAIN_APPROX_TC89_KCOS: ???"""
    # фильтруем контура на длинные и короткие
    contours_long, contours_short = [], []
    for cont in contours_:
        len_ = len(cont)
        # (contours_long, contours_short)[len_ < contour_length].append(cont)
        if len_ > contour_length:
            contours_long.append(cont)
        else:
            if len_ > contour_length*0.8:
                contours_short.append(cont)

    # Создаём слой с контурами
    h, w = img_grey.shape[:2]
    img_contours = np.zeros((h, w, 3), np.uint8)
    cv2.drawContours(img_contours, contours_long, -1, (255, 255, 255), 1)
    cv2.drawContours(img_contours, contours_short, -1, (0, 0, 255), 1)

    return img_contours, contours_, hierarchy_


match 0:  # СЮДА МЕНЯТЬ
    case 0:
        cap = cv2.VideoCapture("./Полёт_со_скольжением.mp4")
        start_frame = 0  #
        viewing_angle = 155
    case 1:
        cap = cv2.VideoCapture("./PERFECT_BANDO_FPV_FREESTYLE.mp4")
        # будка с дыркой 400 620 640 дом 1000 1130 1200 завод 1440 1550 1600 1900 дверь 1990 ангар 2400 2500 крыша 2875
        start_frame = start_frame
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

# инструкция
print('Пробел - пауза, ~ - 9 режимов отображения главного слоя, переключаемых нажатиями(серый,RGB,R,G,B,HSV,H,S,V),\n'
      '1 - добавить стрелочки, 2 - добавить HSV, 3 - окно HSV, Q/Esc - закрыть окна\n'
      '# из show_colored_iterable можно убрать ненужные')
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
# 0 сер, 1 цвет, 2R, 3G, 4B, 5 HSV, 6 H оттенок, 7 S насыщенность, 8 V яркость,
show_colored_iterable = [0, 1, 6, 7, 8]  # [0, 1, 2, 3, 4, 5, 6, 7, 8] [1, 6, 7, 8]
iterator = iter(show_colored_iterable)
# начальное время для расчёта показа (убрать)
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

    # считаем поток, только если его нужно отображать
    flow = None
    sparse_flow = None
    sparse_points = None
    sparse_flow_img = None
    if add_flow or add_hsv or show_hsv:
        flow = calculate_optical_flow(prev=prev_gray, next=gray)
    if add_sparse_flow or add_sparse_hsv:
        sparse_flow_img, sparse_flow, sparse_points = get_flow_lk(prev_gray, gray, step=30) # 300
        # cv2.imshow('Sparse flow', sparse_flow_img)

    prev_gray = gray
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = img_hsv[:, :, 1]
    saturation_hsv = cv2.merge([s, s, s, ])

    # переключение основного выходного изображения в зависимости от кнопки ~
    output_bgr = mode = None
    if show_colored == 0:
        output_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        mode = "Оттенки серого"
    elif show_colored == 1:
        output_bgr = img
        mode = "RGB"
    elif show_colored == 2:
        z = np.zeros((height, width, 1), np.uint8)
        output_bgr = cv2.merge([z, z, img[:, :, 2], ])  # Красный
        mode = "R"
    elif show_colored == 3:
        z = np.zeros((height, width, 1), np.uint8)
        output_bgr = cv2.merge([z, img[:, :, 1], z, ])  # Зелёный
        mode = "G"
    elif show_colored == 4:
        z = np.zeros((height, width, 1), np.uint8)
        output_bgr = cv2.merge([img[:, :, 0], z, z, ])  # Синий
        mode = "B"
    elif (show_colored == 5) or (show_colored == 6) or (show_colored == 7) or (show_colored == 8):
        if show_colored == 5:
            output_bgr = img_hsv
            mode = "HSV"
        elif show_colored == 6:
            h = img_hsv[:, :, 0]
            output_bgr = cv2.merge([h, h, h, ])
            # z = np.zeros((height, width, 1), np.uint8)
            # output_bgr = cv2.merge([z, z, img_hsv[:, :, 0], ])  # Hue оттенок
            mode = "H оттенок"
        elif show_colored == 7:
            output_bgr = saturation_hsv
            # z = np.zeros((height, width, 1), np.uint8)
            # output_bgr = cv2.merge([z, img_hsv[:, :, 1], z, ])  # Saturation насыщенность
            mode = "S насыщенность"
        elif show_colored == 8:
            v = img_hsv[:, :, 2]
            output_bgr = cv2.merge([v, v, v, ])  # Value яркость
            # z = np.zeros((height, width, 1), np.uint8)
            # output_bgr = cv2.merge([img_hsv[:, :, 2], z, z, ])  # Value яркость
            mode = "V яркость"

    # сложение слоёв для формирования картинки
    if add_flow:
        output_bgr = cv2.add(output_bgr, draw_flow(output_bgr.shape[:2], flow))
    if add_sparse_flow:
        output_bgr = cv2.add(output_bgr, sparse_flow_img)
    if add_hsv:
        output_bgr = cv2.add(output_bgr, draw_hsv(flow))
    if add_sparse_hsv and type(sparse_flow) is not None:
        output_bgr = cv2.add(output_bgr, draw_sparse_lamps(output_bgr.shape, sparse_flow, sparse_points))
    if show_lamps and type(sparse_flow) is not None:
        cv2.imshow('lamps', draw_sparse_lamps(output_bgr.shape, sparse_flow, sparse_points))
    if show_hsv:
        cv2.imshow('flow HSV', cv2.add(draw_hsv(flow), draw_grid((height, width), colored_cross=True, cross=True, )))
    if show_contours:
        contours_frame, contours, hierarchy = draw_contours(gray)  # saturation_hsv[:, :, 0])
        cv2.imshow('contours', contours_frame)

    # Конец кода для прототипа. Конечное время вычислений, большинство из которых планируется делать на устройстве
    # Текстовая информация в углу
    cv2.putText(output_bgr, f"{cap.get(cv2.CAP_PROP_POS_FRAMES):.0f} кадр", (20, 70), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(output_bgr, f"{cap.get(cv2.CAP_PROP_POS_MSEC) / 1000:.3f} секунд", (20, 110),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(output_bgr, mode, (20, 150), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 255, 0), 2)
    # cv2.putText(output_bgr, f"{time.time() - start_video:.3f} с время показа", (20, 190),
    #             cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)

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
    if (key == ord('`')) or (key == ord('ё')):
        # if show_colored == 8:
        #     show_colored = 0
        # else:
        #     show_colored += 1
        try:
            show_colored = next(iterator)
        except StopIteration:
            iterator = iter(show_colored_iterable)
            show_colored = next(iterator)
    if key == ord('1'):
        add_flow = not add_flow
    if key == ord('2'):
        add_hsv = not add_hsv
    if key == ord('3'):
        show_hsv = not show_hsv
        # print('HSV flow visualization is', ['off', 'on'][show_hsv])
    if key == ord('4'):
        show_contours = not show_contours
    if (key == ord('q')) or (key == ord('й')) or (key == 27):
        break
    if key == ord('5'):
        add_sparse_flow = not add_sparse_flow
    if key == ord('6'):
        add_sparse_hsv = not add_sparse_hsv
    if key == ord('7'):
        show_lamps = not show_lamps
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
    # Было бы удобно масштабировать окна, но некогда
    # for win in ['flow', 'flow HSV', ]:
    #     if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE):
    #         cv2.resizeWindow(win, 900, 500)
    #         print(win, cv2.getWindowProperty(win, cv2.WND_PROP_AUTOSIZE),
    #               cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE))

cap.release()
cv2.destroyAllWindows()
