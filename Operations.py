import time
from typing import List, SupportsInt

import numpy as np
import cv2


def get_and_open_image(pos: tuple[int, int] = (1000, 1800), to_open: bool = False,
                       source: str = "./PERFECT_BANDO_FPV_FREESTYLE.mp4", ) -> List[np.ndarray] | np.ndarray | None:
    """Вернёт и откроет при условии перечисленные кадры"""
    cap = cv2.VideoCapture(source)
    if not hasattr(pos, '__iter__'):
        pos = (pos,)
    images = []
    for p in pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, p)
        result, frame = cap.read()
        if result:
            images.append(frame)
        else:
            print(f'Потеряли кадр {p}')
            return
    if to_open:
        for p, image in zip(pos, images):
            cv2.imshow(str(p), image)
        while True:
            ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
            if ch == 27:  # Esc
                break
        cap.release()
        for p in pos:
            cv2.destroyWindow(str(p))
    return images if len(images) > 1 else images[0]


def resize_image(image: np.ndarray, des_w=100, des_h: int = None, interpolation=cv2.INTER_AREA) -> np.ndarray:
    """Масштабирует переданную картинку по ширине или в требуемый размер"""
    # соотношение сторон: ширина, делённая на ширину оригинала
    aspect_ratio = des_w / image.shape[1]
    if des_h is None:
        # желаемая высота: высота, умноженная на соотношение сторон
        des_h = int(image.shape[0] * aspect_ratio)
    dim = (des_w, des_h)  # итоговые размеры
    # Масштабируем картинку
    # Шикарная статья https://robocraft.ru/computervision/3956?ysclid=m2209mlw3o754377593
    resized_image = cv2.resize(image, dsize=dim, interpolation=interpolation)

    return resized_image


def open_to_compare_images(i):
    """Скрипт гауссового размытия и бинаризации"""
    g = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)

    b3 = cv2.GaussianBlur(g, (3, 3), 0)
    b7 = cv2.GaussianBlur(g, (7, 7), 0)

    ret, tg = cv2.threshold(g, 70, 255, cv2.THRESH_BINARY)
    ret, tb3 = cv2.threshold(b3, 70, 255, cv2.THRESH_BINARY)
    ret, tb7 = cv2.threshold(b7, 70, 255, cv2.THRESH_BINARY)

    cv2.imshow('tg', tg)
    cv2.imshow('tb3', tb3)
    cv2.imshow('tb7', tb7)
    while True:
        ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
        if ch == 27:  # Esc
            break
    cv2.destroyAllWindows()


def open_images(images, names='Name'):
    number = len(images)
    if not number < 10:
        images = (images,)
    names = names.split()
    if len(names) != number:
        names = [names[0]]
        for i in range(1, number):
            names.append(names[0] + str(i))
    for name, image in zip(names, images):
        cv2.imshow(name, image)
    while cv2.getWindowProperty(names[0], cv2.WND_PROP_VISIBLE) >= 1:
        ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
        if ch == 27:  # Esc
            break
    cv2.destroyAllWindows()


def mark_points(img, points, to_draw=True, to_return=False):
    """Отметит крестиком указанные точки на img, или на пустом изображении размерами img=[h, w]"""
    if not hasattr(img, 'shape'):
        h, w = img
    else:
        h, w = img.shape[:2]
    layer = np.zeros((h, w, 3), np.uint8)
    for x, y in points:
        assert (x - 7 > 0) and (y - 7 > 0)
        cv2.polylines(layer, np.int32([[[x - 7, y], [x + 7, y]], [[x, y - 7], [x, y + 7]]]), False, (0, 0, 255), 1)
    if to_draw:
        open_images(cv2.add(img, layer), 'fig')
    if to_return:
        return layer


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
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, y, in points:
        cv2.circle(bgr, center=(x, y), radius=6, color=(int(bgr[y, x, 0]), int(bgr[y, x, 1]), int(bgr[y, x, 2])),
                   thickness=-1)

    return bgr


class Plotter:
    """Строит интерактивный график в opencv"""
    def __init__(self, plot_width, plot_height, sample_buffer=None):
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0, 0)
        self.plot_canvas = np.ones((self.height, self.width, 3)) * 255
        self.ltime = 0
        self.plots = {}
        self.plot_t_last = {}
        self.margin_l = 10
        self.margin_r = 10
        self.margin_u = 10
        self.margin_d = 50
        self.sample_buffer = self.width if sample_buffer is None else sample_buffer

    def plot(self, val: SupportsInt, label: str = "plot") -> None:
        """Обновляет данные в графике"""
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0

        self.plots[label].append(int(val))
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            self.show_plot(label)
            # Выводит график через opencv imshow

    def show_plot(self, label: str, time_bool=False) -> None:
        """
        Выводит график в отдельном окне
        :param label: имя окна
        :param time_bool: нужна ли строка времени на графике
        """
        self.plot_canvas = np.zeros((self.height, self.width, 3)) * 255
        cv2.line(self.plot_canvas,
                 (self.margin_l, int((self.height - self.margin_d - self.margin_u) / 2) + self.margin_u), (
                 self.width - self.margin_r,
                 int((self.height - self.margin_d - self.margin_u) / 2) + self.margin_u), (0, 0, 255), 1)

        # Масштабирование графика по y в размер буфера
        scale_h_max = max(self.plots[label])
        scale_h_min = min(self.plots[label])
        scale_h_min = -scale_h_min if scale_h_min < 0 else scale_h_min
        scale_h = scale_h_max if scale_h_max > scale_h_min else scale_h_min
        scale_h = ((self.height - self.margin_d - self.margin_u) / 2) / scale_h if not scale_h == 0 else 0

        for (j, i) in enumerate(np.linspace(0, self.sample_buffer - 2, self.width - self.margin_l - self.margin_r)):
            i = int(i)
            cv2.line(self.plot_canvas,
                     (j + self.margin_l,
                      int((self.height - self.margin_d - self.margin_u) / 2 + self.margin_u -
                          self.plots[label][i] * scale_h)),
                     (j + self.margin_l,int((self.height - self.margin_d - self.margin_u) / 2 + self.margin_u -
                                            self.plots[label][i + 1] * scale_h)), self.color, 1)

        cv2.rectangle(self.plot_canvas, (self.margin_l, self.margin_u),
                      (self.width - self.margin_r, self.height - self.margin_d), (255, 255, 255), 1)
        if time_bool:  # строка времени на графике
            cv2.putText(self.plot_canvas,
                        f" {label} : {self.plots[label][-1]} , dt : {int((time.time() - self.plot_t_last[label]) * 1000)}ms",
                        (int(0), self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.circle(self.plot_canvas, (self.width - self.margin_r,
                                      int(self.margin_u + (self.height - self.margin_d - self.margin_u) / 2 -
                                          self.plots[label][-1] * scale_h)), 2, (0, 200, 200), -1)

        self.plot_t_last[label] = time.time()
        cv2.imshow(label, self.plot_canvas)
        while cv2.getWindowProperty(label, cv2.WND_PROP_VISIBLE) >= 1:
            ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
            if ch == 27:  # Esc
                break
        cv2.destroyAllWindows()


def draw_plot(values: list, label: str = 'graph') -> None:
    """Делает динамичный график не динамичным"""
    sample_buffer = len(values)-1
    p = Plotter(sample_buffer*2, 1080, sample_buffer=sample_buffer)

    for v in values:
        p.plot(v, label=label)


def color_hsv_division(img1) -> None:
    """скрипт для разбиения изображения по слоям и расчёта гистограмм"""
    img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    color = ('b', 'g', 'r')
    hists = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img1_hsv], [i], None, [256], [0, 256])
        hists.append(histr)
        # plt.plot(histr, color=col)
        # plt.xlim([0, 256])
    # plt.show()
    hist_h = hists[0]
    h = img1_hsv[:, :, 0]
    output_bgr = cv2.merge([h, h, h, ])
    z = np.zeros((h.shape[0], h.shape[1], 1), np.uint8)
    output_bgr = cv2.merge([z, h, z, ])
    open_images((img1, output_bgr), 'img1 output_bgr')


def change_format():
    """Поменяет разрешение видео"""
    import sys

    def rescale_frame(frame_input, percent=75):
        width = int(frame_input.shape[1] * percent / 100)
        height = int(frame_input.shape[0] * percent / 100)
        dim = (width, height)
        return cv2.resize(frame_input, dim, interpolation=cv2.INTER_AREA)

    cap = cv2.VideoCapture("./PERFECT_BANDO_FPV_FREESTYLE.mp4")

    if cap.isOpened():
        ret, frame = cap.read()
        rescaled_frame = rescale_frame(frame)
        (h, w) = rescaled_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter('Video_output.mp4', fourcc, 15.0, (w, h), True)
    else:
        print("Камера не открыта")
        print("\\(-_-)/")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()

        rescaled_frame = rescale_frame(frame)

        # запись результирующего изображения в файл
        writer.write(rescaled_frame)

        cv2.imshow("Output", rescaled_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    writer.release()
