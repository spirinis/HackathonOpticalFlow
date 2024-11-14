import numpy as np
import cv2
import time
from random import random
import math


def get_and_open_image(pos=(1000, 1800), to_open=False, source="./PERFECT_BANDO_FPV_FREESTYLE.mp4",):
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
            print('Потеряли кадр', p)
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


def resize_image(image, des_w=100, des_h=None, interpolation=cv2.INTER_AREA):
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
    """Просто скрипт"""
    # not (np.bitwise_xor(frame1, frame2).any())
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
            names.append(names[0] + str(i))  # да не class 'str'
    for name, image in zip(names, images):
        cv2.imshow(name, image)
    while cv2.getWindowProperty(names[0], cv2.WND_PROP_VISIBLE) >= 1:
        ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
        if ch == 27:  # Esc
            break
    cv2.destroyAllWindows()


def get_flow_lk(img1, img2, step=30):
    # imgs = get_and_open_image((1440, 1441, 1460, 1461))
    # img1 = imgs[0]
    # img2 = imgs[1]
    # step=100
    h, w = img1.shape[:2]
    frame_layer = np.zeros((h, w, 3), np.uint8)
    points_grid = np.mgrid[step / 2:w:step, step / 2:h:step].astype(int)  # (2, 11, 19)
    # points = points_grid.reshape(2, -1)
    # points = points_grid.reshape(-1, 2).astype(np.float32)
    points = []
    for x, y in zip(points_grid[0].flatten(), points_grid[1].flatten()):
        points.append([x, y])
    points = np.array(points).astype(np.float32).reshape(-1, 2)
    # (-1, 2), (-1, 1), (-1, 1)
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, winSize=(15, 15), maxLevel=2,
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # nextPtsr, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points, None, winSize=(15, 15), maxLevel=2,
    #                                                 criteria=(
    #                                                     cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # d = abs(nextPts - nextPtsr).reshape(-1, 2).max(-1)
    # good = d < 1
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
    # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    flow_ = nextPts - points
    fx, fy = flow_[:, 0], flow_[:, 1]
    ang = np.arctan2(fy, fx) + np.pi
    modulus = np.sqrt(fx * fx + fy * fy)

    mask = np.not_equal(points, nextPts)
    mask = mask.T  # mask.reshape(2, -1)
    mask = mask[0] | mask[1]
    # points = points[fm]
    # nextPts = nextPts[fm]
    mask = np.where(modulus < 2 * np.mean(modulus), mask, np.False_)
    points = points[mask]
    nextPts = nextPts[mask]
    modulus = modulus[mask]
    ang = ang[mask]
    lines = np.concatenate((points, nextPts), axis=1)
    rlines = lines.reshape(-1, 2, 2)
    # flines = []
    # for x0, y0, x1, y1 in lines:
    #     if (x0 != x1) and (y0 != y1):
    #         flines.append([x0, y0, x1, y1])
    # flines = np.array(flines)
    # rflines = flines.reshape(-1, 2, 2)
    # точки н-нада?

    if 1:
        for (x1, y1, _x2, _y2), e in zip(lines, err):
            cv2.circle(frame_layer, center=(x1, y1), radius=2, color=(255, 0, 255), thickness=1)
            # if (e[0] < 100) and (e[0] > 100):
            #     if e[0] > 0:
            #         cv2.circle(frame_layer, center=(x1, y1), radius=int(e[0] // 2 + 1), color=(0, 0, 255), thickness=1)
            #     elif e[0] < 0:
            #         cv2.circle(frame_layer, center=(x1, y1), radius=int(e[0] // 2 + 1), color=(0, 255, 0), thickness=1)
            # else:
            #     if e[0] > 0:
            #         cv2.circle(frame_layer, center=(x1, y1), radius=10, color=(255, 255, 0,), thickness=1)
            #     elif e[0] < 0:
            #         cv2.circle(frame_layer, center=(x1, y1), radius=10, color=(0, 255, 255,), thickness=1)
    _ = cv2.polylines(frame_layer, rlines, False, (255, 0, 255))
    # flow = np.concatenate((flow, points), axis=1)
    return frame_layer, flow_, points, status, err


def mark_points(img, points, to_draw=True, to_return=False):
    """Отметит крестиком указанные точки на img, или на пустом изображении размерами img=[h, w]"""
    if not hasattr(img, 'shape'):
        h, w = img
    else:
        h, w = img.shape[:2]
    layer = np.zeros((h, w, 3), np.uint8)
    for x, y in points:
        # assert (x - 7 > 0) and (y - 7 > 0), ''
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
    # hsv[..., 0] = ang * (180 / np.pi / 2)
    # hsv[..., 1] = 255
    # hsv[..., 2] = np.minimum(modulus * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    for x, y, in points:
        cv2.circle(bgr, center=(x, y), radius=6, color=(int(bgr[y, x, 0]), int(bgr[y, x, 1]), int(bgr[y, x, 2])),
                   thickness=-1)

    return bgr


# Plot values in opencv program
class Plotter:
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

    # Update new values in plot
    def plot(self, val, label="plot"):
        if not label in self.plots:
            self.plots[label] = []
            self.plot_t_last[label] = 0

        self.plots[label].append(int(val))
        while len(self.plots[label]) > self.sample_buffer:
            self.plots[label].pop(0)
            self.show_plot(label)
            # Show plot using opencv imshow

    def show_plot(self, label):

        self.plot_canvas = np.zeros((self.height, self.width, 3)) * 255
        cv2.line(self.plot_canvas,
                 (self.margin_l, int((self.height - self.margin_d - self.margin_u) / 2) + self.margin_u), (
                 self.width - self.margin_r,
                 int((self.height - self.margin_d - self.margin_u) / 2) + self.margin_u), (0, 0, 255), 1)

        # Scaling the graph in y within buffer
        scale_h_max = max(self.plots[label])
        scale_h_min = min(self.plots[label])
        scale_h_min = -scale_h_min if scale_h_min < 0 else scale_h_min
        scale_h = scale_h_max if scale_h_max > scale_h_min else scale_h_min
        scale_h = ((self.height - self.margin_d - self.margin_u) / 2) / scale_h if not scale_h == 0 else 0

        for j, i in enumerate(np.linspace(0, self.sample_buffer - 2, self.width - self.margin_l - self.margin_r)):
            i = int(i)
            cv2.line(self.plot_canvas,
                     (j + self.margin_l,
                      int((self.height - self.margin_d - self.margin_u) / 2 + self.margin_u -
                          self.plots[label][i] * scale_h)),
                     (j + self.margin_l,int((self.height - self.margin_d - self.margin_u) / 2 + self.margin_u -
                                            self.plots[label][i + 1] * scale_h)), self.color, 1)

        cv2.rectangle(self.plot_canvas, (self.margin_l, self.margin_u),
                      (self.width - self.margin_r, self.height - self.margin_d), (255, 255, 255), 1)
        # cv2.putText(self.plot_canvas,
        #             f" {label} : {self.plots[label][-1]} , dt : {int((time.time() - self.plot_t_last[label]) * 1000)}ms",
        #             (int(0), self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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


def draw_plot(values, label='graph'):
    # Create dummy values using for loop
    sample_buffer = len(values)-1
    p = Plotter(sample_buffer*2, 1080, sample_buffer=sample_buffer)

    for v in values:
        p.plot(v, label=label)

    # p.plot(int(math.cos(v * 3.14 / 180) * 50), label='cos')


def color_hsv_division(img1):
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


# imgs = get_and_open_image((1440, 1441, 1460, 1461))
# img1 = imgs[0]
# img2 = imgs[1]
# step = 100
# frame_layer, flow, points, status, err = get_flow_lk(img1, img2, step=30)
# # res = cv2.add(frame_layer, flow)
# open_images((frame_layer,), 'frame_layer')



def open_layer(layer, name='Name'):
    """писал-писал, не дописал"""
    h, w, depth = layer.shape
    img = np.zeros((h, w, depth), np.uint8)
    cv2.imshow(name, img)
    while True:
        ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
        if ch == 27:  # Esc
            break
    cv2.destroyAllWindows()


def change_format():
    """Поменяет разрешение видео вроде может быть"""
    import time
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
        print("Camera is not opened")
        print("\\(-_-)/")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()

        rescaled_frame = rescale_frame(frame)

        # write the output frame to file
        writer.write(rescaled_frame)

        cv2.imshow("Output", rescaled_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
    writer.release()
