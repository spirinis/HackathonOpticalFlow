import numpy as np
import cv2


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
    while True:
        ch = 0xFF & cv2.waitKey(1)  # Ждём секунду
        if ch == 27:  # Esc
            break
    cv2.destroyAllWindows()


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
