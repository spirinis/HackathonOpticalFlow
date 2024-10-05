import numpy as np
import cv2


def GetAndOpenImage():
    cap = cv2.VideoCapture("./PERFECT_BANDO_FPV_FREESTYLE.mp4")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 116)
    ret, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 117)
    ret, frame2 = cap.read()
    # not (np.bitwise_xor(frame, frame2).any())
    cv2.imshow('Name', frame1)
    cv2.imshow('Name', frame2)
    while True:
        ch = 0xFF & cv2.waitKey(1)  # Wait for a second
        if ch == 27:  # Esc
            break
    cap.release()
    cv2.destroyAllWindows()