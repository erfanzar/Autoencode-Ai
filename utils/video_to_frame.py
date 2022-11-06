import os.path

import cv2 as cv
import numpy as np
import numba as nb


def read_video(path):
    fr = cv.VideoCapture(path)
    return fr


def write_video_frame(video):
    if not os.path.exists('../data/reader'):
        os.mkdir("../data/reader")
    cap = read_video(video)
    f = 0
    while True:
        ret, frame = cap.read()
        # print(ret,frame)
        if ret:
            f += 1
            cv.imshow('windows', frame)
            cv.imwrite(f'reader/{f}.jpg', frame)
            cv.waitKey(1)
            if cv.waitKey(1) == ord('q'):
                break
        else:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":

    write_video_frame('Nebula.mp4')
