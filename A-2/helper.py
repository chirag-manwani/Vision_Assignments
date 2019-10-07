import numpy as np 
import cv2


def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])
    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame


def hisEqulColor(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def get_alpha(width):
    steps = width - 1
    alpha = [1]
    diff = (1) / steps
    for step in range(steps):
        last = alpha[-1]
        alpha.append(last - diff)
    alpha[-1] = 0
    return np.vstack([alpha, alpha, alpha]).T
