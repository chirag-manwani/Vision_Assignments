import cv2
import numpy as np


def calibrate(
    img,
    pat=(5, 3)
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, pat, None)
    img2 = img
    corners2 = []
    if ret:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        img2 = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
    return img2, corners


def record(
    save_path
):
    cap = cv2.VideoCapture(0)
    frames = []
    while(len(frames) != 300):
        ret, frame = cap.read()
        frames.append(frame)
        print(len(frames))
    h, w, _ = frames[0].shape
    size = (w, h)
    fps = 30
    out = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          fps,
                          size)
    for frame in frames:
        out.write(frame)
    out.release()
