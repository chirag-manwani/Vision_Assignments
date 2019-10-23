import numpy as np
import sys
import os
import cv2

from helper import calibrate
from helper import record

data_base_path = '../../Assignment_data/A-3'


def checkerBoard(args):
    objp = np.zeros((4*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)

    cap = cv2.VideoCapture(args[0])

    while(True):
        ret, frame = cap.read()
        if(not ret):
            break
        frame_, corners = calibrate(frame, (6, 4))
        if(corners is not None):
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp],
                                                               [corners],
                                                               frame_.shape[:2][::-1],
                                                               None,
                                                               None)
            print(mtx)
            print(dist)

        cv2.imshow('frame', frame_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        exit()
    # checkerBoard(args)
