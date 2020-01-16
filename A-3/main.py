import numpy as np
import sys
import os
import cv2
import pywavefront
from helper import calibrate
from helper import record
from helper import get_keypoints
from helper import matchKeypoints
from helper import projection_matrix
from helper import OBJ, render

data_base_path = '../../Assignment_data/A-3'


def checkerBoard(args):
    objp = np.zeros((4*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:4].T.reshape(-1, 2)

    cap = cv2.VideoCapture('abc.avi')
    # print(objp, objp.shape, type(objp))
    while(True):
        # print('here')
        ret, frame = cap.read()
        if(not ret):
            break
        frame_, corners = calibrate(frame, (6, 4))
        
        if(corners is not None):
            print('Corners', corners.shape, objp.shape)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp],
                                                               [corners],
                                                               frame_.shape[:2][::-1],
                                                               None,
                                                               None)
            # print(mtx)
            # print(dist)

        cv2.imshow('frame', frame_)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def part3(
    args,
    vis=False
):
    img_m = cv2.imread(args[1])
    kpsm, desm = get_keypoints(img_m)

    cap = cv2.VideoCapture(args[0])
    
    while(True):
        ret, frame = cap.read()
        if(not ret):
            break
        # frame = cv2.flip(frame, 1)

        kpsf, desf = get_keypoints(frame)
        matches, H, ptsm, ptsf = matchKeypoints(kpsm, kpsf, desm, desf, 5)

        # Reshaping
        b = np.zeros((ptsm.shape[0], ptsm.shape[1]+1))
        b[:,:-1] = ptsm
        ptsm = np.array(b, dtype='float32')
        ptsf = np.reshape(ptsf, (ptsf.shape[0], 1, 2))
        # Reshaping end

        if H is not None:
            ret, c_mat, dist, rvecs, tvecs = cv2.calibrateCamera([ptsm], [ptsf], frame.shape[:2][::-1], None, None)
            size = frame.shape[:2]
            focal_length = size[1] 
            center = (size[1]/2, size[0]/2) 
            camera_matrix = np.array( 
                            [[ focal_length ,      0       , center[0] ], 
                             [      0       , focal_length , center[1] ], 
                             [      0       ,      0       ,     1     ]], dtype = "double"
                             )
            p_mat = projection_matrix(camera_matrix, H)
            if (vis):
                h, w = img_m.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)
                img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
                cv2.imshow('frame', img2)
                # cv2.waitKey(0)
            # obj = pywavefront.Wavefront('pc/police-car.obj')
            
            obj = OBJ('lp/lp.obj')
            frame_ = render(frame, obj, p_mat, img_m, color=False)
            cv2.imshow('frame', frame_)
            # cv2.waitKey(0)
            # print(cam_mat)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def part4(
    args,
    vis=False
):
    img_m = cv2.imread(args[1])
    kpsm, desm = get_keypoints(img_m)

    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        if(not ret):
            break
        # frame = cv2.flip(frame, 1)

        kpsf, desf = get_keypoints(frame)
        matches, H, ptsm, ptsf = matchKeypoints(kpsm, kpsf, desm, desf, 5)

        # Reshaping
        b = np.zeros((ptsm.shape[0], ptsm.shape[1]+1))
        b[:,:-1] = ptsm
        ptsm = np.array(b, dtype='float32')
        ptsf = np.reshape(ptsf, (ptsf.shape[0], 1, 2))
        # Reshaping end

        if H is not None:
            ret, c_mat, dist, rvecs, tvecs = cv2.calibrateCamera([ptsm], [ptsf], frame.shape[:2][::-1], None, None)
            size = frame.shape[:2]
            focal_length = size[1] 
            center = (size[1]/2, size[0]/2) 
            camera_matrix = np.array( 
                            [[ focal_length ,      0       , center[0] ], 
                             [      0       , focal_length , center[1] ], 
                             [      0       ,      0       ,     1     ]], dtype = "double"
                             )
            p_mat = projection_matrix(camera_matrix, H)
            if (vis):
                h, w = img_m.shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, H)
                img2 = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
                cv2.imshow('frame', img2)
                # cv2.waitKey(0)
            # obj = pywavefront.Wavefront('pc/police-car.obj')
            
            obj = OBJ('lp/lp.obj')
            frame_ = render(frame, obj, p_mat, img_m, color=False)
            cv2.imshow('frame', frame_)
            # cv2.waitKey(0)
            # print(cam_mat)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 1:
        exit()
    # checkerBoard('abc.avi')
    part3(args)
