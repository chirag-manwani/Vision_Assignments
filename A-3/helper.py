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


def get_keypoints(
    self,
    img
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(sigma=self.sigma)
    kps, des = sift.detectAndCompute(gray, None)

    kps = np.float32([kp.pt for kp in kps])
    return kps, des


def matchKeypoints(
    self,
    kpsA,
    kpsB,
    featuresA,
    featuresB,
    ratio,
    reprojThresh
):
    bf = cv2.BFMatcher(crossCheck=True)
    matches = bf.match(featuresA, featuresB)

    matches.sort(key=lambda x: x.distance, reverse=False)

    ptsA = np.float32([kpsA[match.queryIdx] for match in matches[:20]])
    ptsB = np.float32([kpsB[match.trainIdx] for match in matches[:20]])

    (H, status) = cv2.findHomography(ptsA, ptsB,
                                        cv2.RANSAC,
                                        reprojThresh)
    A = cv2.getAffineTransform(ptsA[:3], ptsB[:3])
    return (matches, H, A)