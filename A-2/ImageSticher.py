import cv2
import os
import numpy as np
import natsort

import matplotlib.pyplot as plt

from helper import trim


class ImageSticher:

    def __init__(
        self,
        img_dir,
        sigma=1.6
    ):
        self.__load_dir__(img_dir)
        self.sigma = sigma

    def __load_dir__(
        self,
        img_dir
    ):
        files = os.listdir(img_dir)
        files = natsort.natsorted(files)
        print(files)
        images = []
        for file_ in files:
            if file_.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                continue
            img_path = os.path.join(img_dir, file_)
            img = cv2.imread(img_path)
            dim = (int(img.shape[1]/4), int(img.shape[0]/4))
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            images.append(img)

        self.images = images

    def __get_keypoints__(
        self,
        img
    ):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(sigma=self.sigma)
        kps, des = sift.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])
        return kps, des

    def __matchKeypoints__(
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

        return (matches, H, status)

    def stitch(
        self
    ):
        n = len(self.images)
        result = self.stitch_2(self.images[1], self.images[0])
        for i in range(2, n, 1):
            print("i=", i)
            result = self.stitch_2(self.images[i], result)
        return result

    def stitch_2(
        self,
        img0,
        img1
    ):
        img = []
        img.append(img0)
        img.append(img1)

        kps0, des0 = self.__get_keypoints__(img[0])
        kps1, des1 = self.__get_keypoints__(img[1])

        M = self.__matchKeypoints__(kps0, kps1,
                                    des0, des1,
                                    0.25, 4)

        (matches, H, status) = M

        # exit()
        x0 = kps0[matches[0].queryIdx][0]
        x1 = kps1[matches[1].trainIdx][0]

        overlap = int(abs(x0 - x1))

        result = cv2.warpPerspective(img[0], H,
                                     (img[0].shape[1]+img[1].shape[1], img[0].shape[0]))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

        result[0:img[1].shape[0], 0:img[1].shape[1]] = img[1]
        result = trim(result)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
        return result
