import cv2
import os
import numpy as np
import natsort

import matplotlib.pyplot as plt

from helper import trim, get_alpha


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
        images = []
        for file_ in files:
            if file_.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                continue
            img_path = os.path.join(img_dir, file_)
            img = cv2.imread(img_path)
            dim = (int(img.shape[1]/2), int(img.shape[0]/2))
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
        A = cv2.getAffineTransform(ptsA[:3], ptsB[:3])
        return (matches, H, A)

    def stitch(
        self
    ):
        n = len(self.images)
        result = self.stitch_2(self.images[1], self.images[0])
        for i in range(2, n, 1):
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

        (matches, H, A) = M

        x0 = kps0[matches[0].queryIdx][0]
        x1 = kps1[matches[1].trainIdx][0]

        overlap = int(abs(x0 - x1))
        width = 21
        alpha = get_alpha(width)
        result = cv2.warpPerspective(img[0], H,
                                     (img[0].shape[1]+img[1].shape[1], img[0].shape[0]))
        result[0:img[1].shape[0], 0:img[1].shape[1]-width] = img[1][0:img[1].shape[0], 0:img[1].shape[1]-width]

        result[0:img[1].shape[0], img[1].shape[1]-width:img[1].shape[1]] = \
            alpha * img[1][0:img[1].shape[0], img[1].shape[1]-width:img[1].shape[1]] + \
            (1-alpha) * result[0:img[1].shape[0], img[1].shape[1]-width:img[1].shape[1]]
        result = trim(result)

        # result_affine = cv2.warpAffine(img[0],
        #                                A,
        #                                (int(img[0].shape[1] + abs(A[0, 2])), int(img[0].shape[0] + abs(A[1, 2]))))
        # print(result_affine.shape)
        # result_affine[0:img[1].shape[0], 0:img[1].shape[1]] = img[1]
        # plt.imshow(cv2.cvtColor(result_affine, cv2.COLOR_BGR2RGB))
        # plt.show()

        return result
