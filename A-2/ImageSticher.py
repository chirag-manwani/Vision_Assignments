import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


class ImageSticher:

    def __init__(
        self,
        img_dir,
        n_keypoints=10000,
        sigma=1.6
    ):
        self.__load_dir__(img_dir)
        self.n_keypoints = n_keypoints
        self.sigma = sigma

    def __load_dir__(
        self,
        img_dir
    ):
        files = os.listdir(img_dir)
        images = []
        for file_ in files:
            if file_.split('.')[-1] not in ['jpg', 'png', 'jpeg']:
                continue
            img_path = os.path.join(img_dir, file_)
            img = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
            images.append(img)

        self.images = images

    def __get_keypoints__(
        self,
        img
    ):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sift = cv2.xfeatures2d.SIFT_create(self.n_keypoints, sigma=self.sigma)
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
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

    def stitch(
        self
    ):
        img1 = self.images[0]
        img2 = self.images[1]

        kps1, des1 = self.__get_keypoints__(img1)
        kps2, des2 = self.__get_keypoints__(img2)

        M = self.__matchKeypoints__(kps1, kps2,
                                    des1, des2,
                                    0.75, 4)

        (matches, H, status) = M
        result = cv2.warpPerspective(img1, H,
                                     (img1.shape[1]+img2.shape[1], img1.shape[0]))
        result[0:img2.shape[0], 0:img2.shape[1]] = img2

        # vis = drawMatches(imageA, imageB, kpsA, kpsB, matches,
        #                        status)

        # return a tuple of the stitched image and the
        # visualization
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.imsave('out.jpg', result)

        return result
