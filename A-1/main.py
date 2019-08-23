import cv2
import numpy as np


def remove_background(path):
    # f = frames[0]
    # print(frames[-1])
    # for frame in frames:
    count = 0
    total = 0
    back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=200,
                                                  detectShadows=False)
    vidObj = cv2.VideoCapture(path)
    while True:
        success, frame = vidObj.read()
        if frame is None:
            break
        total += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = back_sub.apply(gray)

        # diff = cv2.absdiff(f, frame)
        # mask = cv2.medianBlur(mask, 3)
        # cv2.imshow("Original", frame)
        edges = cv2.Canny(mask, 150, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/360, 120)
        # lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, 150)

        x1, y1, x2, y2 = 0, 0, 0, 0
        if lines is not None:
            len_ = 400
            # print(len)
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 += int(x0 + len_*(-b))
                    y1 += int(y0 + len_*(a))
                    x2 += int(x0 - len_*(-b))
                    y2 += int(y0 - len_*(a))

            x1 //= len(lines)
            y1 //= len(lines)
            x2 //= len(lines)
            y2 //= len(lines)
            cv2.line(frame, (x1, y1), (x2, y2),
                     color=(0, 255, 255),
                     thickness=5)
            count += 1
        # cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        # cv2.imshow("Edges", edges)

        k = cv2.waitKey(2)
        if k == 'q' or k == 27:
            break
    print(count/total)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "../../Assignment_data/A-1/1.mp4"
    # frames = get_frames(path)
    remove_background(path)
