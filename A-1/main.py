import cv2
import numpy as np


def get_frames(path):
    vidObj = cv2.VeideoCapture(path)
    count = 0
    success = 1
    frames = []
    success, image = vidObj.read()
    while success:
        count += 1
        frames.append(image)
        success, image = vidObj.read()
    vidObj.release()
    return frames


def remove_background(path):
    # f = frames[0]
    # print(frames[-1])
    # for frame in frames:
    count = 0
    total = 0
    back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=150,
                                                  detectShadows=False)
    vidObj = cv2.VideoCapture(path)
    while True:
        success, frame = vidObj.read()
        if frame is None:
            break
        total += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = back_sub.apply(frame)

        # diff = cv2.absdiff(f, frame)
        # mask = cv2.medianBlur(mask, 3)
        # cv2.imshow("Original", frame)
        edges = cv2.Canny(mask, 150, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/360, 50)
        if lines is not None:
            print(len(lines))
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 10)
            count += 1
        cv2.imshow("Original", frame)
        # cv2.imshow("Mask", mask)
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
