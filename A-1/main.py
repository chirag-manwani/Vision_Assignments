import cv2
import numpy as np
import copy


def remove_background(path):
    count = 0
    total = 0
    alpha = 0.65
    back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=120,
                                                  detectShadows=False)
    vidObj = cv2.VideoCapture(path)
    points_old = [0, 0, 0, 0]
    frames = []
    while True:
        _, frame = vidObj.read()
        if frame is None:
            break
        total += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = back_sub.apply(gray)
        mask = cv2.morphologyEx(mask,
                                cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        edges = cv2.Canny(mask, 150, 120)
        lines = cv2.HoughLines(edges, 1, np.pi/360, 120)

        points_curr = [0, 0, 0, 0]
        if lines is not None:
            len_ = 800
            # print(len)
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    points_curr[0] += int(x0 + len_*(-b))
                    points_curr[1] += int(y0 + len_*(a))
                    points_curr[2] += int(x0 - len_*(-b))
                    points_curr[3] += int(y0 - len_*(a))

            points_curr[0] = int((points_curr[0] / len(lines)) * alpha +
                                 (points_old[0]) * (1 - alpha))
            points_curr[1] = int((points_curr[1] / len(lines)) * alpha +
                                 (points_old[1]) * (1 - alpha))
            points_curr[2] = int((points_curr[2] / len(lines)) * alpha +
                                 (points_old[2]) * (1 - alpha))
            points_curr[3] = int((points_curr[3] / len(lines)) * alpha +
                                 (points_old[3]) * (1 - alpha))

            points_old = copy.deepcopy(points_curr)
            frames.append(frame)
            cv2.line(frame,
                     (points_curr[0], points_curr[1]),
                     (points_curr[2], points_curr[3]),
                     color=(0, 255, 255),
                     thickness=5)
            count += 1
        cv2.imshow("Original", frame)
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Edges", edges)

        k = cv2.waitKey(2)
        if k == 'q' or k == 27:
            break
    print('Fraction of frames with lines detected:', count/total)

    h, w, _ = frames[0].shape
    size = (w, h)
    fps = 24
    out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for frame in frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "../../Assignment_data/A-1/8.mp4"
    # frames = get_frames(path)
    remove_background(path)
