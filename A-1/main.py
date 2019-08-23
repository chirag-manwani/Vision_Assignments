import cv2
import matplotlib.pyplot as plt


def get_frames(path):
    vidObj = cv2.VideoCapture(path)
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
    back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=150,
                                                  detectShadows=False)
    vidObj = cv2.VideoCapture(path)
    while True:
        success, frame = vidObj.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = back_sub.apply(frame)

        # mask = cv2.medianBlur(mask, 3)
        # cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        # cv2.imshow("diff", mask)

        k = cv2.waitKey(2)
        if k == 'q' or k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = "../../Assignment_data/A-1/1.mp4"
    frames = get_frames(path)
    remove_background(path)
