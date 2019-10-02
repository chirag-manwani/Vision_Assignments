import cv2
import sys
import os

from ImageSticher import ImageSticher
from helper import hisEqulColor


def main(
    img_dir
):
    sticher = ImageSticher(img_dir)
    result = sticher.stitch()
    result = hisEqulColor(result)
    cv2.imwrite('out.jpg', result)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print("Invalid number of Arguments. Usage- python main.py /path/to/img/dir")
        exit()
    main(args[0])
