import cv2
import sys
import os

from ImageSticher import ImageSticher


def main(
    img_dir
):
    sticher = ImageSticher(img_dir)
    sticher.stitch()


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print("Invalid number of Arguments. Usage- python main.py /path/to/img/dir")
        exit()
    main(args[0])
