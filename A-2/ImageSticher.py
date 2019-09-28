import cv2
import os


class ImageSticher:

    def __init__(
        self,
        img_dir
    ):
        load_dir(img_dir)

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
            img = cv2.imread(img_path)
            images.append(img)

        self.images = images