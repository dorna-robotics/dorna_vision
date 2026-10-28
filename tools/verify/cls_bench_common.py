import cv2
import numpy as np
from PIL import Image

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class LetterboxTransform:  # train_classification.ipynb, verbatim
    def __init__(self, target_size=(224, 224), color=128):
        self.target_size = target_size
        self.color = color

    def __call__(self, image):
        arr = np.array(image)
        ih, iw = arr.shape[:2]
        tw, th = self.target_size
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((th, tw, 3), self.color, dtype=np.uint8)
        pw = (tw - nw) // 2
        ph = (th - nh) // 2
        out[ph:ph + nh, pw:pw + nw] = resized
        return Image.fromarray(out)


