import cv2
import numpy as np
from matplotlib import pyplot
from itertools import product


def detection(ref, source):
    source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    ref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)

    scales = np.arange(0.8, 1.3, 0.1)
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_CLOCKWISE]

    threshold = 0.8
    out = []
    for angle in angles:
        ref = cv2.rotate(ref, angle)
        for scale in scales:
            h, w = ref.shape[:2]
            w_t = int(w * scale)
            h_t = int(h * scale)
            ref_tmp = cv2.resize(ref, (w_t, h_t))

            res = cv2.matchTemplate(source, ref_tmp, cv2.TM_CCOEFF_NORMED)

            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                out.append([pt, (pt[0] + w, pt[1] + h)])

    # TODO NMS
    return out
