from pathlib import Path
import cv2
from data import collect_data
from training_classificator import SiameseDataset
import numpy as np
from algorithm import detection
from matplotlib import pyplot


if __name__ == '__main__':
    train_coll, val_coll = collect_data()
    train_dataset = SiameseDataset(train_coll)
    # valid_dataset = SiameseDataset(val_coll)
    i = iter(train_dataset.compose_iterations_raw())

    image, ref_image, target = next(i)
    image = np.array(image)
    ref_image = np.array(ref_image)

    out = detection(ref_image, image)

    for (xmin, ymin), (xmax, ymax) in out:
        image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 3)

    fg, ax = pyplot.subplots(1, 2)
    ax[0].imshow(image)
    ax[1].imshow(ref_image)
    pyplot.show()
