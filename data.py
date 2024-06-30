import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pathlib import Path
import albumentations
import json
from xml.dom import minidom
import xml
import xml.etree.ElementTree as ET


def read_photos(dir_path):
    data_x = []  # all file names of images

    for image_path in dir_path.glob('*.jpg'):
        data_x.append(image_path.__str__())

    return np.array(data_x)


def read_annot(annot_dir):
    data_y = []
    for name in annot_dir.glob('*.xml'):
        data_tmp = []
        tree = ET.parse(name.__str__())
        root = tree.getroot()
        for object in root.findall('object'):
            name = object.find('name').text
            object = object.find('bndbox')
            xmin = int(object.find('xmin').text)
            ymin = int(object.find('ymin').text)
            xmax = int(object.find('xmax').text)
            ymax = int(object.find('ymax').text)
            data_tmp.append([name, (xmin, ymin), (xmax, ymax)])
        data_y.append(data_tmp)
    return data_y


def split(data_x, data_y, amount):
    """
    Train test split
    """
    return (data_x[:-amount], data_y[:-amount]), (data_x[-amount:], data_y[-amount:])


def collect_data():
    """
    Gather all data
    """
    dir_path = Path(__file__).parent / 'small-weak-UVA-object-dataset' / 'small-weak-UVA-object-dataset'
    photos_names = read_photos(dir_path / 'JPEGImages')
    objects = read_annot(dir_path / 'Annotations')

    (train_x, train_y), (test_x, test_y) = split(photos_names, objects, 200)
    return (train_x, train_y), (test_x, test_y)


def tst():
    name = Path(__file__).parent / 'small-weak-UVA-object-dataset' / 'small-weak-UVA-object-dataset' / 'Annotations' / '1.xml'
    tree = ET.parse(name.__str__())
    root = tree.getroot()
    for object in root.findall('object'):
        name = object.find('name').text
        print(f"name: {name}")


if __name__ == '__main__':
    collect_data()