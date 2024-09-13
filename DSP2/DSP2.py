import cv2
import numpy as np
from PIL import Image as img
from matplotlib import pyplot as plt

import elem_wise_proc as ep
import filter_proc as fp
import histogram as hst

import object_feature as oft
import object_recognition as orc

image = img.open("lab_2_task_small.png")
image = image.convert('RGB')
image_array = np.array(image)

orc.plot_objects(ep.cut_window_preparation(ep.grayscale(image_array), 191, 0, 0, 255))