import cv2
import numpy as np
from PIL import Image as img
from matplotlib import pyplot as plt

import elem_wise_proc as ep
import filter_proc as fp
import histogram as hst

import object_feature as oft
import object_recognition as orc

import lab_1_task

#lab_1_task.do_lab_1_task()

image = img.open("lab_1_task\\1695138157776_blur_custom_filter.png")
image = image.resize((image.size[0] // 2, image.size[1] // 2))
image = image.convert('RGB')
image_array = np.array(image)

orc.plot_objects(ep.cut_window_preparation(ep.grayscale(image_array), 255, 145, 255, 0))