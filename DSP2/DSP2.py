import cv2
import numpy as np
from PIL import Image as img
from matplotlib import pyplot as plt

import elem_wise_proc as ep
import filter_proc as fp
import histogram as hst

import object_feature as oft
import object_recognition as orc

image = img.open('C:/Users/Urij/Downloads/objects.png')
image = image.convert('RGB')
image_array = np.array(image)

object_map, object_count = orc.recognize_sequential(ep.cut_window_preparation(ep.grayscale(image_array), 255, 240, 0, 255))

a = orc.split_objects(object_map, object_count)

objects_image_array = ep.cut_window_preparation(object_map[:, :, 0], 255, 1, 255, 0)
objects_image = img.fromarray(objects_image_array)
objects_image.save('obj.jpg')

objects_inner_image_array = ep.cut_window_preparation(object_map[:, :, 1], 255, 1, 255, 0)
objects_inner_image = img.fromarray(objects_inner_image_array)
objects_inner_image.save('obj_inner.jpg')

objects_outer_image_array = ep.cut_window_preparation(object_map[:, :, 2], 255, 1, 255, 0)
objects_inner_image = img.fromarray(objects_outer_image_array)
objects_inner_image.save('obj_outer.jpg')