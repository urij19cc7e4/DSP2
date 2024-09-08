import math
import numpy as np
import random as rnd
from matplotlib import pyplot as plt
from matplotlib import gridspec as gsp
from dataclasses import dataclass

import object_recognition as orc


def calc_axis_eccentricity(object_array, center_x: float, center_y: float):

	height, width = object_array.shape
	mu_xy, mu_xx, mu_yy = 0.0, 0.0, 0.0

	for i in range(height):
		for j in range(width):
			if object_array[i, j] != 0:
				delta_x = float(i) - center_x
				delta_y = float(j) - center_y

				mu_xy += delta_x * delta_y * float(object_array[i, j])
				mu_xx += delta_x * delta_x * float(object_array[i, j])
				mu_yy += delta_y * delta_y * float(object_array[i, j])

	axis_angle = math.atan((mu_xy / (mu_xx - mu_yy)) * 2.0) * 0.5
	eccentricity = (mu_xx + mu_yy + ((mu_xx - mu_yy) ** 2.0 + mu_xy * mu_xy * 4.0) ** 0.5) \
				 / (mu_xx + mu_yy - ((mu_xx - mu_yy) ** 2.0 + mu_xy * mu_xy * 4.0) ** 0.5)

	return axis_angle, eccentricity


def calc_center(object_array):

	height, width = object_array.shape
	result, result_x, result_y = 0.0, 0.0, 0.0

	for i in range(height):
		for j in range(width):
			if object_array[i, j] != 0:
				result += float(object_array[i, j])
				result_x += float(i) * float(object_array[i, j])
				result_y += float(j) * float(object_array[i, j])

	return result_x / result, result_y / result


def calc_density(perimeter: float, square: float):
	return perimeter * perimeter / square


def calc_perimeter(edges_array):

	height, width = edges_array.shape
	result = 0.0

	for i in range(height):
		for j in range(width):
			if edges_array[i, j] != 0:
				result += 1.0

	return result


def calc_square(object_array):

	height, width = object_array.shape
	result = 0.0

	for i in range(height):
		for j in range(width):
			if object_array[i, j] != 0:
				result += float(object_array[i, j])

	return result / 255.0


@dataclass
class object_info:
	axis_angle: float = 0.0
	eccentricity: float = 0.0
	center_x: float = 0.0
	center_y: float = 0.0
	density: float = 0.0
	perimeter: float = 0.0
	square: float = 0.0


def plot_objects(image_array):

	height, width = image_array.shape
	object_map, object_count = orc.recognize_recursive(image_array)
	objects_list, edges_list = orc.split_objects(image_array, object_map, object_count)

	color_array = np.array([
		[np.uint8(rnd.randint(64, 191)) for _ in range(3)] for _ in range(object_count)
	])

	objects_image_array = np.empty((height, width, 3), dtype = np.uint8)
	object_infos = []

	for i in range(object_count):
		object_info_tmp = object_info()

		object_info_tmp.center_x, object_info_tmp.center_y = calc_center(objects_list[i])
		object_info_tmp.axis_angle, object_info_tmp.eccentricity = calc_axis_eccentricity(
			objects_list[i], object_info_tmp.center_x, object_info_tmp.center_y
		)
		object_info_tmp.perimeter = calc_perimeter(edges_list[i])
		object_info_tmp.square = calc_square(objects_list[i])
		object_info_tmp.density = calc_density(object_info_tmp.perimeter, object_info_tmp.square)

		object_infos.append(object_info_tmp)

	for i in range(height):
		for j in range(width):
			if object_map[i, j, 1] != 0:
				objects_image_array[i, j] = color_array[np.uint64(object_map[i, j, 1] - 1)]
			elif object_map[i, j, 0] != 0:
				objects_image_array[i, j] = [np.uint8(255) - image_array[i, j] for _ in range(3)]
			else:
				objects_image_array[i, j] = np.uint8(255), np.uint8(255), np.uint8(255)

	for i in range(object_count):
		o_height, o_width = objects_list[i].shape

		for j in range(o_height):
			for k in range(o_width):
				objects_list[i][j, k] = np.uint8(255) - objects_list[i][j, k]

	fig = plt.figure(figsize = (10, 5))
	gs = gsp.GridSpec(object_count, 3, width_ratios = [4, 1, 1])

	ax_big = plt.subplot(gs[:, 0])
	ax_big.imshow(objects_image_array)
	ax_big.axis("off")

	for i in range(object_count):
		o_title = f"Eccentricity: {object_infos[i].eccentricity:.2f}\n" \
			f"Perimeter: {object_infos[i].perimeter:.2f}\n" \
			f"Square: {object_infos[i].square:.2f}\n" \
			f"Density: {object_infos[i].density:.2f}\n"

		ax_small = plt.subplot(gs[i, 1])
		ax_small.imshow(objects_list[i], cmap = "gray")
		ax_small.axis("off")

		ax_small_2 = plt.subplot(gs[i, 2])
		ax_small_2.text(0.0, 0.0, o_title, fontsize = 9, color = "black")
		ax_small_2.axis("off")

	plt.tight_layout()
	plt.show()