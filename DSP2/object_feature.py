import math
import numpy as np
import random as rnd


def calc_axis_eccentricity(object_array: np.ndarray, center_x: float, center_y: float) -> (float, float):

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


def calc_center(object_array: np.ndarray) -> (float, float):

	height, width = object_array.shape
	result, result_x, result_y = 0.0, 0.0, 0.0

	for i in range(height):
		for j in range(width):
			if object_array[i, j] != 0:
				result += float(object_array[i, j])

				result_x += float(i) * float(object_array[i, j])
				result_y += float(j) * float(object_array[i, j])

	return result_x / result, result_y / result


def calc_density(perimeter: float, square: float) -> float:
	return perimeter * perimeter / square


def calc_perimeter(edges_array: np.ndarray) -> float:

	height, width = edges_array.shape
	result = 0.0

	for i in range(height):
		for j in range(width):
			if edges_array[i, j] != 0:
				result += 1.0

	return result


def calc_square(object_array: np.ndarray) -> float:

	height, width = object_array.shape
	result = 0.0

	for i in range(height):
		for j in range(width):
			if object_array[i, j] != 0:
				result += float(object_array[i, j])

	return result / 255.0


def k_means(features_1: np.ndarray[float], features_2: np.ndarray[float], class_count = 2) -> np.ndarray:

	object_count = features_1.shape[0]
	center_array = np.zeros(class_count, dtype = np.dtype([("f1", float), ("f2", float)]))
	result_array = np.empty(object_count, dtype = np.uint64)

	f1_max, f1_min = features_1.max(), features_1.min()
	f2_max, f2_min = features_2.max(), features_2.min()
	max_distance = float((f1_max - f1_min) ** 2.0 + (f2_max - f2_min) ** 2.0) ** 0.5

	for i in range(class_count):
		center_array[i][0] = rnd.uniform(f1_min, f1_max)
		center_array[i][1] = rnd.uniform(f2_min, f2_max)

	while True:
		__continue__ = False
		center_array_temp = np.zeros(class_count, dtype = \
			np.dtype([("f1", float), ("f2", float), ("__class__", np.uint64)]))

		for i in range(object_count):
			min_distance = max_distance
			nearest_class = -1

			for j in range(class_count):
				distance = float((features_1[i] - center_array[j][0]) ** 2.0 \
					+ (features_2[i] - center_array[j][1]) ** 2.0) ** 0.5

				if min_distance > distance:
					min_distance = distance
					nearest_class = j

			center_array_temp[nearest_class][0] += features_1[i]
			center_array_temp[nearest_class][1] += features_2[i]
			center_array_temp[nearest_class][2] += 1
			result_array[i] = nearest_class

		for i in range(class_count):
			if center_array_temp[i][2] == 0:
				center_array_temp[i][0] = center_array[i][0]
				center_array_temp[i][1] = center_array[i][1]
			else:
				center_array_temp[i][0] /= center_array_temp[i][2]
				center_array_temp[i][1] /= center_array_temp[i][2]

			if center_array_temp[i][0] != center_array[i][0] \
				or center_array_temp[i][1] != center_array[i][1]:
				__continue__ = True

		if __continue__:
			for i in range(class_count):
				center_array[i] = center_array_temp[i][0], center_array_temp[i][1]
		else:
			break

	return result_array