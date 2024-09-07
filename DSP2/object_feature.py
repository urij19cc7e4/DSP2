import numpy as np


def calc_count(array):

	height, width = array.shape
	result = np.uint64(0)

	for i in range(height):
		for j in range(width):
			if array[i, j] == 255:
				result += 1

	return result


def calc_center(array):

	height, width = array.shape
	result_y, result_x = 0.0, 0.0
	count = np.uint64(0)

	for i in range(height):
		for j in range(width):
			if array[i, j] == 255:
				result_y += i
				result_x += j
				count += 1

	return result_y / count, result_x / count