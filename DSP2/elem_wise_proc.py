import numpy as np


def linear_correction(image_array, alpha = 1.0, beta = 0.0):
	return np.clip(image_array * alpha + beta, 0, 255).astype(np.uint8)


def gamma_correction(image_array, gamma = 1.0, c = 1.0):
	return np.clip(((image_array / 255.0) ** gamma) * 255.0 * c, 0, 255).astype(np.uint8)


def log_correction(image_array, c = 1.0):
	return np.clip(np.log(image_array + 1.0) * c, 0, 255).astype(np.uint8)


def preparation(image_array, table):

	height, width, depth = image_array.shape
	preparated_image_array = np.empty((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			for k in range(depth):
				preparated_image_array[i, j, k] = table[image_array[i, j, k]]

	return preparated_image_array


def cut_window_preparation(
		image_array,
		in_max = 191,
		in_min = 64,
		out_max = 255,
		out_min = 0
	):

	table = np.clip(np.array(
		[(out_max if (in_min <= i <= in_max) else out_min) for i in range(256)]
	), 0, 255).astype(np.uint8)

	return preparation(image_array, table)


def cut_diagonal_preparation(
		image_array,
		in_end = 191,
		in_start = 64,
		out_end = 255,
		out_start = 0
	):

	table = np.clip(np.array(
		[(
			out_end if i > in_end else (
				out_start if i < in_start else (
					(i - in_start) * (out_end - out_start) / (in_end - in_start)
				)
			)
		) for i in range(256)]
	), 0, 255).astype(np.uint8)

	return preparation(image_array, table)


def negative(image_array):
	return np.clip(255 - image_array, 0, 255).astype(np.uint8)


def grayscale(image_array):

	height, width, _ = image_array.shape
	grayscale_image_array = np.empty((height, width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]
			grayscale_image_array[i, j] = r * 0.299 + g * 0.587 + b * 0.114

	return grayscale_image_array


def solarize(image_array, k = 1.0):

	table = np.clip(np.array(
		[(255.0 - i) * i * k for i in range(256)]
	), 0, 255).astype(np.uint8)

	return preparation(image_array, table)