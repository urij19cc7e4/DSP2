import numpy as np


def linear_correction(image_array: np.ndarray, alpha = 0.5, beta = 127.0) -> np.ndarray:
	return np.clip(image_array * alpha + beta, 0, 255).astype(np.uint8)


def gamma_correction(image_array: np.ndarray, gamma = 0.5, c = 1.05) -> np.ndarray:
	return np.clip(((image_array / 255.0) ** gamma) * 255.0 * c, 0, 255).astype(np.uint8)


def log_correction(image_array: np.ndarray, c = 20.0) -> np.ndarray:
	return np.clip(np.log(image_array + 1.0) * c, 0, 255).astype(np.uint8)


def preparation(image_array: np.ndarray, table: np.ndarray) -> np.ndarray:

	if image_array.ndim == 2:
		height, width = image_array.shape
		preparated_image_array = np.empty((height, width), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				preparated_image_array[i, j] = table[image_array[i, j]]

		return preparated_image_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		preparated_image_array = np.empty((height, width, depth), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					preparated_image_array[i, j, k] = table[image_array[i, j, k]]

		return preparated_image_array

	else:
		raise Exception("Not implemented.")


def cut_window_preparation(image_array: np.ndarray, in_max = 191, in_min = 64,
						   out_max = 255, out_min = 0) -> np.ndarray:

	table = np.clip(np.array(
		[(out_max if (in_min <= i <= in_max) else out_min) for i in range(256)]
	), 0, 255).astype(np.uint8)

	return preparation(image_array, table)


def cut_diagonal_preparation(image_array: np.ndarray, in_end = 191, in_start = 64,
							 out_end = 255, out_start = 0) -> np.ndarray:

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


def negative(image_array: np.ndarray) -> np.ndarray:
	return np.clip(np.uint8(255) - image_array, 0, 255).astype(np.uint8)


def grayscale(image_array: np.ndarray) -> np.ndarray:

	height, width, _ = image_array.shape
	grayscale_image_array = np.empty((height, width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			grayscale_image_array[i, j] = np.clip(
				float(r) * 0.299 + float(g) * 0.587 + float(b) * 0.114, 0, 255
			).astype(np.uint8)

	return grayscale_image_array


def solarize(image_array: np.ndarray, k = 0.015625) -> np.ndarray:

	table = np.clip(np.array(
		[(255.0 - i) * i * k for i in range(256)]
	), 0, 255).astype(np.uint8)

	return preparation(image_array, table)