import cv2
import math
import numpy as np

import elem_wise_proc as ep


def convo2d(image_array, kernel, floating = False):

	h_kernel, w_kernel = kernel.shape
	h_pad, w_pad = h_kernel // 2, w_kernel // 2

	if floating:
		height, width = image_array.shape
		result_image_array = np.empty((height, width), dtype = float)

		for i in range(height):
			for j in range(width):
				sum = 0.0
				for m in range(h_kernel):
					for n in range(w_kernel):
						ii, jj = i + m - h_pad, j + n - w_pad
						if 0 <= ii < height and 0 <= jj < width:
							sum += float(image_array[ii, jj]) * float(kernel[m, n])
				result_image_array[i, j] = sum

		return result_image_array

	else:
		if image_array.ndim == 2:
			height, width = image_array.shape
			result_image_array = np.empty((height, width), dtype = np.uint8)

			for i in range(height):
				for j in range(width):
					sum = 0.0
					for m in range(h_kernel):
						for n in range(w_kernel):
							ii, jj = i + m - h_pad, j + n - w_pad
							if 0 <= ii < height and 0 <= jj < width:
								sum += float(image_array[ii, jj]) * float(kernel[m, n])
					result_image_array[i, j] = np.clip(sum, 0, 255).astype(np.uint8)

			return result_image_array

		elif image_array.ndim == 3:
			height, width, depth = image_array.shape
			result_image_array = np.empty((height, width, depth), dtype = np.uint8)

			for i in range(height):
				for j in range(width):
					for k in range(depth):
						sum = 0.0
						for m in range(h_kernel):
							for n in range(w_kernel):
								ii, jj = i + m - h_pad, j + n - w_pad
								if 0 <= ii < height and 0 <= jj < width:
									sum += float(image_array[ii, jj, k]) * float(kernel[m, n])
						result_image_array[i, j, k] = np.clip(sum, 0, 255).astype(np.uint8)

			return result_image_array

		else:
			raise Exception("Not implemented.")


def convo_test(image_array, kernel):

	pad_y = kernel.shape[0] // 2
	pad_x = kernel.shape[1] // 2
	padded_array = cv2.copyMakeBorder(
		image_array,
		top = pad_y,
		bottom = pad_y,
		left = pad_x,
		right = pad_x,
		borderType = cv2.BORDER_CONSTANT,
		value = 0
	)

	manual_result = convo2d(image_array, kernel)
	opencv_result = np.clip(
		cv2.filter2D(padded_array, -1, kernel), 0, 255
	).astype(np.uint8)[pad_y:-pad_y, pad_x:-pad_x]

	print("Manual convolution result is equal to OpenCV's: " + np.all(manual_result == opencv_result))


def gaussian_kernel(kernel_size = 3):

	kernel_shift = kernel_size // 2
	sigma_power = ((float(kernel_shift) / 3.0) ** 2.0) * 2.0
	kernel = np.empty((kernel_size, kernel_size), dtype = float)

	for i in range(kernel_size):
		for j in range(kernel_size):
			x, y = float(i - kernel_shift), float(j - kernel_shift)
			kernel[i, j] = np.exp(-((x * x + y * y) / sigma_power))

	return kernel / np.sum(kernel)


def low_pass_filter(image_array):

	kernel = np.array([
		[1.0, 1.0, 1.0],
		[1.0, 1.0, 1.0],
		[1.0, 1.0, 1.0]
	], dtype = float) / 9.0

	return convo2d(image_array, kernel)


def high_pass_filter(image_array):

	kernel = np.array([
		[-1.0, -1.0, -1.0],
		[-1.0, 9.0, -1.0],
		[-1.0, -1.0, -1.0]
	], dtype = float)

	return convo2d(image_array, kernel)


def roberts_operator(image_array):

	kernel_1 = np.array([
		[0.0, 1.0],
		[-1.0, 0.0]
	], dtype = float)
	kernel_2 = np.array([
		[1.0, 0.0],
		[0.0, -1.0]
	], dtype = float)

	gray_image_array = np.copy(image_array) if image_array.ndim == 2 else ep.grayscale(image_array)

	edge_1 = convo2d(gray_image_array, kernel_1)
	edge_2 = convo2d(gray_image_array, kernel_2)

	height, width = gray_image_array.shape
	result_image_array = np.empty((height, width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			result_image_array[i, j] = np.clip(
				(float(edge_1[i, j]) ** 2.0 + float(edge_2[i, j]) ** 2.0) ** 0.5, 0, 255
			).astype(np.uint8)

	return result_image_array


def sobel_operator(image_array):

	kernel_1 = np.array([
		[-1.0, -2.0, -1.0],
		[0.0, 0.0, 0.0],
		[1.0, 2.0, 1.0]
	], dtype = float)
	kernel_2 = np.array([
		[-1.0, 0.0, 1.0],
		[-2.0, 0.0, 2.0],
		[-1.0, 0.0, 1.0]
	], dtype = float)

	gray_image_array = np.copy(image_array) if image_array.ndim == 2 else ep.grayscale(image_array)

	edge_1 = convo2d(gray_image_array, kernel_1)
	edge_2 = convo2d(gray_image_array, kernel_2)

	height, width = gray_image_array.shape
	result_image_array = np.empty((height, width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			result_image_array[i, j] = np.clip(
				(float(edge_1[i, j]) ** 2.0 + float(edge_2[i, j]) ** 2.0) ** 0.5, 0, 255
			).astype(np.uint8)

	return result_image_array


def prewitt_operator(image_array):

	kernel_1 = np.array([
		[-1.0, -1.0, -1.0],
		[0.0, 0.0, 0.0],
		[1.0, 1.0, 1.0]
	], dtype = float)
	kernel_2 = np.array([
		[-1.0, 0.0, 1.0],
		[-1.0, 0.0, 1.0],
		[-1.0, 0.0, 1.0]
	], dtype = float)

	gray_image_array = np.copy(image_array) if image_array.ndim == 2 else ep.grayscale(image_array)

	edge_1 = convo2d(gray_image_array, kernel_1)
	edge_2 = convo2d(gray_image_array, kernel_2)

	height, width = gray_image_array.shape
	result_image_array = np.empty((height, width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			result_image_array[i, j] = np.clip(
				max(float(edge_1[i, j]), float(edge_2[i, j])), 0, 255
			).astype(np.uint8)

	return result_image_array


def laplacian_operator(image_array):
	return cv2.Laplacian(image_array, cv2.CV_64F).astype(np.uint8)


def canny_analyse(magnitude_map, angle_map, ii, jj):

	angle = round(angle_map[ii, jj] / (math.pi / 4.0))
	magnitude = magnitude_map[ii, jj]

	if angle == 0.0 or abs(angle) == 4.0:
		if magnitude >= magnitude_map[ii, jj + 1] and magnitude >= magnitude_map[ii, jj - 1]:
			return True
	elif angle == -1.0 or angle == 3.0:
		if magnitude >= magnitude_map[ii - 1, jj + 1] and magnitude >= magnitude_map[ii + 1, jj - 1]:
			return True
	elif angle == -2.0 or angle == 2.0:
		if magnitude >= magnitude_map[ii + 1, jj] and magnitude >= magnitude_map[ii - 1, jj]:
			return True
	elif angle == -3.0 or angle == 1.0:
		if magnitude >= magnitude_map[ii + 1, jj + 1] and magnitude >= magnitude_map[ii - 1, jj - 1]:
			return True
	else:
		return False

	return False


def canny_checker(image_array, magnitude_map, angle_map, strong, weak, ii, jj):

	if weak <= magnitude_map[ii, jj] < strong and image_array[ii, jj] == 0:
		if canny_analyse(magnitude_map, angle_map, ii, jj):
			image_array[ii, jj] = np.uint8(255)
			canny_recursive(image_array, magnitude_map, angle_map, strong, weak, ii, jj)


def canny_recursive(image_array, magnitude_map, angle_map, strong, weak, ii, jj):

	if 0 < ii < image_array.shape[0] - 1 and 0 < jj < image_array.shape[1] - 1:

		ii_array = np.array([ii - 1, ii - 1, ii - 1, ii, ii, ii + 1, ii + 1, ii + 1])
		jj_array = np.array([jj - 1, jj, jj + 1, jj - 1, jj + 1, jj - 1, jj, jj + 1])

		for k in range(8):
			canny_checker(image_array, magnitude_map, angle_map, strong, weak, ii_array[k], jj_array[k])


def canny_operator(image_array, strong = 125, weak = 75, kernel_size = 3):

	kernel_1 = np.array([
		[-1.0, -2.0, -1.0],
		[0.0, 0.0, 0.0],
		[1.0, 2.0, 1.0]
	], dtype = float)
	kernel_2 = np.array([
		[-1.0, 0.0, 1.0],
		[-2.0, 0.0, 2.0],
		[-1.0, 0.0, 1.0]
	], dtype = float)

	height, width, _ = image_array.shape
	result_image_array = np.zeros((height, width), dtype = np.uint8)

	gray_image_array = np.copy(image_array) if image_array.ndim == 2 else ep.grayscale(image_array)
	blurred_image_array = convo2d(gray_image_array, gaussian_kernel(kernel_size))

	sobel_1 = convo2d(blurred_image_array, kernel_1, True)
	sobel_2 = convo2d(blurred_image_array, kernel_2, True)

	magnitude_map = (sobel_1 ** 2.0 + sobel_2 ** 2.0) ** 0.5
	angle_map = np.arctan2(sobel_1, sobel_2)

	for i in range(1, height - 1):
		for j in range(1, width - 1):
			if strong <= magnitude_map[i, j] and canny_analyse(magnitude_map, angle_map, i, j):
				result_image_array[i, j] = np.uint8(255)

	for i in range(1, height - 1):
		for j in range(1, width - 1):
			if strong <= magnitude_map[i, j]:
				canny_recursive(result_image_array, magnitude_map, angle_map, strong, weak, i, j)

	return result_image_array


def harmonic_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

	if image_array.ndim == 2:
		height, width = image_array.shape
		result_image_array = np.empty((height, width), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				sum = 0.0
				for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
					for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
						if 0 <= m < height and 0 <= n < width:
							sum += 1.0 / (float(image_array[m, n]) + 1.0)
				result_image_array[i, j] = np.clip(
					1.0 / (sum + 1.0) - 1.0, 0, 255
				).astype(np.uint8)

		return result_image_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		result_image_array = np.empty((height, width, depth), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					sum = 0.0
					for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
						for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
							if 0 <= m < height and 0 <= n < width:
								sum += 1.0 / (float(image_array[m, n, k]) + 1.0)
					result_image_array[i, j, k] = np.clip(
						1.0 / (sum + 1.0) - 1.0, 0, 255
					).astype(np.uint8)

		return result_image_array

	else:
		raise Exception("Not implemented.")


def max_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

	if image_array.ndim == 2:
		height, width = image_array.shape
		result_image_array = np.empty((height, width), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				max = 0
				for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
					for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
						if 0 <= m < height and 0 <= n < width and max < image_array[m, n]:
							max = image_array[m, n]
				result_image_array[i, j] = max.astype(np.uint8)

		return result_image_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		result_image_array = np.empty((height, width, depth), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					max = 0
					for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
						for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
							if 0 <= m < height and 0 <= n < width and max < image_array[m, n, k]:
								max = image_array[m, n, k]
					result_image_array[i, j, k] = max.astype(np.uint8)

		return result_image_array

	else:
		raise Exception("Not implemented.")


def min_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

	if image_array.ndim == 2:
		height, width = image_array.shape
		result_image_array = np.empty((height, width), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				min = 255
				for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
					for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
						if 0 <= m < height and 0 <= n < width and min > image_array[m, n]:
							min = image_array[m, n]
				result_image_array[i, j] = min.astype(np.uint8)

		return result_image_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		result_image_array = np.empty((height, width, depth), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					min = 255
					for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
						for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
							if 0 <= m < height and 0 <= n < width and min > image_array[m, n, k]:
								min = image_array[m, n, k]
					result_image_array[i, j, k] = min.astype(np.uint8)

		return result_image_array

	else:
		raise Exception("Not implemented.")


def min_max_filter(image_array, kernel_size = 3):

	min_image_array = min_filter(image_array, kernel_size)
	min_max_image_array = max_filter(min_image_array, kernel_size)

	return min_max_image_array


def median_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

	if image_array.ndim == 2:
		height, width = image_array.shape
		result_image_array = np.empty((height, width), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				list = []
				for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
					for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
						if 0 <= m < height and 0 <= n < width:
							list.append(image_array[m, n])
						else:
							list.append(0)
				result_image_array[i, j] = np.clip(
					np.median(list), 0, 255
				).astype(np.uint8)

		return result_image_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		result_image_array = np.empty((height, width, depth), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					list = []
					for m in range(i - kernel_pad, i + kernel_size - kernel_pad):
						for n in range(i - kernel_pad, i + kernel_size - kernel_pad):
							if 0 <= m < height and 0 <= n < width:
								list.append(image_array[m, n, k])
							else:
								list.append(0)
					result_image_array[i, j, k] = np.clip(
						np.median(list), 0, 255
					).astype(np.uint8)

		return result_image_array

	else:
		raise Exception("Not implemented.")


def emboss_filter(image_array):

	kernel = np.array([
		[0.0, -1.0, 0.0],
		[1.0, 0.0, -1.0],
		[0.0, 1.0, 0.0]
	], dtype = float)

	return convo2d(image_array, kernel)


def halftone_filter(image_array):

	kernel_size = 4
	kernel = np.array([
		[0, 4, 2, 10],
		[12, 4, 14, 6],
		[3, 11, 1, 9],
		[15, 7, 13, 5]
	], dtype = np.uint8) * np.uint8(16)

	if image_array.ndim == 2:
		height, width = image_array.shape
		result_image_array = np.empty((height, width), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				result_image_array[i, j] = 0 if (
					image_array[i, j] < kernel[i % kernel_size, j % kernel_size]
				) else 255

		return result_image_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		result_image_array = np.empty((height, width, depth), dtype = np.uint8)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					result_image_array[i, j, k] = 0 if (
						image_array[i, j, k] < kernel[i % kernel_size, j % kernel_size]
					) else 255

		return result_image_array

	else:
		raise Exception("Not implemented.")