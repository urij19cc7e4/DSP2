import cv2
import numpy as np

import elem_wise_proc as ep


def convo2d(image_array, kernel):

	h_kernel, w_kernel = kernel.shape
	h_pad, w_pad = (h_kernel // 2, w_kernel // 2)

	height, width, depth = image_array.shape
	result_image_array = np.empty((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			for k in range(depth):
				sum = 0.0
				for m in range(h_kernel):
					for n in range(w_kernel):
						ii, jj = (i + m - h_pad, j + n - w_pad)
						if 0 <= ii < height and 0 <= jj < width:
							sum += float(image_array[ii, jj, k]) * float(kernel[m, n])
				result_image_array[i, j, k] = np.clip(sum, 0, 255).astype(np.uint8)

	return result_image_array


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

	print("Manual convolution result is equal to OpenCV's: ")
	print(np.all(manual_result == opencv_result))


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
		[1.0, 0.0, -1.0],
		[2.0, 0.0, -2.0],
		[1.0, 0.0, -1.0]
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
		[1.0, 0.0, -1.0],
		[1.0, 0.0, -1.0],
		[1.0, 0.0, -1.0]
	], dtype = float)

	gray_image_array = np.copy(image_array) if image_array.ndim == 2 else ep.grayscale(image_array)

	edge_1 = convo2d(gray_image_array, kernel_1)
	edge_2 = convo2d(gray_image_array, kernel_2)

	height, width = gray_image_array.shape
	result_image_array = np.empty((height, width), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
				result_image_array[i, j] = np.clip(
					max(edge_1[i, j], edge_2[i, j]), 0, 255
				).astype(np.uint8)

	return result_image_array


def laplacian_operator(image_array):
	return (cv2.Laplacian(image_array, cv2.CV_64F)).astype(np.uint8)


def canny_operator(image_array, strong = 125, weak = 75):
	return (cv2.Canny(image_array, weak, strong)).astype(np.uint8)


def harmonic_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

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
					1.0 / sum - 1.0, 0, 255
				).astype(np.uint8)

	return result_image_array


def max_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

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


def min_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

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


def min_max_filter(image_array, kernel_size = 3):

	min_image_array = min_filter(image_array, kernel_size)
	min_max_image_array = max_filter(min_image_array, kernel_size)

	return min_max_image_array


def median_filter(image_array, kernel_size = 3):

	kernel_pad = kernel_size // 2

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
	], dtype = np.uint8) * 16

	height, width, depth = image_array.shape
	result_image_array = np.empty((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			for k in range(depth):
				result_image_array[i, j, k] = 0 if (
					image_array[i, j, k] < kernel[i % kernel_size, j % kernel_size]
				) else 255

	return result_image_array