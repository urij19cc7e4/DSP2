import cv2
import numpy as np
from PIL import Image as img
import os


def custom_filter_generic(image_array, threshold = 0.25):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if r < g and r < b:
				if g >= b and (float(r) / float(g)) < threshold:
					result_image_array[i, j] = 0, 255, 0
				elif g < b and (float(r) / float(b)) < threshold:
					if (float(g) / float(b)) < 0.775:
						result_image_array[i, j] = 0, 0, 255
					else:
						result_image_array[i, j] = 0, 255, 0
				else:
					result_image_array[i, j] = 0, 0, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def custom_filter_mouse_pad(image_array, threshold = 0.25):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if r < g and r < b:
				if g >= b and (float(r) / float(g)) < threshold:
					if (float(b) / float(g)) < 0.75 or (float(b) + float(g)) > 310.0:
						result_image_array[i, j] = 0, 255, 0
				elif g < b and (float(r) / float(b)) < threshold \
					and (r < 15 and g > 95 or r < 30 and g > 120 or r < 45 and g > 145):
					if (float(g) / float(b)) < 0.775:
						result_image_array[i, j] = 0, 0, 255
					else:
						result_image_array[i, j] = 0, 255, 0
				else:
					result_image_array[i, j] = 0, 0, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def custom_filter_remote_empty_null_soft(image_array, threshold = 0.25):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if r < g and r < b:
				if g >= b and (float(r) / float(g)) < threshold:
					result_image_array[i, j] = 0, g, b
				elif g < b and (float(r) / float(b)) < threshold:
					result_image_array[i, j] = 0, g, b
				else:
					result_image_array[i, j] = 0, 0, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def custom_filter_remote_empty_null_ultimate(image_array, threshold = 0.55):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if g == 0 and b == 0:
				result_image_array[i, j] = 0, 0, 0
			elif g >= b and (float(g) / 255.0) > threshold:
				result_image_array[i, j] = 0, 255, 0
			elif g < b and (float(b) / 255.0) > threshold:
				if (float(g) / float(b)) < 0.775:
					result_image_array[i, j] = 0, 0, 255
				else:
					result_image_array[i, j] = 0, 255, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def custom_filter_seven_above_cells_soft(image_array, threshold = 0.40):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if r < g and r < b:
				if g >= b and (float(r) / float(g)) < threshold:
					result_image_array[i, j] = 0, 255, 0
				elif g < b and (float(r) / float(b)) < threshold:
					if (float(g) / float(b)) < 0.775:
						result_image_array[i, j] = 0, 0, 255
					else:
						result_image_array[i, j] = 0, 255, 0
				else:
					result_image_array[i, j] = 0, 0, 0
			elif np.clip(b - 10, 0, 255).astype(np.uint8) < r < g \
				and (float(r) / float(g)) < (threshold * 2.0 + 1.0) \
				and (float(b) / float(g)) < threshold:
				result_image_array[i, j] = 0, 255, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def custom_filter_seven_above_cells_ultimate(image_array, threshold = 0.40):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if g == 0 and b == 0:
				result_image_array[i, j] = 0, 0, 0
			elif g >= b and ((float(g) + float(b)) / 255.0) > threshold:
				result_image_array[i, j] = 0, 255, 0
			elif g < b and ((float(g) + float(b)) / 255.0) > threshold:
				result_image_array[i, j] = 0, 0, 255
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def do_lab_1_task(work_path = "C:/Users/Urij/Downloads/02. Фигурки"):
	files = os.listdir(f"{work_path}")

	for file in files:
		if file.endswith(".jpg"):
			file = file[:-4]

			if file == "1695138157794":
				image = img.open(f"{work_path}/" + file + ".jpg")
				image = image.resize((image.size[0] // 2, image.size[1] // 2))
				image = image.convert('RGB')
				image_array = np.array(image)

				blur_custom_filter = custom_filter_mouse_pad(
					cv2.GaussianBlur(cv2.GaussianBlur(image_array, (9, 9), 0), (5, 5), 0)
				)
				opencv_canny_operator = cv2.Canny(blur_custom_filter, 25, 75)

				img.fromarray(blur_custom_filter).save(
					f"{work_path}/" + file + "_blur_custom_filter.png"
				)
				img.fromarray(opencv_canny_operator).save(
					f"{work_path}/" + file + "_opencv_canny_operator.png"
				)
			elif file == "1695138157752":
				image = img.open(f"{work_path}/" + file + ".jpg")
				image = image.resize((image.size[0] // 2, image.size[1] // 2))
				image = image.convert('RGB')
				image_array = np.array(image)

				blur_custom_filter = custom_filter_remote_empty_null_ultimate(
					cv2.GaussianBlur(cv2.GaussianBlur(cv2.GaussianBlur(cv2.GaussianBlur(
						custom_filter_remote_empty_null_soft(cv2.GaussianBlur(cv2.GaussianBlur(
							image_array, (9, 9), 0
						), (5, 5), 0)), (19, 19), 0
					), (15, 15), 0), (9, 9), 0), (5, 5), 0)
				)
				opencv_canny_operator = cv2.Canny(blur_custom_filter, 25, 50)

				img.fromarray(blur_custom_filter).save(
					f"{work_path}/" + file + "_blur_custom_filter.png"
				)
				img.fromarray(opencv_canny_operator).save(
					f"{work_path}/" + file + "_opencv_canny_operator.png"
				)
			elif file == "1725544579789":
				image = img.open(f"{work_path}/" + file + ".jpg")
				image = image.convert('RGB')
				image_array = np.array(image)

				blur_custom_filter = custom_filter_seven_above_cells_ultimate(
					cv2.GaussianBlur(cv2.GaussianBlur(cv2.GaussianBlur(
						custom_filter_seven_above_cells_soft(cv2.GaussianBlur(
							image_array, (5, 5), 0
						)), (9, 9), 0
					), (7, 7), 0), (5, 5), 0)
				)
				opencv_canny_operator = cv2.Canny(blur_custom_filter, 50, 150)

				img.fromarray(blur_custom_filter).save(
					f"{work_path}/" + file + "_blur_custom_filter.png"
				)
				img.fromarray(opencv_canny_operator).save(
					f"{work_path}/" + file + "_opencv_canny_operator.png"
				)
			else:
				image = img.open(f"{work_path}/" + file + ".jpg")
				image = image.resize((image.size[0] // 2, image.size[1] // 2))
				image = image.convert('RGB')
				image_array = np.array(image)

				blur_custom_filter = custom_filter_generic(
					cv2.GaussianBlur(cv2.GaussianBlur(image_array, (5, 5), 0), (5, 5), 0)
				)
				opencv_canny_operator = cv2.Canny(blur_custom_filter, 25, 75)

				img.fromarray(blur_custom_filter).save(
					f"{work_path}/" + file + "_blur_custom_filter.png"
				)
				img.fromarray(opencv_canny_operator).save(
					f"{work_path}/" + file + "_opencv_canny_operator.png"
				)