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
				if g > b and (float(r) / float(g)) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 2.25 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * (-2.25) + 1.0), 0, 255).astype(np.uint8)
				elif g < b and (float(r) / float(b)) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * (-2.25) + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * 2.25 + 1.0), 0, 255).astype(np.uint8)
				elif g == b and (float(r) / (float(g) + float(b))) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 2.25 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * 2.25 + 1.0), 0, 255).astype(np.uint8)
				else:
					result_image_array[i, j] = 0, 0, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


def custom_filter_seven_above_cells(image_array, threshold = 0.4):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if r < g and r < b:
				if g > b and (float(r) / float(g)) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 3.5 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * (-3.5) + 1.0), 0, 255).astype(np.uint8)
				elif g < b and (float(r) / float(b)) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * (-3.5) + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * 3.5 + 1.0), 0, 255).astype(np.uint8)
				elif g == b and (float(r) / (float(g) + float(b))) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 3.5 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * 3.5 + 1.0), 0, 255).astype(np.uint8)
				else:
					result_image_array[i, j] = 0, 0, 0
			elif np.clip(b - 10, 0, 255).astype(np.uint8) < r < g \
				and (float(b) / float(g)) < threshold and (float(r) / float(g)) < (threshold * 2.0 + 1.0):
				result_image_array[i, j] = np.clip(float(r) * 0.5, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 3.5 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * (-3.5) + 1.0), 0, 255).astype(np.uint8)
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


files = os.listdir("C:/Users/Urij/Downloads/02. Фигурки")

for file in files:
	if file.endswith(".jpg"):
		file = file[:-4]

		if file == "1725544579789":
			image = img.open("C:/Users/Urij/Downloads/02. Фигурки/" + file + ".jpg")
			image = image.convert('RGB')
			image_array = np.array(image)

			blur_custom_filter = custom_filter_seven_above_cells(cv2.GaussianBlur(image_array, (9, 9), 0))
			opencv_canny_operator = cv2.Canny(blur_custom_filter, 50, 150)

			img.fromarray(blur_custom_filter).save(
				"C:/Users/Urij/Downloads/02. Фигурки/" + file + "_blur_custom_filter.jpg"
			)
			img.fromarray(opencv_canny_operator).save(
				"C:/Users/Urij/Downloads/02. Фигурки/" + file + "_opencv_canny_operator.jpg"
			)
		else:
			image = img.open("C:/Users/Urij/Downloads/02. Фигурки/" + file + ".jpg")
			image = image.resize((image.size[0] // 2, image.size[1] // 2))
			image = image.convert('RGB')
			image_array = np.array(image)

			blur_custom_filter = custom_filter_generic(cv2.GaussianBlur(image_array, (9, 9), 0))
			opencv_canny_operator = cv2.Canny(blur_custom_filter, 25, 75)

			img.fromarray(blur_custom_filter).save(
				"C:/Users/Urij/Downloads/02. Фигурки/" + file + "_blur_custom_filter.jpg"
			)
			img.fromarray(opencv_canny_operator).save(
				"C:/Users/Urij/Downloads/02. Фигурки/" + file + "_opencv_canny_operator.jpg"
			)