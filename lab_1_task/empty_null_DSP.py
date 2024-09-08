import cv2
import numpy as np
from PIL import Image as img

def custom_filter_empty_null(image_array, threshold = 0.25):

	height, width, depth = image_array.shape
	result_image_array = np.zeros((height, width, depth), dtype = np.uint8)

	for i in range(height):
		for j in range(width):
			r, g, b = image_array[i, j]

			if r < g and r < b:
				if g > b and (float(r) / float(g)) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.75, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 1.75 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * (-1.75) + 1.0), 0, 255).astype(np.uint8)
				elif g < b and (float(r) / float(b)) < threshold and g < 193 and b > 215:
					result_image_array[i, j] = np.clip(float(r) * 0.75, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * (-1.75) + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * 0.05 + 1.0), 0, 255).astype(np.uint8)
				elif g == b and (float(r) / (float(g) + float(b))) < threshold:
					result_image_array[i, j] = np.clip(float(r) * 0.75, 0, 255).astype(np.uint8), \
						np.clip(float(g) * (threshold * 1.75 + 1.0), 0, 255).astype(np.uint8), \
						np.clip(float(b) * (threshold * 1.75 + 1.0), 0, 255).astype(np.uint8)
				else:
					result_image_array[i, j] = 0, 0, 0
			else:
				result_image_array[i, j] = 0, 0, 0

	return result_image_array


file = "1695138157752"

image = img.open("C:/Users/Urij/Downloads/02. Фигурки/" + file + ".jpg")
image = image.resize((image.size[0] // 2, image.size[1] // 2))
image = image.convert('RGB')
image_array = np.array(image)

blur_custom_filter = custom_filter_empty_null(
	cv2.GaussianBlur(cv2.GaussianBlur(image_array, (5, 5), 0), (5, 5), 0)
)
opencv_canny_operator = cv2.Canny(blur_custom_filter, 25, 50)

img.fromarray(blur_custom_filter).save(
	"C:/Users/Urij/Downloads/02. Фигурки/empty_null_" + file + "_blur_custom_filter.jpg"
)
img.fromarray(opencv_canny_operator).save(
	"C:/Users/Urij/Downloads/02. Фигурки/empty_null_" + file + "_opencv_canny_operator.jpg"
)