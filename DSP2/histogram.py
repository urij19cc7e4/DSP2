import numpy as np
from matplotlib import pyplot as plt

def histogram(image_array: np.ndarray) -> np.ndarray:

	if image_array.ndim == 2:
		height, width = image_array.shape
		result_array = np.zeros(256, dtype = np.uint64)

		for i in range(height):
			for j in range(width):
				result_array[image_array[i, j]] += 1

		return result_array

	elif image_array.ndim == 3:
		height, width, depth = image_array.shape
		result_array = np.zeros((depth + 1, 256), dtype = np.uint64)
		temp_array = np.zeros(256, dtype = float)

		for i in range(height):
			for j in range(width):
				for k in range(depth):
					result_array[k, image_array[i, j, k]] += 1

					if k == 0:
						temp_array[image_array[i, j, k]] += 0.299
					elif k == 1:
						temp_array[image_array[i, j, k]] += 0.587
					elif k == 2:
						temp_array[image_array[i, j, k]] += 0.114
					else:
						raise Exception("Shit happens.")

		for i in range(256):
			result_array[depth, i] = round(temp_array[i])

		return result_array

	else:
		raise Exception("Not implemented.")


def plot_image(image_list: list[np.ndarray], wnd_string = "Plot", histo_plot = True):

	count = len(image_list)

	if histo_plot:
		fig, axes = plt.subplots(count, 2, figsize = (6, count * 3))
		fig.canvas.manager.set_window_title(wnd_string)

		for i in range(count):
			axes_num = i * 2
			histogram_array = histogram(image_list[i])

			if image_list[i].ndim == 2:
				axes[axes_num + 0].imshow(image_list[i], cmap = "gray")
				axes[axes_num + 1].plot(histogram_array, color = "gray")
			elif image_list[i].ndim == 3:
				axes[axes_num + 0].imshow(image_list[i])
				axes[axes_num + 1].plot(histogram_array[0], color = "red")
				axes[axes_num + 1].plot(histogram_array[1], color = "green")
				axes[axes_num + 1].plot(histogram_array[2], color = "blue")
				axes[axes_num + 1].plot(histogram_array[3], color = "gray")
			else:
				raise Exception("Shit happens.")

			axes[axes_num + 0].axis("off")
			axes[axes_num + 0].set_title("Image")

			axes[axes_num + 1].set_title("Histogram")
			axes[axes_num + 1].set_xlabel("Value")
			axes[axes_num + 1].set_ylabel("Frequency")
			axes[axes_num + 1].set_xlim([0, 255])

		plt.tight_layout()
		plt.show()

	else:
		fig, axes = plt.subplots(count, 1, figsize = (3, count * 3))
		fig.canvas.manager.set_window_title(wnd_string)

		for i in range(count):
			if image_list[i].ndim == 2:
				axes[i].imshow(image_list[i], cmap = "gray")
			elif image_list[i].ndim == 3:
				axes[i].imshow(image_list[i])
			else:
				raise Exception("Shit happens.")

			axes[i].axis("off")
			axes[i].set_title("Image")

		plt.tight_layout()
		plt.show()