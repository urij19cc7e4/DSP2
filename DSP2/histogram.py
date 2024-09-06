import numpy as np
from matplotlib import pyplot as plt

def histogram(image_array):

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


def plot_image(image_array, wnd_string = "Another plot", histo_plot = True):

	if histo_plot:
		histogram_array = histogram(image_array)

		fig, axes = plt.subplots(1, 2, figsize = (6, 3))
		fig.canvas.manager.set_window_title(wnd_string)

		if image_array.ndim == 2:
			axes[0].imshow(image_array, cmap = "gray")
			axes[1].plot(histogram_array, color = "gray")
		elif image_array.ndim == 3:
			axes[0].imshow(image_array)
			axes[1].plot(histogram_array[0], color = "red")
			axes[1].plot(histogram_array[1], color = "green")
			axes[1].plot(histogram_array[2], color = "blue")
			axes[1].plot(histogram_array[3], color = "gray")
		else:
			raise Exception("Shit happens.")

		axes[0].axis("off")
		axes[0].set_title("Image")
		axes[1].set_xlim([0, 255])
		axes[1].set_title("Histogram")
		axes[1].set_xlabel("Value")
		axes[1].set_ylabel("Frequency")

		plt.tight_layout()
		plt.show()

	else:

		fig, axes = plt.subplots(1, 1, figsize = (3, 3))
		fig.canvas.manager.set_window_title(wnd_string)

		axes[0].imshow(image_array)
		axes[0].axis("off")
		axes[0].set_title("Image")

		plt.tight_layout()
		plt.show()