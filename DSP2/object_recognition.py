from collections import defaultdict
import numpy as np
import sys


# FOR RECURSIVE IMPLEMENTATION
# INCREASE IN CASE OF STACKOVERFLOW
sys.setrecursionlimit(65535)


# OBJECT(S) OUTER EDGE(S) SHOULD NOT BE OUTSIDE OF IMAGE OR OVERLAP
# GRAYSCALING(#1) AND BINARIZATION(#2) ARE REQUIRED
# 255 level is for object(s) (WHITE)
# 0 level is for background (BLACK)
# OBJECT MAP STRUCTURE:
# lvl0 - 0 is for background, # is for # object
# lvl1 - 0 is for background, # is for # object inner edge
# lvl2 - 0 is for background, # is for # object outer edge
# OBJECT COUNT CAN BE WRONG IF OBJECT(S) HOLEY/OVERLAP/CONNECTED


def scan_object_recursive(image_array, object_map, object_count, ii, jj):

	if object_map[ii, jj, 0] == 0:
		height, width = image_array.shape
		if image_array[ii, jj] == 255:
			object_map[ii, jj, 0] = object_count

			if ii == 0 or ii == height - 1 or jj == 0 or jj == width - 1 \
				or image_array[ii + 1, jj] == 0 or image_array[ii - 1, jj] == 0 \
				or image_array[ii, jj + 1] == 0 or image_array[ii, jj - 1] == 0:
				object_map[ii, jj, 1] = object_count

			ii_array = np.array([ii - 1, ii - 1, ii - 1, ii, ii, ii + 1, ii + 1, ii + 1])
			jj_array = np.array([jj - 1, jj, jj + 1, jj - 1, jj + 1, jj - 1, jj, jj + 1])

			for k in range(8):
				i, j = ii_array[k], jj_array[k]
				if 0 <= i < height and 0 <= j < width:
					scan_object_recursive(image_array, object_map, object_count, i, j)

		elif image_array[ii, jj] == 0:
			if ii < height - 1 and image_array[ii + 1, jj] == 255 \
				or ii > 0 and image_array[ii - 1, jj] == 255 \
				or jj < width - 1 and image_array[ii, jj + 1] == 255 \
				or jj > 0 and image_array[ii, jj - 1] == 255:
				object_map[ii, jj, 2] = object_count

		else:
			raise Exception("Image must be binarized.")


def recognize_recursive(image_array):

	if image_array.ndim == 2:
		height, width = image_array.shape
		object_map = np.zeros((height, width, 3), dtype = np.uint64)
		object_count = np.uint64(0)

		for i in range(height):
			for j in range(width):
				if image_array[i, j] == 255 and object_map[i, j, 0] == 0:
					object_count += 1
					scan_object_recursive(image_array, object_map, object_count, i, j)

		return object_map, int(object_count)

	else:
		raise Exception("Image must be grayscaled.")


# Written by chatgpt
# Function to find all connected components
def find_connected_component(neighbors, node, visited):
	stack = [node]
	component = set()
	while stack:
		current = stack.pop()
		if current not in visited:
			visited.add(current)
			component.add(current)
			stack.extend(neighbors[current] - visited)
	return component


# Written by chatgpt
# Example usage
# Input: [[11, 27, 13], [11, 27, 55], [22, 0, 43], [22, 0, 96],
#		  [13, 27, 11], [13, 27, 55], [43, 0, 22], [43, 0, 96], [55, 27, 11]]
# Output: [[0, 43, 96, 22], [27, 11, 13, 55]]
def merge_lists_with_common_elements(lists):

	# Create a dictionary to store the neighbors of each element
	neighbors = defaultdict(set)

	# Populate the neighbors dictionary
	for sublist in lists:
		for item in sublist:
			neighbors[item].update(sublist)

	# Find all unique components
	visited = set()
	components = []
	for node in neighbors:
		if node not in visited:
			component = find_connected_component(neighbors, node, visited)
			components.append(component)

	# Convert components back to list of lists
	result = [list(component) for component in components]
	return result


def recognize_sequential(image_array):

	if image_array.ndim == 2:
		height, width = image_array.shape
		object_map = np.zeros((height, width, 3), dtype = np.uint64)
		object_count = np.uint64(0)
		objects_parts_lists = []

		for i in range(height):
			for j in range(width):
				if image_array[i, j] == 255:
					i_same_j_prev = object_map[i, j - 1, 0]
					i_prev_j_next = object_map[i - 1, j + 1, 0]
					i_prev_j_same = object_map[i - 1, j, 0]
					i_prev_j_prev = object_map[i - 1, j - 1, 0]
					connections = np.uint64(i_same_j_prev != 0) + np.uint64(i_prev_j_next != 0) \
						+ np.uint64(i_prev_j_same != 0) + np.uint64(i_prev_j_prev != 0)

					if connections == 0:
						object_count += 1
						object_map[i, j, 0] = object_count
						objects_parts_lists.append([object_count])

					elif connections == 1:
						object_map[i, j, 0] = np.uint64(
							i_same_j_prev | i_prev_j_next | i_prev_j_same | i_prev_j_prev
						)

					else:
						object_map[i, j, 0] = i_same_j_prev if i_same_j_prev != 0 else (
							i_prev_j_same if i_prev_j_same != 0 else i_prev_j_prev
						)

						for object_parts_list in objects_parts_lists:
							if i_same_j_prev != 0 and i_same_j_prev in object_parts_list:
								if i_prev_j_next != 0 and i_prev_j_next not in object_parts_list:
									object_parts_list.append(i_prev_j_next)
								if i_prev_j_same != 0 and i_prev_j_same not in object_parts_list:
									object_parts_list.append(i_prev_j_same)
								if i_prev_j_prev != 0 and i_prev_j_prev not in object_parts_list:
									object_parts_list.append(i_prev_j_prev)
								break
							if i_prev_j_next != 0 and i_prev_j_next in object_parts_list:
								if i_same_j_prev != 0 and i_same_j_prev not in object_parts_list:
									object_parts_list.append(i_same_j_prev)
								if i_prev_j_same != 0 and i_prev_j_same not in object_parts_list:
									object_parts_list.append(i_prev_j_same)
								if i_prev_j_prev != 0 and i_prev_j_prev not in object_parts_list:
									object_parts_list.append(i_prev_j_prev)
								break
							if i_prev_j_same != 0 and i_prev_j_same in object_parts_list:
								if i_same_j_prev != 0 and i_same_j_prev not in object_parts_list:
									object_parts_list.append(i_same_j_prev)
								if i_prev_j_next != 0 and i_prev_j_next not in object_parts_list:
									object_parts_list.append(i_prev_j_next)
								if i_prev_j_prev != 0 and i_prev_j_prev not in object_parts_list:
									object_parts_list.append(i_prev_j_prev)
								break
							if i_prev_j_prev != 0 and i_prev_j_prev in object_parts_list:
								if i_same_j_prev != 0 and i_same_j_prev not in object_parts_list:
									object_parts_list.append(i_same_j_prev)
								if i_prev_j_next != 0 and i_prev_j_next not in object_parts_list:
									object_parts_list.append(i_prev_j_next)
								if i_prev_j_same != 0 and i_prev_j_same not in object_parts_list:
									object_parts_list.append(i_prev_j_same)
								break

				elif image_array[i, j] != 0:
					raise Exception("Image must be binarized.")

		merged_parts_lists = merge_lists_with_common_elements(objects_parts_lists)
		object_count = len(merged_parts_lists)

		for i in range(height):
			for j in range(width):
				if object_map[i, j, 0] != 0:
					for k, merged_parts_list in enumerate(merged_parts_lists):
						if object_map[i, j, 0] in merged_parts_list:
							object_map[i, j, 0] = k + 1

		for i in range(height):
			for j in range(width):
				if image_array[i, j] == 255:
					if i == 0 or i == height - 1 or j == 0 or j == width - 1 \
						or image_array[i + 1, j] == 0 or image_array[i - 1, j] == 0 \
						or image_array[i, j + 1] == 0 or image_array[i, j - 1] == 0:
						object_map[i, j, 1] = object_map[i, j, 0]

				elif image_array[i, j] == 0:
					if i < height - 1 and image_array[i + 1, j] == 255:
						object_map[i, j, 2] = object_map[i + 1, j, 0]
					elif i > 0 and image_array[i - 1, j] == 255:
						object_map[i, j, 2] = object_map[i - 1, j, 0]
					elif j < width - 1 and image_array[i, j + 1] == 255:
						object_map[i, j, 2] = object_map[i, j + 1, 0]
					elif j > 0 and image_array[i, j - 1] == 255:
						object_map[i, j, 2] = object_map[i, j - 1, 0]

				else:
					raise Exception("Image must be binarized.")

		return object_map, int(object_count)

	else:
		raise Exception("Image must be grayscaled.")


def split_objects(image_array, objects_map, objects_count):

	height, width, _ = objects_map.shape
	objects_params = np.tile([height, width, 0, 0], (objects_count, 4))
	objects_list = []
	edges_list = []

	for i in range(height):
		for j in range(width):
			if objects_map[i, j, 0] != 0:
				index = np.uint64(objects_map[i, j, 0] - 1)

				if objects_params[index, 0] > i:
					objects_params[index, 0] = i
				if objects_params[index, 1] > j:
					objects_params[index, 1] = j
				if objects_params[index, 2] < i:
					objects_params[index, 2] = i
				if objects_params[index, 3] < j:
					objects_params[index, 3] = j

	for i in range(objects_count):
		index = i + 1
		o_height = objects_params[i, 2] - objects_params[i, 0] + 1
		o_width = objects_params[i, 3] - objects_params[i, 1] + 1
		object_array = np.zeros((o_height, o_width), dtype = np.uint8)
		edges_array = np.zeros((o_height, o_width), dtype = np.uint8)

		for ii in range(o_height):
			for jj in range(o_width):
				index_1 = np.uint64(objects_params[i, 0] + ii)
				index_2 = np.uint64(objects_params[i, 1] + jj)

				if objects_map[index_1, index_2, 0] == index:
					object_array[ii, jj] = image_array[index_1, index_2]
				if objects_map[index_1, index_2, 1] == index:
					edges_array[ii, jj] = np.uint8(255)

		objects_list.append(object_array)
		edges_list.append(edges_array)

	return objects_list, edges_list