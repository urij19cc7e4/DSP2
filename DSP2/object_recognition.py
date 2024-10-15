from dataclasses import dataclass
from collections import defaultdict
from matplotlib import gridspec as gsp
from matplotlib import pyplot as plt
from scipy.ndimage import laplace

import cv2
import math
import numpy as np
import random as rnd
import sys

import object_feature as oft


# FOR RECURSIVE IMPLEMENTATION
# INCREASE IN CASE OF STACKOVERFLOW
sys.setrecursionlimit(1048575)


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
					i_same_j_prev = object_map[i, j - 1, 0] if j > 0 else 0
					i_prev_j_next = object_map[i - 1, j + 1, 0] if i > 0 and j < width - 1 else 0
					i_prev_j_same = object_map[i - 1, j, 0] if i > 0 else 0
					i_prev_j_prev = object_map[i - 1, j - 1, 0] if i > 0 and j > 0 else 0
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


def bresenham_line(image_array, x0, y0, x1, y1):

	dx = abs(x1 - x0)
	dy = abs(y1 - y0)

	sx = 1 if x0 < x1 else -1
	sy = 1 if y0 < y1 else -1

	err = dx - dy

	while x0 != x1 or y0 != y1:
		image_array[x0, y0] = np.uint8(0)

		if err * 2 > -dy:
			err -= dy
			x0 += sx
		elif err * 2 < dx:
			err += dx
			y0 += sy

	image_array[x0, y0] = np.uint8(0)


def search_recursive(check_array, dist_array, __abs_tol__, ii, jj, ii_prev, jj_prev, ir, jr):

	if not check_array[ii, jj]:
		check_array[ii, jj] = True
		ii_array = np.array([ii, ii - 1, ii - 1, ii - 1, ii, ii + 1, ii + 1, ii + 1])
		jj_array = np.array([jj - 1, jj - 1, jj, jj + 1, jj + 1, jj + 1, jj, jj - 1])

		for k in range(ir, jr):
			kk = k % 8

			if math.isclose(dist_array[ii, jj], dist_array[ii_array[kk], jj_array[kk]], abs_tol = __abs_tol__):
				if not math.isclose(0.0, dist_array[ii_array[kk], jj_array[kk]], abs_tol = __abs_tol__) and not (ii_array[kk] == ii_prev or jj_array[kk] == jj_prev):
					if search_recursive(check_array, dist_array, __abs_tol__, ii_array[kk], jj_array[kk], ii, jj, ir, jr) == 1:
						return 1
			else:
				if dist_array[ii, jj] < dist_array[ii_array[kk], jj_array[kk]]:
					return 1

	return 0


def separate_recursive(dist_array, ii, jj):

	ii_array = np.array([ii, ii - 1, ii - 1, ii - 1, ii, ii + 1, ii + 1, ii + 1])
	jj_array = np.array([jj - 1, jj - 1, jj, jj + 1, jj + 1, jj + 1, jj, jj - 1])

	next_coords = 0

	for k in range(1, 8):
		if dist_array[ii_array[next_coords], jj_array[next_coords]] > dist_array[ii_array[k], jj_array[k]]:
			next_coords = k

	if dist_array[ii, jj] > dist_array[ii_array[next_coords], jj_array[next_coords]]:
		return separate_recursive(dist_array, ii_array[next_coords], jj_array[next_coords])
	else:
		return ii, jj


def separate_objects(image_array, thickness = 0.1):

	height, width = image_array.shape
	dist_array = cv2.distanceTransform(image_array, cv2.DIST_L2, 0)
	laplace_array = laplace(dist_array)

	list_coord = []
	max_dist = (float(height) ** 2.0 + float(width) ** 2.0) ** 0.5
	abs_thickness = np.uint64(max_dist * thickness * 0.5)
	__abs_tol__ = 1.0 / max_dist

	for i in range(2, height - 2):
		for j in range(2, width - 2):
			if dist_array[i, j] < abs_thickness:
				i_array = np.array([i, i - 1, i - 1, i - 1, i, i + 1, i + 1, i + 1])
				j_array = np.array([j - 1, j - 1, j, j + 1, j + 1, j + 1, j, j - 1])
				local_max_min_array = np.empty(8, dtype = int)

				for k in range(8):
					if math.isclose(dist_array[i, j], dist_array[i_array[k], j_array[k]], abs_tol = __abs_tol__):
						check_array = np.zeros((height, width), dtype = bool)

						if search_recursive(check_array, dist_array, __abs_tol__, i_array[k], j_array[k], i, j, k - 2, k + 3) == 1:
							local_max_min_array[k] = 2
						else:
							local_max_min_array[k] = 0
					else:
						local_max_min_array[k] = 1 if dist_array[i, j] > dist_array[i_array[k], j_array[k]] else -1

				start_index = 0
				for k in range(1, 8):
					if local_max_min_array[k] != local_max_min_array[k - 1]:
						start_index = k
						break

				lmm_concatenated = [local_max_min_array[start_index]]
				for val in np.roll(local_max_min_array, -start_index)[1:]:
					if val == 2 and lmm_concatenated[-1] == 0:
						lmm_concatenated[-1] = 2
					elif val != lmm_concatenated[-1]:
						lmm_concatenated.append(val)

				lmm_count = 0
				for k in range(len(lmm_concatenated)):
					if lmm_concatenated[k] == 0 or lmm_concatenated[k] == 2:
						if lmm_concatenated[k - 1] == lmm_concatenated[(k + 1) % len(lmm_concatenated)] == 1 and lmm_concatenated[k] == 2:
							lmm_count += 1
						elif lmm_concatenated[k - 1] == lmm_concatenated[(k + 1) % len(lmm_concatenated)] == -1:
							lmm_count += 1
						elif lmm_concatenated[k - 1] == lmm_concatenated[(k + 1) % len(lmm_concatenated)]:
							lmm_count -= 1
					else:
						lmm_count += 1

				if lmm_count == 4:
					list_coord.append((i, j))

	final_list_coord = []
	group_list_coord = []

	for val in list_coord:
		new_val = True

		for g_lst in group_list_coord:
			for g_val in g_lst:
				avg_dist = float(dist_array[g_val] + dist_array[val]) / 2.0
				val_dist = (float(g_val[0] - val[0]) ** 2.0 + float(g_val[1] - val[1]) ** 2.0) ** 0.5

				if avg_dist > val_dist:
					g_lst.append(val)
					new_val = False
					break

			if not new_val:
				break

		if new_val:
			group_list_coord.append([val])

	for g_lst in group_list_coord:
		val = g_lst[0]

		for g_val in g_lst:
			if laplace_array[val] > laplace_array[g_val]:
				val = g_val

		final_list_coord.append(val)

	for f_val in final_list_coord:
		ii_array = np.array([f_val[0], f_val[0] - 1, f_val[0] - 1, f_val[0] - 1, f_val[0], f_val[0] + 1, f_val[0] + 1, f_val[0] + 1])
		jj_array = np.array([f_val[1] - 1, f_val[1] - 1, f_val[1], f_val[1] + 1, f_val[1] + 1, f_val[1] + 1, f_val[1], f_val[1] - 1])

		next_coords = 0

		for k in range(1, 8):
			if dist_array[ii_array[next_coords], jj_array[next_coords]] > dist_array[ii_array[k], jj_array[k]]:
				next_coords = k

		line_pt_1 = separate_recursive(dist_array, ii_array[next_coords], jj_array[next_coords])
		line_pt_2 = separate_recursive(dist_array, ii_array[(next_coords + 4) % 8], jj_array[(next_coords + 4) % 8])

		bresenham_line(image_array, line_pt_1[0], line_pt_1[1], line_pt_2[0], line_pt_2[1])


def split_objects(image_array, object_map, object_count):

	height, width, _ = object_map.shape
	objects_params = np.tile([height, width, 0, 0], (object_count, 4))
	objects_list = []
	edges_list = []

	for i in range(height):
		for j in range(width):
			if object_map[i, j, 0] != 0:
				index = np.uint64(object_map[i, j, 0] - 1)

				if objects_params[index, 0] > i:
					objects_params[index, 0] = i
				if objects_params[index, 1] > j:
					objects_params[index, 1] = j
				if objects_params[index, 2] < i:
					objects_params[index, 2] = i
				if objects_params[index, 3] < j:
					objects_params[index, 3] = j

	for i in range(object_count):
		index = i + 1
		o_height = objects_params[i, 2] - objects_params[i, 0] + 1
		o_width = objects_params[i, 3] - objects_params[i, 1] + 1
		object_array = np.zeros((o_height, o_width), dtype = np.uint8)
		edges_array = np.zeros((o_height, o_width), dtype = np.uint8)

		for ii in range(o_height):
			for jj in range(o_width):
				index_1 = np.uint64(objects_params[i, 0] + ii)
				index_2 = np.uint64(objects_params[i, 1] + jj)

				if object_map[index_1, index_2, 0] == index:
					object_array[ii, jj] = image_array[index_1, index_2]
				if object_map[index_1, index_2, 1] == index:
					edges_array[ii, jj] = np.uint8(255)

		objects_list.append(object_array)
		edges_list.append(edges_array)

	return objects_list, edges_list


@dataclass
class object_info:
	__class__: str = ""
	axis_angle: float = 0.0
	eccentricity: float = 0.0
	center_x: float = 0.0
	center_y: float = 0.0
	density: float = 0.0
	perimeter: float = 0.0
	square: float = 0.0


# Written by chatgpt
def calculate_intersections(x_max, x_min, y_max, y_min, theta):

	m = math.tan(theta)
	
	if theta >= 0:
		# Intersection with right boundary
		y_right = m * x_max
		if y_min <= y_right <= y_max:
			line_x_max = x_max
			line_y_max = y_right
		else:
			line_x_max = y_max / m
			line_y_max = y_max
		
		# Intersection with bottom boundary
		y_left = m * x_min
		if y_min <= y_left <= y_max:
			line_x_min = x_min
			line_y_min = y_left
		else:
			line_x_min = y_min / m
			line_y_min = y_min
	else:
		# Intersection with left boundary
		y_left = m * x_min
		if y_min <= y_left <= y_max:
			line_x_min = x_min
			line_y_min = y_left
		else:
			line_x_min = y_min / m
			line_y_min = y_min
		
		# Intersection with top boundary
		y_right = m * x_max
		if y_min <= y_right <= y_max:
			line_x_max = x_max
			line_y_max = y_right
		else:
			line_x_max = y_max / m
			line_y_max = y_max
	
	return int(round(line_x_min)), int(round(line_y_min)), \
		   int(round(line_x_max)), int(round(line_y_max))


def plot_objects(image_array):

	height, width = image_array.shape
	separate_objects(image_array, 0.075)
	object_map, object_count = recognize_recursive(image_array)
	objects_list, edges_list = split_objects(image_array, object_map, object_count)

	bright_colors = ([
		[255, 0, 0],
		[0, 255, 0],
		[0, 0, 255],
		[255, 255, 0],
		[0, 255, 255],
		[255, 0, 255],
		[255, 127, 0],
		[127, 255, 0],
		[255, 100, 155]
	])

	color_array = np.array([
		[bright_colors[k % 8]] for k in range(object_count)
	])

	objects_image_array = np.empty((height, width, 3), dtype = np.uint8)
	object_infos = []

	features_1 = np.empty(object_count, dtype = float)
	features_2 = np.empty(object_count, dtype = float)

	for i in range(object_count):
		object_info_tmp = object_info()

		object_info_tmp.center_x, object_info_tmp.center_y = oft.calc_center(objects_list[i])
		object_info_tmp.axis_angle, object_info_tmp.eccentricity = oft.calc_axis_eccentricity(
			objects_list[i], object_info_tmp.center_x, object_info_tmp.center_y
		)
		object_info_tmp.perimeter = oft.calc_perimeter(edges_list[i])
		object_info_tmp.square = oft.calc_square(objects_list[i])
		object_info_tmp.density = oft.calc_density(object_info_tmp.perimeter, object_info_tmp.square)

		# VALUES FOR PICTURE OF FIGURES
		features_1[i] = object_info_tmp.density
		features_2[i] = object_info_tmp.square

		## VALUES FOR PICTURE IN MANUAL (3 screws, cube, ring, and plate)
		## would separate screws from 'rounded' objects
		## but no need of logs in lab tests, 0.0 can be changed to density
		## do not forget about metric properties of the Euclidean feature space
		#features_1[i] = math.log(math.log(object_info_tmp.eccentricity, 10), 10)
		#features_2[i] = 0.0

		object_infos.append(object_info_tmp)

	classes = oft.k_means(features_1, features_2, 2)

	for i in range(object_count):
		object_infos[i].__class__ = f"{chr(ord('A') + int(classes[i]))}"
	
	rgb_objects_list = []

	for i in range(object_count):
		o_height, o_width = objects_list[i].shape
		rgb_object_array = np.empty((o_height, o_width, 3), dtype = np.uint8)

		for j in range(o_height):
			for k in range(o_width):
				rgb_object_array[j, k] = np.uint8(255) - objects_list[i][j, k]

		axis_angle = object_infos[i].axis_angle
		center_x, center_y = int(round(object_infos[i].center_x)), int(round(object_infos[i].center_y))
		x_max, x_min = min(center_x + o_height // 4, o_height - 1), max(center_x - o_height // 4, 0)
		y_max, y_min = min(center_y + o_width // 4, o_width - 1), max(center_y - o_width // 4, 0)

		cross_x_max, cross_x_min = min(center_x + o_height // 8, o_height - 1), max(center_x - o_height // 8, 0)
		cross_y_max, cross_y_min = min(center_y + o_width // 8, o_width - 1), max(center_y - o_width // 8, 0)

		intersections = calculate_intersections(
			x_max - center_x, x_min - center_x, y_max - center_y, y_min - center_y, axis_angle
		)
		line_pt_1 = intersections[1] + center_y, intersections[0] + center_x
		line_pt_2 = intersections[3] + center_y, intersections[2] + center_x
		thickness = max(1, int(round(max(o_height, o_width) / 375.0)))

		cv2.line(rgb_object_array, line_pt_1, line_pt_2, (0, 255, 0), thickness)

		cv2.line(rgb_object_array, (cross_y_max, center_x), (cross_y_min, center_x), (255, 0, 0), thickness)
		cv2.line(rgb_object_array, (center_y, cross_x_max), (center_y, cross_x_min), (255, 0, 0), thickness)

		rgb_objects_list.append(rgb_object_array)

	for i in range(height):
		for j in range(width):
			if object_map[i, j, 0] != 0:
				objects_image_array[i, j] = color_array[np.uint64(object_map[i, j, 0] - 1)]
			elif object_map[i, j, 0] != 0:
				objects_image_array[i, j] = [np.uint8(255) - image_array[i, j] for _ in range(3)]
			else:
				objects_image_array[i, j] = np.uint8(255), np.uint8(255), np.uint8(255)

	fig = plt.figure(figsize = (10, 5))
	gs = gsp.GridSpec(object_count, 3, width_ratios = [4, 1, 1])

	ax_big = plt.subplot(gs[:, 0])
	ax_big.imshow(objects_image_array)
	ax_big.axis("off")

	for i in range(object_count):
		o_title = f"CLASS {object_infos[i].__class__}\n" \
			f"Eccentricity: {object_infos[i].eccentricity:.2f}\n" \
			f"Density: {object_infos[i].density:.2f}\n" \
			f"Perimeter: {object_infos[i].perimeter:.2f}\n" \
			f"Square: {object_infos[i].square:.2f}\n"

		ax_small = plt.subplot(gs[i, 1])
		ax_small.imshow(rgb_objects_list[i])
		ax_small.axis("off")

		ax_small_2 = plt.subplot(gs[i, 2])
		ax_small_2.text(0.0, 0.0, o_title, horizontalalignment = "center", verticalalignment = "center",
						transform = ax_small_2.transAxes, fontsize = 8, color = "black")
		ax_small_2.axis("off")

	plt.tight_layout()
	plt.show()