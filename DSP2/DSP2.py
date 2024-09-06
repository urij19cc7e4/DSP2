import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import elem_wise_proc as ep
import filter_proc as fp
import histogram as hst

image = Image.open('C:/Users/Urij/Downloads/photo_2024-09-03_19-43-39.jpg')
image = image.convert('RGB')
image_array = np.array(image)

#hst.plot_image(image_array)
#hst.plot_image(ep.linear_correction(image_array))
#hst.plot_image(ep.gamma_correction(image_array))
#hst.plot_image(ep.log_correction(image_array))
#hst.plot_image(ep.grayscale(image_array))
#hst.plot_image(ep.cut_window_preparation(image_array))
#hst.plot_image(ep.cut_diagonal_preparation(image_array))

#linear_correction_image_array = ep.linear_correction(image_array)
#linear_correction_image = Image.fromarray(linear_correction_image_array)
#linear_correction_image.save('linear_correction.jpg')

#gamma_correction_image_array = ep.gamma_correction(image_array)
#gamma_correction_image = Image.fromarray(gamma_correction_image_array)
#gamma_correction_image.save('gamma_correction.jpg')

#log_correction_image_array = ep.log_correction(image_array)
#log_correction_image = Image.fromarray(log_correction_image_array)
#log_correction_image.save('log_correction.jpg')

#cut_window_preparation_image_array = ep.cut_window_preparation(image_array)
#cut_window_preparation_image = Image.fromarray(cut_window_preparation_image_array)
#cut_window_preparation_image.save('cut_window_preparation.jpg')

#cut_diagonal_preparation_image_array = ep.cut_diagonal_preparation(image_array)
#cut_diagonal_preparation_image = Image.fromarray(cut_diagonal_preparation_image_array)
#cut_diagonal_preparation_image.save('cut_diagonal_preparation.jpg')

#negative_image_array = ep.negative(image_array)
#negative_image = Image.fromarray(negative_image_array)
#negative_image.save('negative.jpg')

#grayscale_image_array = ep.grayscale(image_array)
#grayscale_image = Image.fromarray(grayscale_image_array)
#grayscale_image.save('grayscale.jpg')

#solarize_image_array = ep.solarize(image_array)
#solarize_image = Image.fromarray(solarize_image_array)
#solarize_image.save('solarize.jpg')

#low_pass_filter_image_array = fp.low_pass_filter(image_array)
#low_pass_filter_image = Image.fromarray(low_pass_filter_image_array)
#low_pass_filter_image.save('low_pass_filter.jpg')

#high_pass_filter_image_array = fp.high_pass_filter(image_array)
#high_pass_filter_image = Image.fromarray(high_pass_filter_image_array)
#high_pass_filter_image.save('high_pass_filter.jpg')

#roberts_operator_image_array = fp.roberts_operator(image_array)
#roberts_operator_image = Image.fromarray(roberts_operator_image_array)
#roberts_operator_image.save('roberts_operator.jpg')

#sobel_operator_image_array = fp.sobel_operator(image_array)
#sobel_operator_image = Image.fromarray(sobel_operator_image_array)
#sobel_operator_image.save('sobel_operator.jpg')

#prewitt_operator_image_array = fp.prewitt_operator(image_array)
#prewitt_operator_image = Image.fromarray(prewitt_operator_image_array)
#prewitt_operator_image.save('prewitt_operator.jpg')

#laplacian_operator_image_array = fp.laplacian_operator(image_array)
#laplacian_operator_image = Image.fromarray(laplacian_operator_image_array)
#laplacian_operator_image.save('laplacian_operator.jpg')

opencv_canny_operator_image_array = cv2.Canny(image_array, 150, 50)
opencv_canny_operator_image = Image.fromarray(opencv_canny_operator_image_array)
opencv_canny_operator_image.save('opencv_canny_operator.jpg')

canny_operator_image_array = fp.canny_operator(image_array, 150, 50)
canny_operator_image = Image.fromarray(canny_operator_image_array)
canny_operator_image.save('canny_operator.jpg')

#harmonic_filter_image_array = fp.harmonic_filter(image_array)
#harmonic_filter_image = Image.fromarray(harmonic_filter_image_array)
#harmonic_filter_image.save('harmonic_filter.jpg')

#max_filter_image_array = fp.max_filter(image_array)
#max_filter_image = Image.fromarray(max_filter_image_array)
#max_filter_image.save('max_filter.jpg')

#min_filter_image_array = fp.min_filter(image_array)
#min_filter_image = Image.fromarray(min_filter_image_array)
#min_filter_image.save('min_filter.jpg')

#min_max_filter_image_array = fp.min_max_filter(image_array)
#min_max_filter_image = Image.fromarray(min_max_filter_image_array)
#min_max_filter_image.save('min_max_filter.jpg')

#median_filter_image_array = fp.median_filter(image_array)
#median_filter_image = Image.fromarray(median_filter_image_array)
#median_filter_image.save('median_filter.jpg')

#emboss_filter_image_array = fp.emboss_filter(image_array)
#emboss_filter_image = Image.fromarray(emboss_filter_image_array)
#emboss_filter_image.save('emboss_filter.jpg')

#halftone_filter_image_array = fp.halftone_filter(image_array)
#halftone_filter_image = Image.fromarray(halftone_filter_image_array)
#halftone_filter_image.save('halftone_filter.jpg')