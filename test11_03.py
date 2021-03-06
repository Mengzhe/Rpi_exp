import cv2
import numpy as np
#from skimage.measure import block_reduce
#from matplotlib import pyplot as plt

# scipy.signal.find_peaks is the ideal solution to find peaks, but it relies on Scipy 1.1.0
#from scipy.signal import find_peaks

## peakutils is an alternative to find peaks in 1d array
#import peakutils

## detect_peaks is another function to find peaks in 1d array; only depends on Numpy; .py file downloaded
from detect_peaks import detect_peaks

# crop an image
def crop_image(img, lower_bound, upper_bound):
	# img_original_size = img.shape
	# img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:int(img.shape[1]*0.8)]
	# img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),int(img.shape[1]*0.5):]
	img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
	return img_cropped

# use color filter to show lanes in the image
def lane_filter(img, lower_lane_color, upper_lane_color):
	laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
	return laneIMG
		
# find the centers for two lines
def find_lane_centers(laneIMG_binary):
	# find peaks as the starting points of the lanes (left and right)
	vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis = 0)
#    peaks, _ = find_peaks(vector_sum_of_lane_marks, distance=peaks_distance) 
#    peaks = peakutils.indexes(vector_sum_of_lane_marks, min_dist=peaks_distance)
	peaks = detect_peaks(vector_sum_of_lane_marks, mpd=peaks_distance)

	if (peaks.shape[0] < 2):
		# print('only one line')
		lane_center_left, lane_center_right = False, False
	else:
		# we only use the first two peaks as the starting points of the lanes
		peaks = peaks[:2] 
		lane_center_left = peaks[0]
		lane_center_right = peaks[1]
	return lane_center_left, lane_center_right

# to find pixels/indices of one of the left and the right lane
# need to call twice, one for left line, and the other for right lane
def find_pixels_of_lane(laneIMG_binary, lane_center, window_size, width_of_laneIMG_binary):
	indices_nonzero = np.nonzero(laneIMG_binary[:,np.max([0, lane_center-window_size]):np.min([width_of_laneIMG_binary, lane_center+window_size])])
	x = indices_nonzero[0]
	y = indices_nonzero[1] + np.max([0,lane_center-window_size]) # shifted because we are using a part of laneIMG to find non-zero elements 
	return x, y

# return the value of dy/dx at point x
def first_order_derivative_of_second_poly(w, x):
	result = 2*w[0]*x+w[1]
	return result

# return the value of d2y/dx2 at point x
def second_order_derivative_of_second_poly(w, x):
	result = 2*w[0]
	return result

# compute intercept of the line goes through (x,y) with the given slope 
def compute_intercept(slope, x, y):
	return y-slope*x

# this is the tangent line function at point x
def slope_function(slope, intercept, x):
	result = slope*x + intercept
	return result

# compute two points on the tangent line that tangent to the mid_fitted line at x_curv
def compute_points_on_tangent_line(slope, intercept, x_curv, length_tangent_line):
	x_curv_1 = x_curv - int(length_tangent_line/2)
	y_curv_1 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_1))
	x_curv_2 = x_curv + int(length_tangent_line/2)
	y_curv_2 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_2))
	return x_curv_1, y_curv_1, x_curv_2, y_curv_2


	

#For the color filter
lane_color = np.uint8([[[0,0,0]]])
lower_lane_color1 = lane_color
upper_lane_color1 = lane_color+40

# the distances of peaks should be tuned 
# the peaks indicate the possible center of left and right lanes
peaks_distance = 50

# size after downsampling
size_after_downsampling = (192, 32)

# this is the width of laneIMG
width_of_laneIMG_binary = size_after_downsampling[0]
height_of_laneIMG_binary = size_after_downsampling[1]

# use a window to find all pixels of the left lane and the right lane
# here, the window size is half as the distance between peaks
window_size = int(peaks_distance/2) 

# x_axis for polynomial fitting
number_points_for_poly_fit = 50
x_fitted = np.linspace(0, height_of_laneIMG_binary, number_points_for_poly_fit)
# polynomial fitting for lanes 
poly_order = 2

# this number determines how long the tangent line is (in pixel) 
length_tangent_line = 20


img=cv2.imread('09.jpg')
img = np.array(img)
# plt.figure(1)
# plt.imshow(img)

# cropping image
# img_original_size = img.shape
img_cropped = crop_image(img, 0.35, 0.6)
# plt.figure(2)
# plt.imshow(img_cropped)

# downsampling cropped image
img_downsampled = cv2.resize(img_cropped, size_after_downsampling, interpolation=cv2.INTER_LINEAR)
# plt.figure(3)
# plt.imshow(img_downsampled)

# color filtering image
# laneIMG = cv2.inRange(img_downsampled, lower_lane_color1, upper_lane_color1)
laneIMG = lane_filter(img_downsampled, lower_lane_color1, upper_lane_color1)
# making image to a binary representation
laneIMG_binary = laneIMG/255

# # find peaks as the starting points of the lanes (left and right)
# vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis = 0)
# peaks, _ = find_peaks(vector_sum_of_lane_marks, distance=peaks_distance) 
# # we only use the first two peaks as the starting points of the lanes
# peaks = peaks[:2] 
# lane_center_left = peaks[0]
# lane_center_right = peaks[1]
lane_center_left, lane_center_right = find_lane_centers(laneIMG_binary)

if (lane_center_left == False and lane_center_right== False): 
	print('only one line')
else:

	# use a window to find all pixels of the left lane and the right lane
	# polynmial fitting for left lane 
	# y = np.int(w_mid[0]*x**2 + w_mid[1]*x + w_mid[2])

	# indices_left_lane = np.nonzero(laneIMG_binary[:,np.max([0,lane_center_left-window_size]):np.min([width_of_laneIMG_binary, lane_center_left+window_size])])
	# x_left = indices_left_lane[0]
	# y_left = indices_left_lane[1] + np.max([0,lane_center_left-window_size]) # shifted because we are using a part of laneIMG to find non-zero elements 
	x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, width_of_laneIMG_binary)
	w_left = np.polyfit(x_left, y_left, poly_order)
	poly_fit_left = np.poly1d(w_left) # It is convenient to use poly1d objects for dealing with polynomials

	# x_left_fitted = np.linspace(0, laneIMG_binary.shape[0])
	# print('laneIMG_binary.shape[0]', laneIMG_binary.shape[0])
	y_left_fitted = poly_fit_left(x_fitted) 

	# polynmial fitting for right lane
	# indices_right_lane = np.nonzero(laneIMG_binary[:,np.max([0, lane_center_right-window_size]):np.min([width_of_laneIMG_binary, lane_center_right+window_size])])
	# x_right = indices_right_lane[0]
	# y_right = indices_right_lane[1] + np.max([0, lane_center_right-window_size]) # shifted because we are using a part of laneIMG to find non-zero elements 
	x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, width_of_laneIMG_binary)
	w_right = np.polyfit(x_right, y_right, poly_order)
	poly_fit_right = np.poly1d(w_right) # It is convenient to use poly1d objects for dealing with polynomials

	# lane_right_fitted = np.zeros(laneIMG_binary.shape)
	# x_right_fitted = np.linspace(0, laneIMG_binary.shape[0])
	y_right_fitted = poly_fit_right(x_fitted)

	print('w_left', w_left)
	print('w_right', w_right)
	
	# plot the lane centerline
	w_mid = (w_left+w_right)/2
	poly_fit_mid = np.poly1d(w_mid)
	y_mid_fitted = poly_fit_mid(x_fitted)

	# plot the bottom point of the lane centerline
	x_bottom = np.int(x_fitted[-1])
	y_bottom = np.int(y_mid_fitted[-1])
	# y_bottom_test = np.int(w_mid[0]*x_bottom**2 + w_mid[1]*x_bottom + w_mid[2])

	# points to show on image, including two lines for the lane, one lane centerline
	pts_left = np.array([y_left_fitted, x_fitted], np.int32).transpose()
	pts_right = np.array([y_right_fitted, x_fitted], np.int32).transpose()
	pts_mid = np.array([y_mid_fitted, x_fitted], np.int32).transpose()
	pts_mid_bottom = np.array([y_bottom, x_bottom], np.int32).transpose()
	image_center = np.int(width_of_laneIMG_binary/2)

	# compute the pixel distance between the image center and 
	# the bottom point of the lane centerline
	# the car on the right: negative;
	# the car on the left: positive;
	distance_to_center = y_bottom - image_center

	# compute curvature at some point x
	# now, point x is in the middle (from height) of the lane centerline 
	x_curv = np.int(x_fitted[int(number_points_for_poly_fit/2)])
	y_curv = np.int(y_mid_fitted[int(number_points_for_poly_fit/2)])

	first_order_derivative_at_x = first_order_derivative_of_second_poly(w_mid, x_curv)
	intercept = compute_intercept(first_order_derivative_at_x, x_curv, y_curv)
	second_order_derivative_at_x = second_order_derivative_of_second_poly(w_mid, x_curv)

	# we can use first_order_derivative_at_x and intercept
	# to plot the tangent line at x_curv
	# x_curv_1 = x_curv - int(length_tangent_line/2)
	# y_curv_1 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_1))
	# x_curv_2 = x_curv + int(length_tangent_line/2)
	# y_curv_2 = int(slope_function(first_order_derivative_at_x, intercept, x_curv_2))
	x_curv_1, y_curv_1, x_curv_2, y_curv_2 = compute_points_on_tangent_line(first_order_derivative_at_x, \
											 intercept, x_curv, length_tangent_line)

	# here, we officially compute curvature at some point x 
	curvature_at_x =  np.abs(second_order_derivative_at_x) / (1+first_order_derivative_at_x**2)**(3./2)


	# the center of the image indicates the position of the car
	vehicle_pos = (image_center, height_of_laneIMG_binary-2)
	cv2.circle(img_downsampled, center=vehicle_pos, radius=2, color=(255,0,255))

	# lane_center_bottom is the location where the fitted mid line crosses the lower boundary of the image
	# we use (y_bottom, x_bottom) to indicate it
	lane_center_bottom = (y_bottom, x_bottom-2)
	cv2.circle(img_downsampled, center=lane_center_bottom, radius=2, color=(255,0,0))

	# draw an arrowedline from vehicle_pos to lane_center_bottom
	# to indicate the relative position
	cv2.arrowedLine(img_downsampled, (image_center, height_of_laneIMG_binary-3), \
									 (y_bottom, x_bottom-3), (0,0,255), 1)

	# this plots a circle for (y_curv, x_curv)
	cv2.circle(img_downsampled, center=(y_curv, x_curv), radius=2, color=(255,0,0))



	# plot tangent line at (x_curv, y_curv), from (y_curv_1,x_curv_1) to (y_curv_2,x_curv_2)
	cv2.line(img_downsampled, (y_curv_1,x_curv_1), (y_curv_2,x_curv_2),(0,0,255), 1, lineType = 8)

	cv2.polylines(img_downsampled, [pts_left], False, (0,255,255), 1)
	cv2.polylines(img_downsampled, [pts_right], False, (0,255,255), 1)
	cv2.polylines(img_downsampled, [pts_mid], False, (0,255,0), 1)
	img_downsampled_zoomed = cv2.resize(img_downsampled, (0,0), fx=6, fy=6)

	# img_original_zoomed = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
	# cv2.imshow('image_original_zoomed', img_original_zoomed)
	# cv2.imshow('image_cropped', img_cropped)
	# cv2.imshow('image_downsampled', img_downsampled)
	cv2.imshow('image_downsampled_zoomed', img_downsampled_zoomed)
	k = cv2.waitKey(0)
	if k == 27:         # wait for ESC key to exit
		cv2.destroyAllWindows()




