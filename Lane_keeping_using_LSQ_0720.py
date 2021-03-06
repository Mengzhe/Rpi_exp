'''
Only lane tracking with no car following
'''
import io
import time
import picamera
from PIL import Image
import numpy as np
import cv2
import RPi.GPIO as GPIO
##from processImage import processImage
##from serverClass import Server
import sys

MARGIN_LEFT = 260
MARGIN_RIGHT = 380
turningKL = 4.2 #f5or lane tracking, turning left
turningKR = 1 #for lane tracking, turning right


HOST = '0.0.0.0'
PORT = 5001
'''---------------GPIO SECTION ----------'''
GPIO.setmode(GPIO.BCM)


ENA = 26
ENB = 11
IN1 = 19
IN2 = 13
IN3 = 6
IN4 = 5

time.sleep(1) 

#  Motor Pins
GPIO.setup(ENA, GPIO.OUT) #ENA
GPIO.setup(ENB, GPIO.OUT) #ENB
GPIO.setup(IN1, GPIO.OUT) #IN1
GPIO.setup(IN2, GPIO.OUT) #IN2
GPIO.setup(IN3, GPIO.OUT)  #IN3
GPIO.setup(IN4, GPIO.OUT)  #IN4

# PWM pin and Frequency
pwmR = GPIO.PWM(26, 100)
pwmL = GPIO.PWM(11, 100)

pwmR.start(0)
pwmL.start(0)

time.sleep(1)

'''--------------------------------------'''

# Set motor forwards
GPIO.output(19, GPIO.HIGH)
GPIO.output(13, GPIO.LOW)
GPIO.output(6, GPIO.HIGH)
GPIO.output(5, GPIO.LOW)
time.sleep(1)
print ('GPIO INITIALIZED')

'''---functions for image processing---'''
## detect_peaks is another function to find peaks in 1d array; only depends on Numpy; .py file downloaded
from detect_peaks import detect_peaks

# crop an image
def crop_image(img, lower_bound, upper_bound):
    # img_original_size = img.shape
    img_cropped = img[int(img.shape[0]*lower_bound):int(img.shape[0]*upper_bound),:]
    return img_cropped

# use color filter to show lanes in the image
def lane_filter(img, lower_lane_color, upper_lane_color):
    laneIMG = cv2.inRange(img, lower_lane_color, upper_lane_color)
    return laneIMG
        
# find the centers for two lines
def find_lane_centers(laneIMG_binary, left_peak_previous, right_peak_previous):
        # find peaks as the starting points of the lanes (left and right)
        vector_sum_of_lane_marks = np.sum(laneIMG_binary, axis = 0)
#    peaks, _ = find_peaks(vector_sum_of_lane_marks, distance=peaks_distance) 
#    peaks = peakutils.indexes(vector_sum_of_lane_marks, min_dist=peaks_distance)
        peaks = detect_peaks(vector_sum_of_lane_marks, mpd=peaks_distance)

        if (peaks.shape[0] == 1):
                # print('only one line')
##                lane_center_left, lane_center_right = False, False
                current_peak = peaks[0]
                # if the current peak is closer to previous left line center, we say right line is missing
                if (np.abs(current_peak-left_peak_previous) <= np.abs(current_peak-right_peak_previous)):
                    lane_center_right = False
                    lane_center_left = current_peak
##                    print('left line remains, right line is missing')
                else:
                    lane_center_left = False
                    lane_center_right = current_peak
##                    print('right line remains, left line is missing')
        # no peak is detected
        elif (peaks.shape[0] == 0):
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

# initialize the lane_peaks_previous
left_peak_previous, right_peak_previous = 0, width_of_laneIMG_binary

# to store the previous curves for left and right lane
w_left_previous, w_right_previous = np.zeros((3)), np.zeros((3))
 

'''--------------------------------------'''


'''---functions for control---'''
# actions
l1, l2, l3  =  70, 35, 0
r1, r2, r3 =  70, 35, 0
action_set = np.array([[l1, r1],
                       [l1, r2],
                       [l1, r3],
                       [l2, r1],
                       [l3, r1]])
dim_action = action_set.shape[0]
go_straight = action_set[0,:]
turn_left = action_set[4,:]
turn_right = action_set[2,:]


#  threshold in pixels
distance_threshold = 4

# naive policy to achive lane keeping, using simple rules
def choose_action_using_simple_logic(distance_to_center):
    if np.abs(distance_to_center) <= distance_threshold:
##        action = go_straight
        chosen_action_number = 0
    else:
        if distance_to_center < 0:
##            action = turn_left
            chosen_action_number = 4
        else:
##            action = turn_right
            chosen_action_number = 2

    return chosen_action_number

# epsilon greedy: epsilon% to choose random action
epsilon_greedy = 0
# suppose we have two states: distance_to_center, curvature
dim_state = 2

# memory for storing states and actions
memory_size = 1000
memory_counter = 0
memory = np.zeros((memory_size,dim_state+1))

# this function is the kernel function of state s
def feature_pre(s):
    s = s[np.newaxis,:]
    feature_sub = np.hstack((np.eye(1), s, s**2,[s[:,0]*s[:,1]])).transpose()
    return feature_sub


# load learned weights
w_02 = np.load('learned_weights_w_02_0719.npy')
w_matrix = w_02.reshape((5,6))
def choose_action_from_policy_w(x):
    q_values = np.matmul(w_matrix, x)
    chosen_action_number = np.argmax(q_values)    
    return chosen_action_number

'''--------------------------------------'''

distance = []
data = []
stream = io.BytesIO()
distance_to_center_last = -15
firstStart = True
try:
    with picamera.PiCamera(resolution='VGA') as camera:
            with io.BytesIO() as stream:
                    for frame in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                        #aa = time.time()
                        stream.seek(0)
                        image = Image.open(stream)
                        img = np.array(image)
                        key = cv2.waitKey(1) & 0xFF #necessary for cv2
                        
                        if (firstStart == True):
                            pwmR.ChangeDutyCycle(60)
                            pwmL.ChangeDutyCycle(50)
                            firstStart = False

                        # cropping image
                        img_cropped = crop_image(img, 0.35, 0.6)
##                        img_cropped = crop_image(img, 0.7, 1)
                        # downsampling cropped image
                        img_downsampled = cv2.resize(img_cropped, size_after_downsampling, interpolation=cv2.INTER_LINEAR)

                        # color filtering image
                        laneIMG = lane_filter(img_downsampled, lower_lane_color1, upper_lane_color1)
                        # making image to a binary representation
                        laneIMG_binary = laneIMG/255

                        ## find peaks as the starting points of the lanes (left and right)
                        lane_center_left, lane_center_right = find_lane_centers(laneIMG_binary, left_peak_previous, right_peak_previous)
                        
                        if (lane_center_left == False and lane_center_right== False):
                            print('No lines')
                            pwmR.stop()
                            pwmL.stop()  
                            GPIO.cleanup()
                            break
                        elif (lane_center_left == False):
                            print('left line is missing')
                            w_left = w_left_previous
                            # polynmial fitting for right lane
                            x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, width_of_laneIMG_binary)
                            try:
                                w_right = np.polyfit(x_right, y_right, poly_order)
                            except ValueError:
                                print('ValueError! Stop!')
                                w_right = w_right_previous
                                pwmR.stop()
                                pwmL.stop()  
                                GPIO.cleanup()
                                break
                        elif (lane_center_right== False):
                            print('right line is missing')
                            w_right = w_right_previous
                            # polynmial fitting for left lane
                            x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, width_of_laneIMG_binary)
                            try:
                                w_left = np.polyfit(x_left, y_left, poly_order)
                            except ValueError:
                                print('ValueError! Stop!')
                                w_left = w_left_previous
                                pwmR.stop()
                                pwmL.stop()  
                                GPIO.cleanup()
                                break                           
##                            pwmR.stop()
##                            pwmL.stop()  
##                            GPIO.cleanup()
##                            break                                    
                        else:
                            left_peak_previous, right_peak_previous = lane_center_left, lane_center_right

                            # use a window to find all pixels of the left
                            # lane and the right lane

                            # polynmial fitting for left lane
                            x_left, y_left = find_pixels_of_lane(laneIMG_binary, lane_center_left, window_size, width_of_laneIMG_binary)
                            try:
                                w_left = np.polyfit(x_left, y_left, poly_order)
                            except ValueError:
                                w_left = w_left_previous
                                print('ValueError! Stop!')
                                w_left = w_left_previous
                                pwmR.stop()
                                pwmL.stop()  
                                GPIO.cleanup()
                                break

                            # polynmial fitting for right lane
                            x_right, y_right = find_pixels_of_lane(laneIMG_binary, lane_center_right, window_size, width_of_laneIMG_binary)
                            try:
                                w_right = np.polyfit(x_right, y_right, poly_order)
                            except ValueError:
                                print('ValueError! Stop!')
                                w_right = w_right_previous
                                pwmR.stop()
                                pwmL.stop()  
                                GPIO.cleanup()
                                break
                                

                            
                        # It is convenient to use poly1d objects
                        # for dealing with polynomials
                        poly_fit_left = np.poly1d(w_left)
                        y_left_fitted = poly_fit_left(x_fitted)                             
                        poly_fit_right = np.poly1d(w_right)
                        y_right_fitted = poly_fit_right(x_fitted)

                        # plot the lane centerline
                        w_mid = (w_left+w_right)/2
                        poly_fit_mid = np.poly1d(w_mid)
                        y_mid_fitted = poly_fit_mid(x_fitted)

                        # to store the previous curves for left and right lane
                        w_left_previous, w_right_previous = w_left, w_right

                        # plot the bottom point of the lane centerline
                        x_bottom = np.int(x_fitted[-1])
                        y_bottom = np.int(y_mid_fitted[-1])


                        
                        # plot for fitting
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
##                        if (np.abs(distance_to_center-distance_to_center_last)>=30 or (np.abs(distance_to_center)>=100)):
                        if (np.abs(distance_to_center-distance_to_center_last)>=30):

##                            print('distance_to_center_final',distance_to_center)
                            print('large error in distance')
                            distance_to_center = distance_to_center_last
##                            pwmR.stop()
##                            pwmL.stop()  
##                            GPIO.cleanup()
##                            break
                        else:
                            distance_to_center_last = distance_to_center

                        if (np.abs(distance_to_center)>=100):
                            pwmR.stop()
                            pwmL.stop()  
                            GPIO.cleanup()
                            break

                        
                        
                        # compute curvature at some point x
                        # now, point x is in the middle (from height) of the lane centerline 
                        x_curv = np.int(x_fitted[int(number_points_for_poly_fit/2)])
                        y_curv = np.int(y_mid_fitted[int(number_points_for_poly_fit/2)])

                        
                        cv2.polylines(img_downsampled, [pts_left], False, (0,255,255), 1)
                        cv2.polylines(img_downsampled, [pts_right], False, (0,255,255), 1)
                        cv2.polylines(img_downsampled, [pts_mid], False, (0,255,0), 1)


                        first_order_derivative_at_x = first_order_derivative_of_second_poly(w_mid, x_curv)
##                        print('derivative_at_x', first_order_derivative_at_x)                        
                        intercept = compute_intercept(first_order_derivative_at_x, x_curv, y_curv)
                        second_order_derivative_at_x = second_order_derivative_of_second_poly(w_mid, x_curv)
                        # we can use first_order_derivative_at_x and intercept
                        # to plot the tangent line at x_curv
                        x_curv_1, y_curv_1, x_curv_2, y_curv_2 = compute_points_on_tangent_line(first_order_derivative_at_x,intercept, x_curv, length_tangent_line)


                        # here, we officially compute curvature at some point x
                        # in order to differentiate between left turn and right turn
                        # here we use the signed curvature
                        # but, here the signed is opposite to the original definition: https://en.wikipedia.org/wiki/Curvature#Signed_curvature
                        # will fix in the future
##                        curvature_at_x =  np.abs(second_order_derivative_at_x) / (1+first_order_derivative_at_x**2)**(3./2)
                        curvature_at_x =  -second_order_derivative_at_x / (1+first_order_derivative_at_x**2)**(3./2)

                        
                        
##                        print('curvature_at_x', curvature_at_x)

                        # the center of the image indicates the position of the car
                        vehicle_pos = (image_center, height_of_laneIMG_binary-2)
                        cv2.circle(img_downsampled, center=vehicle_pos, radius=2, color=(255,0,255))

                        state = np.array([distance_to_center, curvature_at_x])
##                            print('state.shape', state.shape)

                        

                        if np.random.randint(0,10) <= epsilon_greedy*10-1:
                            chosen_action_number = np.random.randint(0,dim_action)
                            
                        else:
##                            chosen_action_number = choose_action_using_simple_logic(distance_to_center)
                            # compute feature using state
                            x = feature_pre(state)
                            chosen_action_number = choose_action_from_policy_w(x)
                            
##                            print('distance_to_center:', distance_to_center, 'curvature_at_x', curvature_at_x, 'chosen_action_number', chosen_action_number)

                        action = action_set[chosen_action_number]
                        pwm_l = action[0]
                        pwm_r = action[1]
                        memory[memory_counter, :] = np.hstack([state, chosen_action_number])                                                
##                            print('memory current:', memory[memory_counter, :])
                        memory_counter += 1
                        
                        
                        pwmL.ChangeDutyCycle(pwm_l)
                        pwmR.ChangeDutyCycle(pwm_r)

                        print('state,', state,'pwm_l', pwm_l, 'pwm_r', pwm_r)

                        
                        # lane_center_bottom is the location where the fitted mid line crosses the lower boundary of the image
                        # we use (y_bottom, x_bottom) to indicate it
                        lane_center_bottom = (y_bottom, x_bottom-2)
                        cv2.circle(img_downsampled, center=lane_center_bottom, radius=2, color=(255,0,0))

                        
                        # this plots a circle for (y_curv, x_curv)
                        cv2.circle(img_downsampled, center=(y_curv, x_curv), radius=2, color=(255,0,0))

                        # plot tangent line at (x_curv, y_curv), from (y_curv_1,x_curv_1) to (y_curv_2,x_curv_2)
                        cv2.line(img_downsampled, (y_curv_1,x_curv_1), (y_curv_2,x_curv_2),(0,0,255), 1, lineType = 8)

                        
                        img_downsampled_zoomed = cv2.resize(img_downsampled, (0,0), fx=4, fy=4)
                        cv2.imshow('img_downsampled_zoomed', img_downsampled_zoomed)
                            
                        stream.seek(0)
                        stream.truncate() # remove all previous images
                        
                                    
except KeyboardInterrupt:
    pwmR.stop()
    pwmL.stop()
    np.save('memory', memory)
    GPIO.cleanup()
    #s.closeSocket()
    
  
