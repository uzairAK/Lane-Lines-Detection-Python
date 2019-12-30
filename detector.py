# https://www.youtube.com/watch?v=eLTLtUVuuy4
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    triangle = np.array([[(100, height), (650, height), (400, 250), (340, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.int32(triangle), 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
def displayLines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1),(x2,y2), (0,255,0), 10)
    return line_image
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/4))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])
def average_slope_intercept(image, lines):
    left_fit=[]
    right_fit=[]
    left_line = np.array([])
    right_line = np.array([])
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
#             cv2.line(line_image, (x1,y1),(x2,y2), (255,0,0), 10)
        parameter = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameter[0]
        intercept = parameter[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        print(left_fit_average, 'left')
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        print(right_fit_average, 'right')
        right_line = make_coordinates(image, right_fit_average)
    # left_fit_average = np.average(left_fit, axis=0)
    # right_fit_average = np.average(right_fit, axis=0)
    # print(left_fit)
    # print(right_fit)
    # left_line = make_coordinates(image, left_fit_average)
    # right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# image = cv2.imread("images.jpg")
# lane_image = np.copy(image)
# canny = canny(lane_image)
# cropped_image = region_of_interest(canny)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap = 70)
# averaged_lines = average_slope_intercept(lane_image, lines)
# print(averaged_lines)
# # line_image = displayLines(lane_image, lines)
# line_image = displayLines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#
# cv2.imshow("",combo_image)
# # plt.show()
# print(lines)
# for i in range(len(lines)):
#   for line in lines[i]:
#      cv2.line(image, (line[0],line[1]), (line[2],line[3]), (0,255,0), 2)
# cv2.imwrite("lines.jpg", image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    frame = cv2.resize(frame, (739, 415))
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=70)
    averaged_lines = average_slope_intercept(frame, lines)
    print(averaged_lines)
    # line_image = displayLines(lane_image, lines)
    line_image = displayLines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    # cv2.imshow("", combo_image)
    cv2.imshow("", combo_image)
    cv2.waitKey(1)
