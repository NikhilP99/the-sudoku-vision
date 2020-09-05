from cv2 import cv2
import numpy as np
from helpers import sort_corners, get_top_view

cv2.namedWindow("warp")

def sudoku_main(image):

    sudoku_contour = find_sudoku(image)
    if sudoku_contour is None:
        return image

    corners = get_corners(sudoku_contour)
    if corners is None:
        return image
    corners = sort_corners(corners)
    
    top_view = get_top_view(image, corners)
    cv2.imshow("warp",top_view)

    return image

def find_sudoku(image):

    # preprocessing
    gray_sudoku = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred_sudoku = cv2.GaussianBlur(gray_sudoku,(7,7),0)
    thresh = cv2.adaptiveThreshold(blurred_sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    # find contours
    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # get the sudoku contour
    sudoku_area = 0
    sudoku_contour = contours[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (0.7 < float(w) / h < 1.3     # aspect ratio
                and area > 150 * 150     # minimal area
                and area > sudoku_area   # biggest area on screen
                and area > .5 * w * h):  # fills bounding rect
            sudoku_area = area
            sudoku_contour = cnt

    perimeter = cv2.arcLength(sudoku_contour,True)
    # define the accuracy here
    epsilon = 0.05*perimeter
    approx = cv2.approxPolyDP(sudoku_contour,epsilon,True)

    # if it is not a sudoku board, just return the frame without doing anything
    if len(approx) != 4:
        return None
    
    return sudoku_contour

# get corners of a contour
# uses the convex hull
def get_corners(contour, max_iter=200):

    corners = None

    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1
        epsilon = coefficient * cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, epsilon, True)

        hull = cv2.convexHull(poly_approx)
        if len(hull) == 4:
            corners = hull
            break
        else:
            if len(hull) > 4:
                coefficient += .01
            else:
                coefficient -= .01

    if corners is None:
        return None
    
    return corners.reshape(4,2)
