from cv2 import cv2
import numpy as np

# get corners of a contour
# uses the convex hull
def get_corners(contours, corner_amount=4, max_iter=200):

    coefficient = 1
    while max_iter > 0 and coefficient >= 0:
        max_iter = max_iter - 1

        epsilon = coefficient * cv2.arcLength(contours, True)

        poly_approx = cv2.approxPolyDP(contours, epsilon, True)
        hull = cv2.convexHull(poly_approx)
        if len(hull) == corner_amount:
            return hull
        else:
            if len(hull) > corner_amount:
                coefficient += .01
            else:
                coefficient -= .01
    return None


def solve_and_print(image):

    # preprocessing
    gray_sudoku = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred_sudoku = cv2.GaussianBlur(gray_sudoku,(7,7),0)
    thresh = cv2.adaptiveThreshold(blurred_sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    # blurred = cv2.medianBlur(thresh,3)

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
        return image

    # Yayy! we found the sudoku grid
    # Now let us find the corners

    corners = get_corners(sudoku_contour)
    if corners is None:
        return image

    corners = corners.reshape(4,2)
    cv2.circle(image, (corners[0][0], corners[0][1]), 5, (255,0,0), thickness=3, lineType=8, shift=0)
    cv2.circle(image, (corners[1][0], corners[1][1]), 5, (255,0,0), thickness=3, lineType=8, shift=0)
    cv2.circle(image, (corners[2][0], corners[2][1]), 5, (255,0,0), thickness=3, lineType=8, shift=0)
    cv2.circle(image, (corners[3][0], corners[3][1]), 5, (255,0,0), thickness=3, lineType=8, shift=0)

    return image