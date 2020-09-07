from cv2 import cv2
import numpy as np
import math
import pytesseract as pt
from helpers import sort_corners, get_top_view

cv2.namedWindow("warp")
cv2.namedWindow("top view")

def sudoku_main(image):

    sudoku_contour = find_sudoku(image)
    if sudoku_contour is None:
        return image

    corners = get_corners(sudoku_contour)
    if corners is None:
        return image
    corners = sort_corners(corners)
    
    top_view = get_top_view(image, corners)
    h = top_view.shape[1]
    w = top_view.shape[0]
    if h < 150 or w < 150:
        return image 
    
    grid = read_grid(top_view)
    cv2.imshow("top view",top_view)

    return image


def read_grid(image):

    sudoku = []
    N = 9
    for i in range(9):
        row = []
        for j in range(9):
            row.append(0)
        sudoku.append(row)
    
    side = image.shape[0] // 9 # this is a square image - side will be 50
    offset = 0.1*side

    i = 1
    j = 1
    top = int(round(side*i + offset))
    left = int(round(side*j + offset))
    right = int(round(side*(j+1) - offset))
    bottom = int(round(side*(i+1) - offset))

    print((top,left,right,bottom))

    crop = image[top:bottom,left:right]
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 1
    largest_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > side*side*0.08 and area > max_area:
            max_area = area
            largest_contour = cnt


    if largest_contour is None:
        print("Nothing here")
        return crop
    
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    

    minx = max(min(box, key=lambda g: g[0])[0], 0)
    miny = max(min(box, key=lambda g: g[1])[1], 0)
    maxx = min(max(box, key=lambda g: g[0])[0], int(side))
    maxy = min(max(box, key=lambda g: g[1])[1], int(side))

    number_image = gray[miny-3:maxy+3, minx-3:maxx+3]
    # thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(thresh,[largest_contour],-1,(255,0,0),1)
    _, t = cv2.threshold(number_image,100,255,cv2.THRESH_BINARY)
    cv2.imshow("warp",crop)

    custom_config = r'--oem 3 --psm 6'
    text = pt.image_to_string(number_image, config=custom_config)
    print(text)
    return crop



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
