from cv2 import cv2
import numpy as np
import math
import pytesseract as pt
from helpers import sort_corners, get_top_view
from sudokuSolver import solve

old = False
solution = []

def sudoku_main(image):
    global old
    global solution
    sudoku_contour = find_sudoku(image)
    if sudoku_contour is None:
        return image

    corners = get_corners(sudoku_contour)
    if corners is None:
        return image
    corners = sort_corners(corners)
    
    top_view, transformation_matrix, original_shape = get_top_view(image, corners)
    if top_view is None:
        return image

    # grid = read_grid(top_view)
    # if grid is not None:
    #     print(grid)
    #     if original == grid:
    #         print("True")
    #     else: 
    #         print("False")

    sudoku = "740030010019068502000004300056370001001800095090020600103407200500200008080001470"
    if old is False:
        solved = solve(sudoku)
        solution = solved
        old = True
    else:
        solved = solution
    
    
    empty_boxes = [[0 for j in range(9)] for i in range(9)]
    k = 0
    for i in range(9):
        for j in range(9):
            if sudoku[k] == '0':
                empty_boxes[i][j] = 1
            k = k + 1

    written = write_solution(top_view,empty_boxes,solved)
    resized = cv2.resize(top_view, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

    warped = cv2.warpPerspective(resized, transformation_matrix,(image.shape[1],image.shape[0]),flags=cv2.WARP_INVERSE_MAP)
    result = np.where(warped.sum(axis=-1,keepdims=True)!=0, warped, image)
        
    return result

def write_solution(image,empty_boxes,solved):
    # Write grid on image
    side = image.shape[0] // 9
    for i in range(9):
        for j in range(9):
            if(empty_boxes[i][j] != 1):
                continue               

            text = str(solved[i][j])
            offset = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
        
            font_scale = 0.4 * side / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = side*j + math.floor((side - text_width) / 2) + offset
            bottom_left_corner_y = side*(i+1) - math.floor((side - text_height) / 2) + offset
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), 
                                                  font, font_scale, (0,255,0), thickness=3, lineType=cv2.LINE_AA)
    
    return image

def read_grid(image):

    sudoku = [[0 for j in range(9)] for i in range(9)]
    side = image.shape[0] // 9 # this is a square image - side will be 50
    offset = 0.1*side

    for i in range(9):
        for j in range(9):

            top = max(int(round(side*i + offset)), 0)
            left = max(int(round(side*j + offset)), 0)
            right = min(int(round(side*(j+1) - offset)), image.shape[0])
            bottom = min(int(round(side*(i+1) - offset)), image.shape[0])

            crop = image[top:bottom,left:right]
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray,(7,7),0)
            thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 1
            largest_contour = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > side*side*0.04 and area > max_area:
                    max_area = area
                    largest_contour = cnt

            if largest_contour is None:
                sudoku[i][j] = 0
                continue
    
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            minx = max(min(box, key=lambda g: g[0])[0] - 3, 0)
            miny = max(min(box, key=lambda g: g[1])[1] - 3, 0)
            maxx = min(max(box, key=lambda g: g[0])[0] + 3, int(side))
            maxy = min(max(box, key=lambda g: g[1])[1] + 3, int(side))

            number_image = gray[miny:maxy, minx:maxx]
            custom_config = r' --psm 6 -c tessedit_char_whitelist=123456789'
            text = pt.image_to_string(number_image, config=custom_config)

            if i == 7 and j == 3:
                cv2.imshow("cropped",number_image)
                print(text)

            try:
                num = int(text)
                sudoku[i][j] = num
            except:
                return None
            
    return sudoku



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
