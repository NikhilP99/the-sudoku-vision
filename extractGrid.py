from cv2 import cv2
import numpy as np
import math
import pytesseract as pt
from helpers import sort_corners, get_top_view
from sudokuSolver import solve


def sudoku_main(image):

    ## find sudoku contour
    sudoku_contour = find_sudoku(image)
    if sudoku_contour is None:
        return image

    ## get corners of the contour
    corners = get_corners(sudoku_contour)
    if corners is None:
        return image
    corners = sort_corners(corners)
    
    ## get top view of the board in square shape
    top_view, transformation_matrix, original_shape = get_top_view(image, corners)
    if top_view is None:
        return image

    ## OCR
    grid = read_grid(top_view)
    if grid is None:
        return image
    print(grid)

    # test sudoku
    test = "740030010019068502000004300056370001001800095090020600103407200500200008080001470"
    if grid == test:
        print("true")

    # solvong the sudoku
    solved = solve(test)
    
    # write the solution over the top view
    empty_boxes = [[0 for j in range(9)] for i in range(9)]
    k = 0
    for i in range(9):
        for j in range(9):
            if grid[k] == '0':
                empty_boxes[i][j] = 1
            k = k + 1
    written = write_solution(top_view,empty_boxes,solved)

    # covert the top view to original size
    resized = cv2.resize(top_view, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)
    # reverse perspective transform
    warped = cv2.warpPerspective(resized, transformation_matrix,(image.shape[1],image.shape[0]),flags=cv2.WARP_INVERSE_MAP)
    # overlay on the original image
    result = np.where(warped.sum(axis=-1,keepdims=True)!=0, warped, image)
        
    return result

''' Writes the solution of sudoku over the given square image '''
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
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y),font, font_scale, (0,255,0), thickness=3, lineType=cv2.LINE_AA)                                                
    
    return image

''' Reads each character from the grid and return a string representing the sudoku ''' 
def read_grid(image):

    sudoku = ""
    side = image.shape[0] // 9  # this is a square image - side will be 50
    offset = 0.1*side           # for borders

    for i in range(9):
        for j in range(9):

            # coordinates if corners of a small box
            top = max(int(round(side*i + offset)), 0)
            left = max(int(round(side*j + offset)), 0)
            right = min(int(round(side*(j+1) - offset)), image.shape[0])
            bottom = min(int(round(side*(i+1) - offset)), image.shape[0])

            # finding the contour inside the box
            crop = image[top:bottom,left:right]
            gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray,(7,7),0)
            thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # finding the largest valid contour
            max_area = 1
            largest_contour = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > side*side*0.04 and area > max_area:
                    max_area = area
                    largest_contour = cnt

            # if there is no contour, then the box must be empty
            if largest_contour is None:
                sudoku += "0"
                continue
            
            # crop out the largest contour i.e. the digit
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            minx = max(min(box, key=lambda g: g[0])[0] - 3, 0)
            miny = max(min(box, key=lambda g: g[1])[1] - 3, 0)
            maxx = min(max(box, key=lambda g: g[0])[0] + 3, int(side))
            maxy = min(max(box, key=lambda g: g[1])[1] + 3, int(side))

            # gray image of only the number
            number_image = gray[miny:maxy, minx:maxx]
            # OCR
            custom_config = r' --psm 6 -c tessedit_char_whitelist=123456789'
            text = pt.image_to_string(number_image, config=custom_config)

            # if i == 7 and j == 3:
            #     cv2.imshow("cropped",number_image)
            #     print(text)

            try:
                num = int(text)
                sudoku += str(num)
            except:
                # if there is a big contour but its not a digit,
                # this frame is invalid, ignore it
                return None
            
    return sudoku


''' Finds the sudoku contour '''
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


''' finds the corners of the contour
    uses convex hull 
'''
def get_corners(contour, max_iter=200):

    corners = None
    accuracy = 1

    while max_iter > 0 and accuracy >= 0:
        max_iter = max_iter - 1
        epsilon = accuracy * cv2.arcLength(contour, True)
        poly_approx = cv2.approxPolyDP(contour, epsilon, True)

        hull = cv2.convexHull(poly_approx)
        # we found 4 corners, return these
        if len(hull) == 4:
            corners = hull
            break
        else:
            # if there are more corners, decrease accuracy
            if len(hull) > 4:
                accuracy += .01
            # else, increase accuracy
            else:
                accuracy -= .01

    if corners is None:
        return None
    
    return corners.reshape(4,2)
