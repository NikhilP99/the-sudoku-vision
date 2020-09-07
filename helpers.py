from cv2 import cv2
import numpy as np


def get_top_view(image, corners, make_square=True):

    # get bounding box
    rect = cv2.minAreaRect(corners)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # rect = (center, shape, angle)
    # dimensions
    height = int(rect[1][1])
    width = int(rect[1][0])
    final = np.float32([[0,0],[width,0],[0,height],[width,height]])

    # perspective transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(corners, final)
    warped = cv2.warpPerspective(image, transformation_matrix, (width, height))
    side = max(width, height)
    if side < 200:
        return None

    # make it a square
    try:
        warped = cv2.resize(warped, (side,side), interpolation=cv2.INTER_CUBIC)
        warped = cv2.resize(warped, (450,450), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(e)

    return warped


def sort_corners(corners):
    rect = np.zeros((4, 2), dtype = "float32")

    # Find top left (sum of coordinates is the smallest)
    sum = 10000
    index = 0
    for i in range(4):
        if(corners[i][0]+corners[i][1] < sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find bottom right (sum of coordinates is the biggest)
    sum = 0
    for i in range(3):
        if(corners[i][0]+corners[i][1] > sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[3] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find top right (Only 2 points left, should be easy
    if(corners[0][0] > corners[1][0]):
        rect[1] = corners[0]
        rect[2] = corners[1]
        
    else:
        rect[1] = corners[1]
        rect[2] = corners[0]

    return rect