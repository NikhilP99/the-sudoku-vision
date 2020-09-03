from cv2 import cv2

def solve_and_print(image):

    # preprocessing
    gray_sudoku = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # blurred_sudoku = cv2.GaussianBlur(gray_sudoku,(5,5),0)
    thresh = cv2.adaptiveThreshold(gray_sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    blurred = cv2.medianBlur(thresh,3)

    # find sudoku
    contours,_ = cv2.findContours(cv2.bitwise_not(blurred),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

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
    approx = cv2.approxPolyDP(sudoku_contour,0.1*perimeter,True)

    if len(approx) == 4:
        cv2.drawContours(image,[sudoku_contour],-1,(0,255,0),2)

    return image