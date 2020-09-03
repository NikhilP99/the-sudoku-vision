from cv2 import cv2

def solve_and_print(image):

    # preprocessing
    gray_sudoku = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred_sudoku = cv2.GaussianBlur(gray_sudoku,(5,5),0)
    thresh = cv2.adaptiveThreshold(blurred_sudoku,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    # find sudoku
    contours,_ = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    largest_c = contours[0]
    max_area = cv2.contourArea(largest_c)

    for c in contours:
        if(cv2.contourArea(c)>max_area):
            max_area = cv2.contourArea(c)
            largest_c = c

    cv2.drawContours(image,[largest_c],-1,(0,255,0),2)


    return image