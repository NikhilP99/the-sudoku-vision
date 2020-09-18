# the-sudoku-vision

It is a vision based sudoku solver. Show your sudoku puzzle to the camera and it will overlay the solution over the frame. You can either turn on your webcam or pass an image containing a sudoku. By default, it runs in video mode.

---
## Installation
You will need to install `opencv`, `numpy` and `pytesseract`.

You might have to install `tesseract-ocr` package before installing pytesseract. Use the following command to install this:
```$ sudo apt-get install tesseract-ocr ```

You can use the following commands to install (preferably in a virtual env) these dependencies. Make sure you have python and pip3 installed.

```$ pip3 install opencv-python ```

```$ pip3 install numpy ```

```$ pip3 install pytesseract ```

---
## Running

Video mode: 

``` python3 sudoku-vision.py ```

Image mode: 

``` python3 sudoku-vision.py --file path_to_file ```

---

## The story

### Step 1: Identify a sudoku
We process the image to find a square contour. We are assumming that the sudoku grid is the largest square contour in the frame. If there is no contour found, we wait for the next frame to come in.

### Step 2: Get a solid top view
Now that we found the square, our next target is to get a stable top view of the sudoku grid. We find the corners of the grid by getting it's convex hull. Since it is a (almost) square, the corners in the convex hull will represent the actual corners in the image quite well. After finding the corners, we use persective transform to get the top view of the grid. Also, resize this top view to a square (450 x 450) for better readability.

### Step 3: OCR
Divide the square top view into 9 x 9 small boxes. In each box, we search for a contour with significant size. If there is no contour, the box must be empty.
If we find the contour, it must be a digit so get the bounding box of the contour and crop it out. Here comes tesseract to our rescue. It will read the image and return the digit.
But if there is a contour but that contour is not a digit, there must be some disturbance in then image so we ignore that frame and start again on next frame.

### Step 4: Solving the sudoku
The naive method i.e. backtracking is good enough but we can improve. We can see the sudoku as a **Constraint Satisfaction Problem**. This efficient  algorithm first eliminates the possibilities (of a cell having a certain value) leading to a wrong solution. Then starting from the cell having least number of possible values, iterates over all possible solutions. One might say it iterrates *greedily*. This effectively solves a sudoku in 0.01 seconds on average. For a better explaination, refer [here](http://norvig.com/sudoku.html).

### Step 5: Writing the solution over the frame
The bulk of the work has already been done. Noe that we have the solution of the sudoku, we place the answers over the empty cells in the *top view*. Return the top view to its original shape and reverse the perspective transform. Then we place it in the original frame. Mission completed!

---

## Improvements

When I started this project, I was a noob in Machine Learning and Deep Learning. This was the reason I chose to use a prebuilt library for OCR. But well, it was too slow. Or probably slow only in my low spec laptop (I haven't had an opportunity to test it in another system thanks to this quarantine).
In MY laptop, it(tesseract) takes a good 3-4 seconds to read the whole grid and the video mode appears to be lagging heavily. 
I will be studying and exploring these domains and eventually replace tesseract with a custom trained model.

