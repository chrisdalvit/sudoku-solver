import cv2 as cv
import numpy as np

def preprocessing(img):
    img = img.copy()
    heigth, width, channels = img.shape
    aspect = width / heigth
    img = cv.resize(img, (1000, int(1000*aspect)))
    img = cv.GaussianBlur(img, (5,5), 2)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5), 2)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return img

def _count_children(contour_index, hierarchy):
    hierarchy = hierarchy[0]
    return len(list(filter(lambda x: x[3] == contour_index, hierarchy)))

def _sort_corners(corners):
    y_sorted = sorted(corners, key=lambda x: x[0][1])
    upper_pts, lower_pts = y_sorted[:2], y_sorted[2:]
    upper_sorted = sorted(upper_pts, key=lambda x: x[0][0])
    lower_sorted = sorted(lower_pts, key=lambda x: x[0][0])
    return np.array([upper_sorted[0], upper_sorted[1], lower_sorted[0], lower_sorted[1]]).astype(np.float32)

def _extract_sudoku_square(img, corners):
    img = img.copy()
    dst_size = 1000
    src_pts = _sort_corners(corners)
    dst_pts = np.array([[0,0], [1,0], [0,1], [1,1]]).astype(np.float32) * dst_size
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    return cv.warpPerspective(img, M, (dst_size, dst_size))

def find_sudoku_square(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours):
        perimeter = cv.arcLength(c, closed=True)
        approx = cv.approxPolyDP(c, epsilon=0.1*perimeter, closed=True)
        if len(approx) == 4:
            num_children = _count_children(i, hierarchy)
            if num_children == 9 or num_children == 81:
                return _extract_sudoku_square(img, approx)
    
    
        