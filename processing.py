from typing import List, Tuple

import cv2 as cv
import numpy as np

def preprocessing(img: cv.typing.MatLike) -> cv.typing.MatLike:
    """Preprocess image for sudoku solving.
    
    Resize the image to 1000x1000 pixels, convert the image to grayscale, apply Gaussian blur and apply adaptive thresholding.

    Args:
        img (cv.typing.MatLike): OpenCV MatLike image to process

    Returns:
        cv.typing.MatLike: Processed image
    """
    img = img.copy()
    heigth, width, _ = img.shape
    aspect = width / heigth
    img = cv.resize(img, (1000, int(1000*aspect)))
    img = cv.GaussianBlur(img, (5,5), 2)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return img

def _count_children(contour_index: int, hierarchy: List[List[int]]) -> int:
    """Count how many children the contour with 'contour_index' has.

    Args:
        contour_index (int): Index of the parent countour
        hierarchy (_type_): Hierarchy of contours returned by OpenCV's findContours method

    Returns:
        int: Number of children
    """
    top_hierarchy = hierarchy[0]
    return len(list(filter(lambda x: x[3] == contour_index, top_hierarchy)))

def _sort_corners(corners: List[List[int]]) -> List[List[int]]:
    """Sort the corners of a rectangle in clockwise order.

    Args:
        corners (List[List[int]]): Corners to sort

    Returns:
        List[List[int]]: Numpy array of sorted corners
    """
    # Since we assume 4 corner points in arbitrary order we first sort the points by y values
    # By y value we can devide the points in upper and lower points. By sorting and dividing by x values we get right/left points
    # We then can return the sorted array in clockwise order
    y_sorted = sorted(corners, key=lambda x: x[0][1])
    upper_pts, lower_pts = y_sorted[:2], y_sorted[2:]
    upper_sorted = sorted(upper_pts, key=lambda x: x[0][0])
    lower_sorted = sorted(lower_pts, key=lambda x: x[0][0])
    return np.array([upper_sorted[0], upper_sorted[1], lower_sorted[0], lower_sorted[1]]).astype(np.float32)

def _extract_sudoku_square(img: cv.typing.MatLike, corners: List[List[int]]) -> Tuple[cv.typing.MatLike, cv.typing.MatLike]:
    """Extract the Sudoku square from an image given four corner points.

    Args:
        img (cv.typing.MatLike): Image from where the Sudoku square should be extracted
        corners (List[List[int]]): Corners of the Sudoku square

    Returns:
        Tuple[cv.typing.MatLike, cv.typing.MatLike]: The extracted image patch and the matrix applied for the perspective transformation
    """
    img = img.copy()
    dst_size = 1000
    src_pts = _sort_corners(corners)
    dst_pts = np.array([[0,0], [1,0], [0,1], [1,1]]).astype(np.float32) * dst_size
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    return cv.warpPerspective(img, M, (dst_size, dst_size)), M

def find_sudoku_square(img: cv.typing.MatLike) -> Tuple[cv.typing.MatLike, cv.typing.MatLike]:
    """Find and return the Sudoku square in an given image.

    Args:
        img (cv.typing.MatLike): Image with Sudoku square

    Returns:
        Tuple[cv.typing.MatLike, cv.typing.MatLike]: The extracted image patch and the matrix applied for the perspective transformation 
    """
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for i, c in enumerate(contours):
        perimeter = cv.arcLength(c, closed=True)
        approx = cv.approxPolyDP(c, epsilon=0.1*perimeter, closed=True)
        if len(approx) == 4:
            num_children = _count_children(i, hierarchy)
            if num_children == 9 or num_children == 81:
                return _extract_sudoku_square(img, approx)
    
    
        