from typing import List, Tuple

import cv2 as cv
import numpy as np

from sudoku import Sudoku

def preprocessing(img: cv.typing.MatLike) -> cv.typing.MatLike:
    """Preprocess image for sudoku solving.
    
    Resize the image to 1000x1000 pixels, convert the image to grayscale, apply Gaussian blur and apply adaptive thresholding.

    Args:
        img (cv.typing.MatLike): OpenCV MatLike image to process

    Returns:
        cv.typing.MatLike: Processed image
    """
    img = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5,5), 2)
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

def _extract_sudoku_square(img: cv.typing.MatLike, corners: List[List[int]]) -> Tuple[cv.typing.MatLike, List[List[int]]]:
    """Extract the Sudoku square from an image given four corner points.

    Args:
        img (cv.typing.MatLike): Image from where the Sudoku square should be extracted
        corners (List[List[int]]): Corners of the Sudoku square

    Returns:
        Tuple[cv.typing.MatLike, List[List[int]]]: The extracted image patch and the list of corner points 
    """
    img = img.copy()
    dst_size = 1000
    src_pts = _sort_corners(corners)
    dst_pts = np.array([[0,0], [1,0], [0,1], [1,1]]).astype(np.float32) * dst_size
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    return cv.warpPerspective(img, M, (dst_size+10, dst_size+10)), src_pts

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
    
def _compute_cell_border_mask(cell: cv.typing.MatLike) -> cv.typing.MatLike:
    """Create mask of cell borders.

    Args:
        cell (cv.typing.MatLike): Cell with borders

    Returns:
        cv.typing.MatLike: Mask where borders have value 0 (or near to zero)
    """
    contours, _ = cv.findContours(cell, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    s = max(contours, key=cv.contourArea)
    mask = np.zeros(cell.shape, dtype="uint8")
    mask = cv.drawContours(mask, [s], -1, 255, -1)
    # Erode to avoid artifacts when subtracting the mask
    return cv.erode(mask, cv.getStructuringElement(cv.MORPH_RECT,(7,7)), iterations=1) 

def _remove_cell_borders(cell: cv.typing.MatLike) -> cv.typing.MatLike:
    """Remove border artifacts on edges of the cell.

    Args:
        cell (cv.typing.MatLike): Cell with border

    Returns:
        cv.typing.MatLike: Cell with removed border
    """
    inverted_cell = 255 - cell
    mask = _compute_cell_border_mask(cell)
    cell = cv.bitwise_and(inverted_cell, mask)
    cell = cv.medianBlur(cell, 5)
    _, cell = cv.threshold(cell,127,255,cv.THRESH_BINARY)
    return cell
    
def extract_cells(img: cv.typing.MatLike) -> List[cv.typing.MatLike]:
    """Extract single cells of Sudoku square.

    Args:
        img (cv.typing.MatLike): Image of Sudoku square

    Returns:
        List[cv.typing.MatLike]: List of cell image patches
    """
    height, width = img.shape
    cell_height, cell_width = height // 9, width // 9
    cells = []
    for i in range(0, height-cell_height, cell_height):
        for j in range(0, width-cell_width, cell_width):
            cell = img[i:i+cell_height, j:j+cell_width]
            cell = _remove_cell_borders(cell)
            cells.append(cv.resize(cell, (28,28)))
    return np.array(cells).astype(np.float32) / 255.
    
def blend_images(foreground, background, mask, color=(0.,0.,1.)):
    foreground = foreground * color
    foreground = cv.multiply(mask, foreground)
    background = cv.multiply((1.0 - mask).astype(np.uint8), background)
    return cv.add((foreground*255).astype(np.uint8), background)

def draw_digits(img, preds, corners, transform_size=1000):
    sudoku = Sudoku(preds)
    sudoku.solve()

    dst_pts = np.array([[0,0], [1,0], [0,1], [1,1]]).astype(np.float32) * 1000
    M = cv.getPerspectiveTransform(corners, dst_pts)
    wraped = np.zeros((transform_size,transform_size, 3)).astype(float)

    cell_height, cell_width = transform_size // 9, transform_size // 9

    s = Sudoku(preds)
    for i, px in enumerate(range(0, transform_size-cell_height, cell_height)):
        for j, py in enumerate(range(0, transform_size-cell_width, cell_width)):
            if s[i,j] is None:
                patch = wraped[px:px+cell_height, py:py+cell_width]
                patch = cv.putText(patch, str(sudoku[i,j]), (int(cell_width*0.2),int(cell_height*0.85)), cv.FONT_HERSHEY_COMPLEX, 3, (1.,1.,1.), 3, cv.LINE_AA)
        
    return cv.warpPerspective(wraped, np.linalg.inv(M), (img.shape[1], img.shape[0]))

        