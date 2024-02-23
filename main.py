import cv2 as cv

from processing import preprocessing, find_sudoku_square, extract_cells

IMAGES_PATH = "./images/"

img = cv.cvtColor(cv.imread(IMAGES_PATH + "sudoku_test.jpg"), cv.COLOR_BGR2RGB)
prep_img = preprocessing(img)
square, _ = find_sudoku_square(prep_img)
cells = extract_cells(square)

for c in cells[:9]:
    cv.imshow("Image", c)
    cv.waitKey(0)
    
cv.destroyAllWindows()