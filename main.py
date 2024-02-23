import cv2 as cv

from processing import preprocessing, find_sudoku_square

IMAGES_PATH = "./images/"

img = cv.cvtColor(cv.imread(IMAGES_PATH + "sudoku_photo.jpg"), cv.COLOR_BGR2RGB)
prep_img = preprocessing(img)
square = find_sudoku_square(prep_img)
cv.imshow("Image", square)

cv.waitKey(0)
cv.destroyAllWindows()