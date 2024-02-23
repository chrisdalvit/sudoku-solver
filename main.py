import cv2 as cv
import numpy as np

from processing import preprocessing

IMAGES_PATH = "./images/"

img = cv.imread(IMAGES_PATH + "sudoku_test.jpg")
img = preprocessing(img)
cv.imshow("Image", img)

cv.waitKey(0)
cv.destroyAllWindows()