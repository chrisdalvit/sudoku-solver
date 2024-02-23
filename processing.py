import cv2 as cv

def preprocessing(img):
    img = img.copy()
    img = cv.resize(img, (1000, 1000))
    img = cv.GaussianBlur(img, (5,5), 2)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return img