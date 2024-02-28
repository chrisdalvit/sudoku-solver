import cv2 as cv
import numpy as np
import torch

from processing import preprocessing, find_sudoku_square, extract_cells, blend_images, draw_digits
from neural_network import NeuralNetwork

IMAGES_PATH = "./images/"

img = cv.cvtColor(cv.imread(IMAGES_PATH + "sudoku_photo.jpg"), cv.COLOR_BGR2RGB)
prep_img = preprocessing(img)
square, corners = find_sudoku_square(prep_img)
cells = extract_cells(square)

model = NeuralNetwork()
model.load_state_dict(torch.load("model_file.pt"))

def is_empty_cell(cell, cutoff=5, threshold=0.03):
    center = cell[cutoff:-cutoff,cutoff:-cutoff]
    center_size = (cell.shape[0]-2*cutoff)*(c.shape[1]-2*cutoff)
    return np.sum(center) / center_size > threshold

preds = []
for c in cells:
    if is_empty_cell(c):
        X = torch.tensor(c).unsqueeze(0).unsqueeze(0)
        preds.append(model(X).argmax(1).item() + 1)
    else:
        preds.append(None)

I = draw_digits(img, preds, corners)
outImage = blend_images(I, img, I)

cv.imshow("Square", outImage)
cv.waitKey(0)