import argparse
import os

import cv2 as cv
import numpy as np
import torch

from processing import preprocessing, find_sudoku_square, extract_cells, blend_images, draw_digits, _count_children
from neural_network import NeuralNetwork
from sudoku import Sudoku

CV_WINDOW_TITLE = "Sudoku Solver"

model = NeuralNetwork()
model.load_state_dict(torch.load("model_file.pt"))

def is_empty_cell(cell, cutoff=5, threshold=0.03):
    center = cell[cutoff:-cutoff,cutoff:-cutoff]
    center_size = (cell.shape[0]-2*cutoff)*(cell.shape[1]-2*cutoff)
    return np.sum(center) / center_size > threshold

def process_sudoku_image(img, model):
    prep_img = preprocessing(img)
    square, corners = find_sudoku_square(prep_img)
    if square is None:
        return cv.putText(img, "No Sudoku found", (200,200), cv.FONT_HERSHEY_COMPLEX, 3, (0,0,255), 3)
    cells = extract_cells(square)
    preds = []
    for c in cells:
        if is_empty_cell(c):
            X = torch.tensor(c).unsqueeze(0).unsqueeze(0)
            preds.append(model(X).argmax(1).item() + 1)
        else:
            preds.append(None)

    sudoku = Sudoku(preds)
    I = draw_digits(img, sudoku, corners)
    return blend_images(I, img, I)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
                    prog='Realtime Sudoku Solver',
                    description='A Python script that automatically solves Sudoku puzzles from image or video input.')
    parser.add_argument('-i', '--image', help="Path to Sudoku image")   
    parser.add_argument('-v', '--video', action="store_true", help="Flag for using the standard video input")   
    args = parser.parse_args()
    if args.video:
        print("Press 'q' key to end the Sudoku Solver.")
        cap = cv.VideoCapture(1)
        if not cap.isOpened():
            print("Sudoku Solver: Cannot open camera.")
            exit(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Sudoku Solver: Can't receive frame from camera. Abort Sudoku Solver.")
                exit(1)
            cv.imshow(CV_WINDOW_TITLE, process_sudoku_image(frame, model))
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
    elif args.image:
        if os.path.exists(args.image):
            img = cv.cvtColor(cv.imread(args.image), cv.COLOR_BGR2RGB)
            cv.imshow(CV_WINDOW_TITLE, process_sudoku_image(img, model))
            cv.waitKey(0)
        else:
            print(f"Sudoku Solver: Path {args.image} does not exist.")
    cv.destroyAllWindows()