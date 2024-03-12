# Sudoku Solver
Just another real-time Sudoku solver coded in Python using OpenCV and PyTorch. Capable of tackling Sudoku puzzles in images and videos, and projecting the solutions back onto the original visuals.

<p align="center">
  <img src="./static/solver_demo.gif" />
</p>

## Setup
After cloning the repository install the dependencies by running
```
pip install -r requirements.txt
```

If you just want to execute ``main.py`` you're good to go, as it comes with a pre-trained neural network for digit classification provided in ``model_file.pt``. 

Run the following command to verify that the setup is completed. The solved Sudoku should be displayed in a new window
```
python main.py -i ./images/sudoku_photo.jpg
```

If you want to retrain the neural network for digit classification you need to download the [dataset](https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset?resource=download) from Kaggle. Unzip the downloaded folder, rename it to ``archive`` and paste it into the project directory. Alternatively, you can store the unzipped folder in a directory of your choice and update the path in the ``ASSETS_PATH`` variable within ``classifier/dataset.py``. Now you should be ready to retrain the neural network using the ``classifier.ipynb`` notebook.

## Usage
For solving a Sudoku puzzle in an image use 
```
python main.py -i <PATH_TO_IMAGE>
```
The puzzle with the projected solution should be displayed in an new window.

For solving a Sudoku puzzle from a videocamera use 
```
python main.py -v
```
The video stream should be displayed in an new window. Optionally you can pass a device index (integer) to the ``-v`` argument used by OpenCV for camera selection. 

## Credits
The dataset used in this project was published by Kshitij Dhama on Kaggle.