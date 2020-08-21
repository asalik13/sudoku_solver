from grid_parser import grid_parser
from mnist_model import loadModel, splitImage
from solver import solveSudoku
import cv2
model = loadModel()
parsed_grid = grid_parser("sudoku.jpg")
board = splitImage(parsed_grid, model)
solveSudoku(board)
print(board)
