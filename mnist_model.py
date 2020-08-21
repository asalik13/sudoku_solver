# +
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json


def loadModel():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    return loaded_model


def splitImage(sudoku, model):
    sudoku = cv2.resize(sudoku, (450, 450))
    grid = np.zeros([9, 9])
    for i in range(9):
        for j in range(9):
            image = sudoku[i * 50:(i + 1) * 50, j * 50:(j + 1) * 50]
            if image.sum() > 100000:
                grid[i][j] = mnist(image, model)
            else:
                grid[i][j] = 0

    grid = grid.astype(int)
    return grid


def mnist(image, model):
    image_resize = cv2.resize(image, (28, 28))
    image = image_resize.reshape(1, 28, 28, 1)
    digit = model.predict_classes(image)[0]
    return digit
