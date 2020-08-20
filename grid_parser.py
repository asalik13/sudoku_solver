import cv2
import numpy as np
import operator
from utils import extract_digit, distance_between


def gausianBlur(image):
    proc = cv2.GaussianBlur(image.copy(), (9, 9), 0)
    proc = cv2.adaptiveThreshold(
        proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return proc


def invertAndDilate(image, skip_dilate):
    proc = cv2.bitwise_not(image.copy(), image.copy())
    if not skip_dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(proc, kernel)
    return proc


def findCorners(image):
    contours, h = cv2.findContours(
        image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                                     polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                                 polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                                    polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                                  polygon]), key=operator.itemgetter(1))

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right]
            [0], polygon[bottom_left][0]]


def perspectiveTransform(crop_rect, image):
    top_left, top_right, bottom_right, bottom_left = crop_rect[
        0], crop_rect[1], crop_rect[2], crop_rect[3]
    src = np.array([top_left, top_right, bottom_right,
                    bottom_left], dtype='float32')
    side = max([distance_between(bottom_right, top_right),
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),
                distance_between(top_left, top_right)])
    # Describe a square with side of the calculated length, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1],
                    [0, side - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv2.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv2.warpPerspective(image, m, (int(side), int(side)))


def gridToSquares(image):
    squares = []
    side = image.shape[:1]
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner
            squares.append((p1, p2))


def preprocessImage(image, skip_dilate):
    proc = gausianBlur(image)
    proc = invertAndDilate(proc, skip_dilate)
    return proc


def getSquares(image):
    squares = []
    side = image.shape[:1]
    side = side[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  # Top left corner of a box
            p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner
            squares.append((p1, p2))
    return squares


def getDigits(squares, image, size):
    digits = []
    img = preprocessImage(image.copy(), skip_dilate=True)
    for square in squares:
        digits.append(extract_digit(img, square, size))
    return digits


def showDigits(digits, colour=255):
    """Shows list of 81 extracted digits in a grid format"""
    rows = []
    with_border = [cv2.copyMakeBorder(
        img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    img = np.concatenate(rows)
    return img


def grid_parser(path):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow("normal", image)
    processed = preprocessImage(image, False)
    corners = findCorners(processed)
    cropped = perspectiveTransform(corners, image)
    squares = getSquares(cropped)
    digits = getDigits(squares, cropped, 28)
    finalImage = showDigits(digits)
    cv2.imshow("final", finalImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image = cv2.imread("sudoku.jpg", cv2.IMREAD_GRAYSCALE)
grid_parser("sudoku.jpg")
