import os
import cv2
import numpy as np


def diff(image_1: np.ndarray, image_2: np.ndarray) -> np.ndarray:
    image = cv2.absdiff(image_1, image_2)
    _, image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
    kernel = np.ones((10, 10), np.uint8)
    image = cv2.erode(image, kernel)
    image = cv2.dilate(image, kernel)
    return image


def find_contours(image: np.ndarray) -> list:
    large_contours = []
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            large_contours.append(contour)

    return large_contours


def detect_with_opencv() -> None:
    files_list = sorted(os.listdir("../data/north-raw"), key=lambda f: int(f.split('_')[0].strip('s')))

    image_1 = None
    for file in files_list:
        image_2 = cv2.imread(os.path.join("../data/north-raw", file))
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        if image_1 is None:
            image_1 = image_2
            continue

        delta = diff(image_1, image_2)
        image_1 = image_2

        contours = find_contours(delta)

        delta = cv2.cvtColor(delta, cv2.COLOR_GRAY2RGB)
        cv2.drawContours(delta, contours, -1, (0, 0, 255), 10)
        cv2.imwrite(f"../data/{file}", delta)


if __name__ == '__main__':
    detect_with_opencv()
