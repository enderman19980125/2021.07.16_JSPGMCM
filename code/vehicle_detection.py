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


def filter_contours(contours: list) -> list:
    large_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 3000:
            large_contours.append(contour)
    return large_contours


def find_contours_1(image: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 10)
    return image


def find_contours_2(image: np.ndarray) -> list:
    cascade_classifier = cv2.CascadeClassifier('cars.xml')
    contours = cascade_classifier.detectMultiScale(image, 1.1, 1)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for (x, y, w, h) in contours:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 10)
    return image


def detect_with_opencv() -> None:
    files_list = sorted(os.listdir("../data/north-raw"), key=lambda f: int(f.split('_')[0].strip('s')))

    image_1 = None
    for file in files_list:
        image_2 = cv2.imread(os.path.join("../data/north-raw", file))
        image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

        if image_1 is None:
            image_1 = image_2
            continue

        # delta = diff(image_1, image_2)
        # delta = find_contours_1(delta)
        delta = find_contours_2(image_2)
        cv2.imwrite(f"../data/{file}", delta)
        image_1 = image_2


if __name__ == '__main__':
    detect_with_opencv()
