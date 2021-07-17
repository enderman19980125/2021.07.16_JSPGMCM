import os
import cv2
import numpy as np
from typing import List
from Logger import Logger
from tqdm import tqdm

KI = 0
KT = 0


def extract_time_images(image: np.ndarray, text_info: tuple) -> List[np.ndarray]:
    time_images = []

    for x1, y1, x2, y2, color in text_info:
        time_image = image[y1:y2, x1:x2]
        time_image = cv2.cvtColor(time_image, cv2.COLOR_BGR2GRAY)
        time_image = 255 - time_image if color == "black" else time_image
        _, time_image = cv2.threshold(time_image, 220, 255, cv2.THRESH_BINARY)
        time_images.append(time_image)

    return time_images


def save_time_images(time_images: List[np.ndarray]) -> None:
    global KT
    for time_image in time_images[5:]:
        file_path = os.path.join("../data/digits/u", f"{KT}.jpg")
        cv2.imwrite(file_path, time_image)
        KT += 1


def save_images(output_path: str, image: np.ndarray, is_first: bool) -> None:
    global KI
    file_path = os.path.join(output_path, f"{KI}s.jpg" if is_first else f"{KI}.jpg")
    cv2.imwrite(file_path, image)
    KI += 1


def video(input_file: str, output_path: str, text_info: tuple) -> None:
    logger = Logger("videos2images", "videos2images.log")
    logger.info(f"Processing {input_file} ...")

    video_capture = cv2.VideoCapture(input_file)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(total_frames)):
        flag, image = video_capture.read()

        # time_images = extract_time_images(image, text_info)
        # save_time_images(time_images)

        save_images(output_path, image, i == 0)


def direction_videos(input_path: str, output_path: str, text_info: tuple) -> None:
    global KI
    KI = 0

    for filename in sorted(os.listdir(input_path)):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        file_path = os.path.join(input_path, filename)
        video(file_path, output_path, text_info)


if __name__ == '__main__':
    direction_videos(
        input_path="../赛题/A题/附件/附件二_北口",
        output_path="../data/north-raw/",
        text_info=(
            (1365, 118, 1418, 176, "black"), (1418, 118, 1475, 176, "black"),
            (1526, 118, 1586, 176, "white"), (1586, 118, 1648, 176, "white"),
            (1688, 118, 1755, 176, "white"), (1755, 118, 1812, 176, "white"),
        ))
    direction_videos(
        input_path="../赛题/A题/附件/附件四_南口",
        output_path="../data/south-raw/",
        text_info=(
            (1365, 118, 1418, 176, "white"), (1418, 118, 1475, 176, "white"),
            (1526, 118, 1586, 176, "white"), (1586, 118, 1648, 176, "white"),
            (1688, 118, 1755, 176, "white"), (1755, 118, 1812, 176, "white"),
        ))
    direction_videos(
        input_path="../赛题/A题/附件/附件五_西口",
        output_path="../data/west-raw/",
        text_info=(
            (636, 84, 670, 132, "black"), (670, 84, 704, 132, "black"),
            (730, 84, 766, 132, "black"), (766, 84, 800, 132, "black"),
            (830, 84, 864, 132, "black"), (864, 84, 896, 132, "black"),
        ))
    direction_videos(
        input_path="../赛题/A题/附件/附件三_东口",
        output_path="../data/east-raw/",
        text_info=(
            (636, 84, 670, 132, "black"), (670, 84, 704, 132, "black"),
            (730, 84, 766, 132, "black"), (766, 84, 800, 132, "black"),
            (830, 84, 864, 132, "black"), (864, 84, 896, 132, "black"),
        ))
