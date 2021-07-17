import os
import re
import cv2
import argparse
import numpy as np
from typing import List
from Logger import Logger
from tqdm import tqdm

KI = 0
KT = 0
KD = 0


def extract_time_images(image: np.ndarray, text_info: tuple) -> List[np.ndarray]:
    time_images = []

    for x1, y1, x2, y2, color in text_info:
        time_image = image[y1:y2, x1:x2]
        # cv2.imwrite(f"{KD}_0-raw.jpg", time_image)
        time_image = cv2.cvtColor(time_image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f"{KD}_1-gray.jpg", time_image)
        time_image = 255 - time_image if color == "black" else time_image
        # cv2.imwrite(f"{KD}_2-reverse.jpg", time_image)
        _, time_image = cv2.threshold(time_image, 220, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f"{KD}_3-bw.jpg", time_image)
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


def videos2images() -> None:
    direction_videos("../赛题/A题/附件/附件二_北口", "../data/north-raw/", NORTH_SOUTH_TEXT_INFO)
    direction_videos("../赛题/A题/附件/附件四_南口", "../data/south-raw/", NORTH_SOUTH_TEXT_INFO)
    direction_videos("../赛题/A题/附件/附件五_西口", "../data/west-raw/", WEST_EAST_TEXT_INFO)
    direction_videos("../赛题/A题/附件/附件三_东口", "../data/east-raw/", WEST_EAST_TEXT_INFO)


def rename_direction_images(images_path: str, text_info: tuple, offset: int = 0) -> None:
    global KD

    hour, minute, second = 0, 0, 0
    previous_time_image = np.array(0, dtype=np.uint8)
    files_list = sorted(os.listdir(images_path), key=lambda x: int(x.split('.')[0].split('_')[0].strip('s')))

    for file in tqdm(files_list):
        file_id = int(file.split('.')[0].split('_')[0].strip('s'))
        if file_id < offset:
            continue

        src_path = os.path.join(images_path, file)
        image = cv2.imread(src_path)
        time_image = extract_time_images(image, text_info)[-1]
        delta_time_image = np.abs(previous_time_image.astype(np.int32) - time_image.astype(np.int32)).astype(np.uint8)
        previous_time_image = time_image

        # cv2.imwrite(f"{KD}_4-delta.jpg", delta_time_image)
        # KD += 1

        if re.match("[0-9]+s?_[0-9]{6}c.jpg", file):
            time = file.split('.')[0].split('_')[1].strip('c')
            hour, minute, second = int(time[:2]), int(time[2:4]), int(time[4:6])
        else:
            num_changed_pixels = np.sum(delta_time_image == 255)
            if num_changed_pixels >= 100:
                second += 1
                if second == 60:
                    second = 0
                    minute += 1
                if minute == 60:
                    minute = 0
                    hour += 1

            dst_path = os.path.join(images_path, f"{file.split('.')[0].split('_')[0]}_{hour:02d}{minute:02d}{second:02d}.jpg")
            os.rename(src_path, dst_path)


def rename_images(op: str) -> None:
    _, direction, offset = op.split('-')
    offset = int(offset)

    if direction == 'north':
        rename_direction_images("../data/north-raw", NORTH_SOUTH_TEXT_INFO, offset)
    elif direction == 'south':
        rename_direction_images("../data/south-raw", NORTH_SOUTH_TEXT_INFO, offset)
    elif direction == 'west':
        rename_direction_images("../data/west-raw", WEST_EAST_TEXT_INFO, offset)
    elif direction == 'east':
        rename_direction_images("../data/east-raw", WEST_EAST_TEXT_INFO, offset)


if __name__ == '__main__':
    NORTH_SOUTH_TEXT_INFO = (
        (1365, 118, 1418, 176, "black"), (1418, 118, 1475, 176, "black"),
        (1526, 118, 1586, 176, "white"), (1586, 118, 1648, 176, "white"),
        (1688, 118, 1755, 176, "white"), (1755, 118, 1812, 176, "white"),
    )
    WEST_EAST_TEXT_INFO = (
        (636, 84, 670, 132, "black"), (670, 84, 704, 132, "black"),
        (730, 84, 766, 132, "black"), (766, 84, 800, 132, "black"),
        (830, 84, 864, 132, "black"), (864, 84, 896, 132, "black"),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("operation")
    args = parser.parse_args()

    if args.operation == "videos2images":
        videos2images()
    elif args.operation.startswith("rename"):
        rename_images(args.operation)
