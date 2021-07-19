import os
import cv2
import pickle
import numpy as np
from typing import Tuple, List, Dict

from bounding_boxes_combination import BoundingBox


class BoundingBoxChain(BoundingBox):
    def __init__(self, bounding_box: BoundingBox, seq_id: str, hour: int, minute: int, second: int):
        super().__init__(vehicle=bounding_box.vehicle, confidence=bounding_box.confidence,
                         left=bounding_box.left, top=bounding_box.top, right=bounding_box.right, bottom=bounding_box.bottom)
        self.seq_id = seq_id
        self.hour = hour
        self.minute = minute
        self.second = second
        self.lane_id = -1
        self.previous = None
        self.next = None


class Mask:
    def __init__(self, image: np.ndarray, direction: str, lane_id: int, is_left: bool, is_straight: bool, is_right: bool):
        self.image = image
        self.direction = direction
        self.lane_id = lane_id
        self.is_left = is_left
        self.is_straight = is_straight
        self.is_right = is_right


def load_bounding_boxes(direction: str) -> Dict[str, List[BoundingBox]]:
    with open(f"../data/tracks/{direction}.pickle", 'rb') as input_file:
        bounding_boxes_dict = pickle.load(input_file)
    return bounding_boxes_dict


def save_tracks(direction: str, obj: object) -> None:
    with open(f"../data/tracks/{direction}-tracks.pickle", 'wb') as output_file:
        pickle.dump(obj, output_file)


def load_masks(direction: str) -> Tuple[Mask, List[Mask]]:
    valid_mask = cv2.imread(f"../data/masks/{direction}-valid.jpg")
    valid_mask = cv2.cvtColor(valid_mask, cv2.COLOR_BGR2GRAY)
    valid_mask = Mask(image=valid_mask, direction=direction, lane_id=0, is_left=False, is_straight=False, is_right=False)

    files_list = os.listdir("../data/masks")
    files_list = [f for f in files_list if f.startswith(f"{direction}-lane") and f.endswith(".jpg")]
    lane_masks_list = []

    for file in files_list:
        file_path = os.path.join("../data/masks", file)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, _, lane_id, lsr = file.split('.')[0].split('-')
        lane_id = int(lane_id)
        mask = Mask(image=image, direction=direction, lane_id=lane_id,
                    is_left=lsr.find('l') >= 0, is_straight=lsr.find('s') >= 0, is_right=lsr.find('r') >= 0)
        lane_masks_list.append(mask)

    return valid_mask, lane_masks_list


def extract_seq_and_time(file_id: str) -> Tuple[str, int, int, int]:
    seq_id, time_id = file_id.split('_')
    hour, minute, second = int(time_id[:2]), int(time_id[2:4]), int(time_id[4:])
    return seq_id, hour, minute, second


def detect_position(bounding_box: BoundingBoxChain, valid_mask: Mask, lane_masks_list: List[Mask]) -> int:
    x, y = bounding_box.center
    lane_id = -1

    if valid_mask.image[y][x] < 100:
        lane_id = 0

    for lane_mask in lane_masks_list:
        if lane_mask.image[y][x] < 100:
            lane_id = lane_mask.lane_id

    return lane_id


def link_bounding_box(bounding_box: BoundingBoxChain, open_bounding_boxes_list: List[BoundingBoxChain]) -> BoundingBoxChain:
    x, y = bounding_box.center
    linked_bounding_box = None
    linked_distance = 0

    for open_bounding_box in open_bounding_boxes_list:
        xx, yy = open_bounding_box.center

        if open_bounding_box.next or xx < x - 500 or x + 500 < xx or yy < y - 100:
            continue
        if 0 < bounding_box.lane_id != open_bounding_box.lane_id:
            continue
        if bounding_box.lane_id == 0 and open_bounding_box.lane_id == -1:
            continue
        if bounding_box.lane_id == -1 and open_bounding_box.lane_id > 0:
            continue

        distance = int(((x - xx) ** 2 + (y - yy) ** 2) ** 0.5)
        if distance > 500:
            continue

        if linked_bounding_box and distance < linked_distance:
            linked_bounding_box = open_bounding_box
            linked_distance = distance
        elif linked_bounding_box is None:
            linked_bounding_box = open_bounding_box
            linked_distance = distance

    return linked_bounding_box


def clean(seq_id: str, open_bounding_boxes_list: List[BoundingBoxChain], closed_bounding_boxes_list: List[BoundingBoxChain]) -> None:
    to_moved_list = []

    for i, bounding_box in enumerate(open_bounding_boxes_list):
        if bounding_box.lane_id == -1 or (bounding_box.lane_id == 0 and int(seq_id) - int(bounding_box.seq_id) > 10):
            to_moved_list.append(i)

    to_moved_list.sort(reverse=True)
    for i in to_moved_list:
        bounding_box = open_bounding_boxes_list.pop(i)
        closed_bounding_boxes_list.append(bounding_box)


def show(closed_bounding_boxes_list) -> None:
    for first_bounding_box in closed_bounding_boxes_list:
        if first_bounding_box.previous is None:
            last_bounding_box = first_bounding_box
            while last_bounding_box.next:
                last_bounding_box = last_bounding_box.next
            print(
                f"Enter [@{first_bounding_box.lane_id}]"
                f"[#{first_bounding_box.seq_id}-{first_bounding_box.hour}:{first_bounding_box.minute}:{first_bounding_box.second}] "
                f"Leave [@{last_bounding_box.lane_id}]"
                f"[#{last_bounding_box.seq_id}-{last_bounding_box.hour}:{last_bounding_box.minute}:{last_bounding_box.second}]"
            )


def detect_single_direction_tracks(direction: str) -> None:
    bounding_boxes_dict = load_bounding_boxes(direction)
    valid_mask, lane_masks_list = load_masks(direction)
    open_bounding_boxes_list = []
    closed_bounding_boxes_list = []

    for file_id, bounding_boxes_list in bounding_boxes_dict.items():
        seq_id, hour, minute, second = extract_seq_and_time(file_id)
        cached_bounding_boxes_list = []
        bounding_boxes_list.sort(key=lambda bb: bb.center[1], reverse=True)

        if seq_id >= '00500':
            break

        for bounding_box in bounding_boxes_list:
            bounding_box = BoundingBoxChain(bounding_box, seq_id, hour, minute, second)
            bounding_box.lane_id = detect_position(bounding_box, valid_mask, lane_masks_list)
            linked_bounding_box = link_bounding_box(bounding_box, open_bounding_boxes_list)

            if linked_bounding_box:
                bounding_box.previous = linked_bounding_box
                linked_bounding_box.next = bounding_box
                open_bounding_boxes_list.remove(linked_bounding_box)
                closed_bounding_boxes_list.append(linked_bounding_box)
                cached_bounding_boxes_list.append(bounding_box)
            elif bounding_box.lane_id > 0:
                cached_bounding_boxes_list.append(bounding_box)

        open_bounding_boxes_list.extend(cached_bounding_boxes_list)
        clean(seq_id, open_bounding_boxes_list, closed_bounding_boxes_list)

    closed_bounding_boxes_list.extend(open_bounding_boxes_list)
    save_tracks(direction, closed_bounding_boxes_list)
    show(closed_bounding_boxes_list)


def detect_tracks(op: str) -> None:
    _, direction = op.split('-')
    detect_single_direction_tracks(direction)


if __name__ == '__main__':
    # detect_tracks("tracks-north")
    # detect_tracks("tracks-south")
    # detect_tracks("tracks-west")
    detect_tracks("tracks-east")
