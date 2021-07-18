import os
import re
import pickle
from tqdm import tqdm
from typing import Tuple, List


class BoundingBox:
    def __init__(self, vehicle: str, confidence: int, left: int, top: int, right: int, bottom: int):
        self.vehicle = vehicle
        self.confidence = confidence
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2


def extract_bounding_boxes(content: List[str]) -> List[BoundingBox]:
    bounding_boxes_list = []
    cached_list = []

    for line in content[1:]:
        if result := re.findall("([a-z ]+): ([0-9]+)%", line):
            vehicle, confidence = result[0]
            confidence = int(confidence)
            bounding_box = BoundingBox(vehicle=vehicle, confidence=confidence, left=0, top=0, right=0, bottom=0)
            cached_list.append(bounding_box)
        else:
            result = re.findall("Bounding Box: Left=([0-9]+), Top=([0-9]+), Right=([0-9]+), Bottom=([0-9]+)", line)
            left, top, right, bottom = result[0]
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            for bounding_box in cached_list:
                bounding_box.left, bounding_box.top, bounding_box.right, bounding_box.bottom = left, top, right, bottom
                bounding_boxes_list.append(bounding_box)
            cached_list = []

    return bounding_boxes_list


def filter_bounding_boxes(bounding_boxes_list: List[BoundingBox]) -> List[BoundingBox]:
    bounding_boxes_list = [box for box in bounding_boxes_list if box.vehicle in ("car", "bus", "truck", "train")]

    to_removed_index_set = set()

    for i, bb1 in enumerate(bounding_boxes_list):
        for j, bb2 in enumerate(bounding_boxes_list):
            if i >= j:
                continue
            elif bb2.top <= bb1.top <= bb1.bottom <= bb2.bottom and bb2.left <= bb1.left <= bb1.right <= bb2.right:
                to_removed_index_set.add(i)
            elif bb1.top <= bb2.top <= bb2.bottom <= bb1.bottom and bb1.left <= bb2.left <= bb2.right <= bb1.right:
                to_removed_index_set.add(j)
            else:
                top, bottom = max(bb1.top, bb2.top), min(bb1.bottom, bb2.bottom)
                left, right = max(bb1.left, bb2.left), min(bb1.right, bb2.right)
                s1 = (bb1.bottom - bb1.top) * (bb1.right - bb1.left)
                s2 = (bb2.bottom - bb2.top) * (bb2.right - bb2.left)
                ss = (bottom - left) * (right - left)
                if top <= bottom and left <= right and 2 * ss > s1 and 2 * ss > s2:
                    to_removed_index_set.add(i)

    for i in sorted(to_removed_index_set, reverse=True):
        bounding_boxes_list.pop(i)

    return bounding_boxes_list


def detect_single_direction(input_path: str, output_file_path: str) -> None:
    files_list = os.listdir(input_path)
    files_list = [f for f in files_list if f.endswith(".txt")]
    files_list = sorted(files_list, key=lambda f: int(f.split('_')[0].strip('s')))

    bounding_boxes_dict = {}

    for file in tqdm(files_list):
        file_path = os.path.join(input_path, file)
        with open(file_path, 'r') as f:
            content = f.readlines()
            bounding_boxes_list = extract_bounding_boxes(content)
            bounding_boxes_list = filter_bounding_boxes(bounding_boxes_list)

            seq_id, time_id = file.split('.')[0].split('_')
            seq_id = f"{int(seq_id.rstrip('s')):05d}"
            time_id = time_id.rstrip('c')
            file_id = f"{seq_id}_{time_id}"
            bounding_boxes_dict[file_id] = bounding_boxes_list

    with open(output_file_path, "wb") as output_file:
        pickle.dump(bounding_boxes_dict, output_file)


def detect(op: str) -> None:
    _, direction = op.split('-')
    input_path = f"../data/{direction}-detect"
    output_file_path = f"../data/tracks/{direction}.pickle"
    detect_single_direction(input_path, output_file_path)


if __name__ == '__main__':
    detect("tracks-north")
    detect("tracks-south")
    detect("tracks-west")
    detect("tracks-east")
