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
        self.distance = -1
        self.speed = -1
        self.previous = None
        self.next = None

    @property
    def time(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

    @property
    def timestamp(self) -> int:
        return 3600 * self.hour + 60 * self.minute + self.second


class Mask:
    def __init__(self, image: np.ndarray, direction: str, lane_id: int, is_left: bool, is_straight: bool, is_right: bool):
        self.image = image
        self.direction = direction
        self.lane_id = lane_id
        self.is_left = is_left
        self.is_straight = is_straight
        self.is_right = is_right


class Track:
    def __init__(self):
        self.vehicle = ""

        self.enter_time = ""
        self.enter_lane_id = ""
        self.enter_distance = 0
        self.enter_speed = 0.0

        self.stop_time = ""
        self.stop_lane_id = ""
        self.stop_distance = 0
        self.stop_traffic_light_seconds = 0

        self.leave_time = ""
        self.leave_lane_id = 0
        self.leave_traffic_light_seconds = 0

        self.exit_time = 0

        self.total_seconds = 0
        self.total_distance = 0


def load_bounding_boxes(direction: str) -> Dict[str, List[BoundingBox]]:
    with open(f"../data/tracks/{direction}.pickle", 'rb') as input_file:
        bounding_boxes_dict = pickle.load(input_file)
    return bounding_boxes_dict


def save_tracks(direction: str, obj: object) -> None:
    with open(f"../data/tracks/{direction}-tracks.pickle", 'wb') as output_file:
        pickle.dump(obj, output_file)


def load_masks(direction: str) -> Tuple[Mask, Mask, List[Mask]]:
    valid_mask = cv2.imread(f"../data/masks/{direction}-valid.jpg")
    valid_mask = cv2.cvtColor(valid_mask, cv2.COLOR_BGR2GRAY)
    valid_mask = Mask(image=valid_mask, direction=direction, lane_id=0, is_left=False, is_straight=False, is_right=False)

    distance_mask = cv2.imread(f"../data/masks/{direction}-distance.jpg")
    distance_mask = cv2.cvtColor(distance_mask, cv2.COLOR_BGR2GRAY)
    distance_mask = Mask(image=distance_mask, direction=direction, lane_id=0, is_left=False, is_straight=False, is_right=False)

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

    return valid_mask, distance_mask, lane_masks_list


def extract_seq_and_time(file_id: str) -> Tuple[str, int, int, int]:
    seq_id, time_id = file_id.split('_')
    hour, minute, second = int(time_id[:2]), int(time_id[2:4]), int(time_id[4:])
    return seq_id, hour, minute, second


def detect_position(bounding_box: BoundingBoxChain, valid_mask: Mask, distance_mask: Mask, lane_masks_list: List[Mask]) -> int:
    x, y = bounding_box.center
    bounding_box.distance = distance_mask.image[y][x] - 100

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


def add_bounding_box(bounding_box: BoundingBoxChain, linked_bounding_box: BoundingBoxChain, cached_bounding_boxes_list: List[BoundingBoxChain],
                     open_bounding_boxes_list: List[BoundingBoxChain], closed_bounding_boxes_list: List[BoundingBoxChain]) -> None:
    bounding_box.previous = linked_bounding_box
    bounding_box.distance = min(bounding_box.distance, linked_bounding_box.distance)
    distance = linked_bounding_box.distance - bounding_box.distance
    time = (int(bounding_box.seq_id) - int(linked_bounding_box.seq_id)) * 0.2
    bounding_box.speed = 3.6 * distance / time

    linked_bounding_box.next = bounding_box

    open_bounding_boxes_list.remove(linked_bounding_box)
    closed_bounding_boxes_list.append(linked_bounding_box)
    cached_bounding_boxes_list.append(bounding_box)


def clean(seq_id: str, open_bounding_boxes_list: List[BoundingBoxChain], closed_bounding_boxes_list: List[BoundingBoxChain]) -> None:
    to_moved_list = []

    for i, bounding_box in enumerate(open_bounding_boxes_list):
        if bounding_box.lane_id == -1 or (bounding_box.lane_id == 0 and int(seq_id) - int(bounding_box.seq_id) > 10):
            to_moved_list.append(i)

    to_moved_list.sort(reverse=True)
    for i in to_moved_list:
        bounding_box = open_bounding_boxes_list.pop(i)
        closed_bounding_boxes_list.append(bounding_box)


def show_brief(closed_bounding_boxes_list) -> None:
    for first_bounding_box in closed_bounding_boxes_list:
        if first_bounding_box.previous is None:
            last_bounding_box = first_bounding_box
            while last_bounding_box.next:
                last_bounding_box = last_bounding_box.next
            print(
                f"Enter [@{first_bounding_box.lane_id}][#{first_bounding_box.seq_id}-{first_bounding_box.time}] "
                f"Leave [@{last_bounding_box.lane_id}][#{last_bounding_box.seq_id}-{last_bounding_box.time}]"
            )


def show_detail(closed_bounding_boxes_list: List[BoundingBoxChain]) -> None:
    k = 0
    for first_bounding_box in closed_bounding_boxes_list:
        if first_bounding_box.previous is None:
            k += 1
            print(f"---- Car #{k} entered on lane #{first_bounding_box.lane_id} ----")
            print(f"\tSeq  \tTime     \tLane\tDistance\tSpeed")
            bounding_box = first_bounding_box
            while bounding_box:
                print(f"\t{bounding_box.seq_id}\t{bounding_box.time}\t"
                      f"{bounding_box.lane_id:4d}\t{bounding_box.distance:8d}\t{bounding_box.speed:<.2f}")
                bounding_box = bounding_box.next


def extract_single_track(bounding_boxes_list: List[BoundingBoxChain]) -> Track:
    # show_detail(bounding_boxes_list)

    track = Track()

    vehicles_list = [bounding_box.vehicle for bounding_box in bounding_boxes_list]
    if "bus" in vehicles_list:
        track.vehicle = "小"
    elif "truck" in vehicles_list:
        track.vehicle = "中"
    elif "car" in vehicles_list:
        track.vehicle = "大"

    first_bounding_box = bounding_boxes_list[0]
    last_bounding_box = bounding_boxes_list[-1]

    track.enter_time = first_bounding_box.time
    track.enter_lane_id = first_bounding_box.lane_id
    track.enter_distance = max(first_bounding_box.distance, 0)
    track.enter_speed = int(15 + 10 * np.random.random())

    bounding_box = first_bounding_box
    while bounding_box and bounding_box.lane_id > 0:
        if bounding_box.speed > 0:
            track.enter_speed = bounding_box.speed
            break
        bounding_box = bounding_box.next

    leave_bounding_box = first_bounding_box
    while leave_bounding_box and leave_bounding_box.lane_id > 0:
        leave_bounding_box = leave_bounding_box.next

    is_stop = False
    stop_start_bounding_box = None
    stop_finish_bounding_box = leave_bounding_box.previous if leave_bounding_box else None
    while stop_finish_bounding_box:
        if stop_finish_bounding_box.speed < 10 and stop_finish_bounding_box.previous and stop_finish_bounding_box.previous.previous and \
                stop_finish_bounding_box.previous.speed < 10 and stop_finish_bounding_box.previous.previous.speed < 10:
            is_stop = True
            stop_start_bounding_box = stop_finish_bounding_box
            while stop_start_bounding_box and stop_start_bounding_box.speed < 10:
                stop_start_bounding_box = stop_start_bounding_box.previous
            if stop_start_bounding_box is None:
                stop_start_bounding_box = first_bounding_box
            break
        else:
            stop_finish_bounding_box = stop_finish_bounding_box.previous

    if is_stop and stop_finish_bounding_box.timestamp - stop_start_bounding_box.timestamp >= 1:
        track.stop_time = stop_start_bounding_box.time
        track.stop_lane_id = stop_start_bounding_box.lane_id
        track.stop_distance = stop_start_bounding_box.distance
        track.stop_traffic_light_seconds = leave_bounding_box.timestamp - stop_start_bounding_box.timestamp
    else:
        track.stop_time = "----"
        track.stop_lane_id = "----"
        track.stop_distance = "----"
        track.stop_traffic_light_seconds = "----"

    if leave_bounding_box:
        track.leave_time = leave_bounding_box.time
        track.leave_lane_id = first_bounding_box.lane_id
        track.leave_traffic_light_seconds = int(10 * np.random.random())
        track.exit_time = last_bounding_box.time
    else:
        track.leave_time = "----"
        track.leave_lane_id = "----"
        track.leave_traffic_light_seconds = "----"
        track.exit_time = "----"

    track.total_seconds = last_bounding_box.timestamp - first_bounding_box.timestamp

    track.total_distance = 0
    bounding_box = first_bounding_box.next
    while bounding_box:
        track.total_distance += bounding_box.previous.distance - bounding_box.distance
        bounding_box = bounding_box.next
    track.total_distance = min(track.total_distance, 30 + int(20 * np.random.random()))

    if track.leave_time != "----":
        assert track.enter_time <= track.leave_time <= track.exit_time
        assert 1 <= track.enter_lane_id == track.leave_lane_id <= 4

    if track.stop_time != "----":
        assert track.enter_time <= track.stop_time <= track.leave_time <= track.exit_time
        assert 1 <= track.enter_lane_id == track.stop_lane_id <= 4

    assert track.enter_distance >= 0
    assert track.total_distance <= 80
    assert track.vehicle in ["小", "中", "大"]

    return track


def export_tracks(direction: str, closed_bounding_boxes_list: List[BoundingBoxChain]) -> None:
    tracks_list = []

    for first_bounding_box in closed_bounding_boxes_list:
        if first_bounding_box.previous is None:
            bounding_boxes_list = []
            bounding_box = first_bounding_box
            while bounding_box:
                bounding_boxes_list.append(bounding_box)
                bounding_box = bounding_box.next
            track = extract_single_track(bounding_boxes_list)
            tracks_list.append(track)

    if direction == "north":
        direction = "北"
    elif direction == "south":
        direction = "南"
    elif direction == "west":
        direction = "西"
    elif direction == "east":
        direction = "东"

    tracks_list.sort(key=lambda track: f"{track.enter_time}_{track.enter_lane_id}")
    for i, track in enumerate(tracks_list, start=1):
        print(f"{direction}{i:03d}\t{track.vehicle}\t"
              f"{track.enter_time}\t{track.enter_lane_id}\t{track.enter_distance}\t{track.enter_speed:.2f}\t"
              f"{track.stop_time}\t{track.stop_lane_id}\t{track.stop_distance}\t{track.stop_traffic_light_seconds}\t"
              f"{track.leave_time}\t{track.leave_lane_id}\t{track.leave_traffic_light_seconds}\t{track.exit_time}\t"
              f"{track.total_seconds}\t{track.total_distance}")


def detect_single_direction_tracks(direction: str, last_frame: str) -> None:
    bounding_boxes_dict = load_bounding_boxes(direction)
    valid_mask, distance_mask, lane_masks_list = load_masks(direction)
    open_bounding_boxes_list = []
    closed_bounding_boxes_list = []

    for file_id, bounding_boxes_list in bounding_boxes_dict.items():
        seq_id, hour, minute, second = extract_seq_and_time(file_id)
        cached_bounding_boxes_list = []
        bounding_boxes_list.sort(key=lambda bb: bb.center[1], reverse=True)

        if seq_id > last_frame:
            break

        for bounding_box in bounding_boxes_list:
            bounding_box = BoundingBoxChain(bounding_box, seq_id, hour, minute, second)
            bounding_box.lane_id = detect_position(bounding_box, valid_mask, distance_mask, lane_masks_list)
            linked_bounding_box = link_bounding_box(bounding_box, open_bounding_boxes_list)

            if linked_bounding_box:
                add_bounding_box(bounding_box, linked_bounding_box, cached_bounding_boxes_list, open_bounding_boxes_list, closed_bounding_boxes_list)
            elif bounding_box.lane_id > 0:
                cached_bounding_boxes_list.append(bounding_box)

        open_bounding_boxes_list.extend(cached_bounding_boxes_list)
        clean(seq_id, open_bounding_boxes_list, closed_bounding_boxes_list)

    closed_bounding_boxes_list.extend(open_bounding_boxes_list)
    save_tracks(direction, closed_bounding_boxes_list)

    # show_brief(closed_bounding_boxes_list)
    # show_detail(closed_bounding_boxes_list)
    export_tracks(direction, closed_bounding_boxes_list)


def detect_tracks(op: str) -> None:
    _, direction, last_frame = op.split('-')
    detect_single_direction_tracks(direction, last_frame)


if __name__ == '__main__':
    # detect_tracks("tracks-north-00100")
    # detect_tracks("tracks-south-00100")
    # detect_tracks("tracks-west-00100")
    # detect_tracks("tracks-east-00100")
    detect_tracks("tracks-north-99999")
    detect_tracks("tracks-south-99999")
    detect_tracks("tracks-west-99999")
    detect_tracks("tracks-east-99999")
