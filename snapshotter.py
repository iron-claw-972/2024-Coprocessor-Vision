import typing
from queue import Empty, Queue, Full
from ultralytics.engine.results import Results # type: ignore
import cv2 as cv # type: ignore
import numpy as np
from time import sleep
import glob
import os
import datetime

snapshot_queue: Queue = Queue(10)

SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "/snapshots")

def write_image(img: np.ndarray, detections: Results) -> None:
    timestamp: str = datetime.datetime.now().isoformat()
    cv.imwrite(os.path.join(SNAPSHOT_PATH, timestamp + ".jpg"), img)
    with open(os.path.join(SNAPSHOT_PATH, timestamp + ".csv"), "w") as f:
        f.write(detections.to_csv())

def count_images() -> int:
    return len(glob.glob(SNAPSHOT_PATH + "/*.jpg"))

def run_snapshotter_thread() -> typing.NoReturn:
    written_with_detections = 0 # try to get some images without detections too
    img_count: int = count_images()
    while True:
        img: np.ndarray
        detections: Results
        img, detections = snapshot_queue.get()

        if (img_count >= 200): # stop at 200 images
            sleep(1)
            continue

        if (detections.boxes is not None):
            write_image(img, detections)
            img_count += 1
            written_with_detections += 1
        elif (img_count > 0): # take some non-detection images
            write_image(img, detections)
            img_count -= 1
            written_with_detections -= 1

