#!/usr/bin/env python3
from threading import Thread
import cv2
import numpy as np
import ultralytics # type: ignore
from ultralytics import YOLO # type: ignore
from ultralytics.engine.results import Results # type: ignore
import time
import ntables
import signal
import line_profiler
from queue import Queue, Empty

# ANSI colors
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

# Define the video files for the trackers
# Path to video files, 0 for webcam, 1 for external camera
# On a robot, these should be in the same order as the cameras in VisionConstants.java
# TODO: This should only be the cameras we need, not every possibility
cameras: list[int] = [i for i in range(5)]

def handle_signal(signalnum, stack_frame):
    raise SystemExit()

# handle SIGTERM nicely
signal.signal(signal.SIGTERM, handle_signal)

# Load the model
model: ultralytics.models.yolo.model.YOLO = YOLO('models/best.pt')
print(type(model))

queues: list[Queue] = []
rendered: Queue = Queue(maxsize=2)
active: dict[int, bool] = {}

# exit gracefully on ^C
is_interrupted: bool = False


#@line_profiler.profile
def run_cam_in_thread(q: Queue, cameraname: int) -> None:
    video: cv2.VideoCapture = cv2.VideoCapture(cameraname)
    print(f"Camera: {cameraname}") # For debugging

    while True:
        # read one frame
        ret: bool
        frame: np.ndarray
        ret, frame = video.read()  # Read the video frames

        if is_interrupted or not ret:
            print(f"CAMERA {cameraname} EXITING")
            break

        #cv2.imshow(f"Tracking_Stream_{cameraname}", frame)
        #key = cv2.waitKey(1)
        q.put_nowait(frame.copy())

    active[cameraname] = False
    q.put(np.zeros((640, 480, 3)))
    video.release()


#@line_profiler.profile
def run_tracker_in_thread(q: Queue, cameraname: int, file_index: int, rendered: Queue) -> None:
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        cameraname (int): The identifier for the webcam/external camera source.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """

    while True:

        if is_interrupted or not active[cameraname]:
            break

        start_time: float = time.time()
        frame = q.get(block=True)

        # Track objects in frames if available
        results: list[Results] = model.track(frame, persist=True)
        res_plotted: np.ndarray = results[0].plot()
        # Calculate offsets and add to NetworkTables
        #ntables.add_results(results, file_index) # FIXME
        end_time: float = time.time()

        #fps = str(round(1/(end_time-start_time), 2))
        #cv2.putText(res_plotted, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA)

        rendered.put(res_plotted.copy())


threads: list[Thread] = []
for i in range(len(cameras)):
    q: Queue = Queue(maxsize=2)
    queues.append(q)
    active[cameras[i]] = True

    # daemon=True makes it shut down if something goes wrong
    tracker = Thread(target=run_tracker_in_thread, args=(q, cameras[i], i, rendered), daemon=True)
    camera_thread = Thread(target=run_cam_in_thread, args=(q, cameras[i]), daemon=True)
    # Add to the array to use later
    threads.append(tracker)
    threads.append(camera_thread)

    # Start the thread
    tracker.start()
    camera_thread.start()

try:
    frame: np.ndarray = np.zeros((640, 320, 3))
    while True:
        frame = rendered.get(block=False)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
except (KeyboardInterrupt, SystemExit) as e:
    print(COLOR_BOLD, "INTERRUPT RECIEVED -- EXITING", COLOR_RESET, sep="")
    is_interrupted = True

# Wait for the tracker threads to finish
for thread in threads:
    thread.join()

# Clean up and close windows
cv2.destroyAllWindows()

