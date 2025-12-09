#! ./venv/bin/python3
from threading import Thread
import time
import cv2
import numpy as np
from ultralytics import YOLO  # type: ignore
from ultralytics.engine.results import Results  # type: ignore
import time
import ntables
import snapshotter
import signal
import sys
import platform
import functools
import subprocess
from mjpeg_streamer import MjpegServer, Stream
from queue import Empty, Queue, Full
import util

# ANSI colors
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

# are we running interactively?
is_interactive: bool = sys.stderr.isatty()

# disable when not needed to improve performance
enable_mjpeg: bool = is_interactive


@functools.cache  # only run once
def get_ips() -> list[str]:
    ip_list: list[str] = []
    if platform.system() == "Linux":  # screw windows
        ipstr: str = (
            subprocess.check_output(["hostname", "-I"])
            .decode("UTF-8")
            .rstrip("\n")
            .strip()
        )
        ip_list = ipstr.split(" ")
    ip_list.append("localhost")
    return ip_list


# Define the video files for the trackers
# Path to video files, 0 for webcam, 1 for external camera
# On a robot, these should be in the same order as the cameras in VisionConstants.java
# TODO: This should only be the cameras we need, not every possibility
cameras: list[int] = [i for i in range(4)]


def handle_signal(signalnum, stack_frame):
    raise SystemExit()


# handle SIGTERM nicely
signal.signal(signal.SIGTERM, handle_signal)

# Load the model
model = YOLO("models/best_Reefscape_2025_model.engine")

# exit gracefully on ^C
is_interrupted: bool = False
print("Hello World")


def run_cam_in_thread(cameraname: int, file_index: int, q: Queue) -> None:
    video: cv2.VideoCapture = cv2.VideoCapture(cameraname)  # Read the video file
    # video.set(cv2.CAP_PROP_BUFFERSIZE, 0) # doesn't work :(

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = video.read()  # Read the video frames
        start_time: float = time.time()

        # exit if no frames remain
        if not ret:
            break

        if is_interrupted:
            break

        frame_res = cv2.resize(frame, (640, 640))

        # Empty the queue if it is full so the frame in it is the most recent one
        if q.full():
            # This should almost never happen, but it avoids any potential errors if it is emptied between calling full and get
            try:
                q.get_nowait()
            except Empty:
                pass
        try:
            q.put_nowait((frame_res, start_time))
        except Full:
            pass

    print(f"CAMERA {cameraname} EXITING (camera thread)")
    # Release video sources
    video.release()


def run_tracker_in_thread(cameraname: int, file_index: int, stream: Stream) -> None:
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

    q: Queue = Queue(maxsize=1)
    cam_thread = Thread(
        target=run_cam_in_thread, args=(cameraname, file_index, q), daemon=False
    )
    cam_thread.start()

    snapshot_time: float = time.time()

    print(f"Camera {cameraname} activating")

    # Exit the loop if no more frames in the video
    while not is_interrupted and cam_thread.is_alive():
        if is_interactive:
            print(f"Camera: {cameraname}")  # For debugging

        try:
            start_time: float
            frame: np.ndarray
            frame, start_time = q.get(block=True, timeout=5)
        except (
            Empty
        ):  # stop the thread getting stuck if the camera thread immidiately dies
            continue

        # start_time: float = time.time()

        # Track objects in frames if available
        results: list[Results] = model.predict(frame, verbose=is_interactive)
        # time.sleep(0.2)
        res_plotted: np.ndarray = results[0].plot()
        # Calculate offsets and add to NetworkTables
        ntables.add_results(results, start_time, file_index)

        end_time: float = time.time()

        if (
            results[0] is not None
            and len(results[0].boxes) != 0
            and len(results[0].boxes[0]) is not None
        ):
            xyxy = results[0].boxes.xyxy[0]
            orig_shape = results[0].boxes.orig_shape
            print("x: " + str(util.get_x_offset_deg_single(xyxy, orig_shape)))
            print("y: " + str(util.get_y_offset_deg_single(xyxy, orig_shape)))

        # if (time.time() - snapshot_time > 10): # snapshot every x seconds
        # snapshotter.submit(results[0])
        # snapshot_time = time.time()

        if enable_mjpeg:
            fps: float = round(1 / (end_time - start_time), 2)
            center: tuple(int, int) = (640 // 2, 480 // 2)
            size: int = 25
            cv2.line(
                res_plotted,
                (center[0] - size, center[1]),
                (center[0] + size, center[1]),
                (0, 128, 255),
                5,
            )
            cv2.line(
                res_plotted,
                (center[0], center[1] - size),
                (center[0], center[1] + size),
                (0, 128, 255),
                5,
            )
            cv2.putText(
                res_plotted,
                str(fps),
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (100, 255, 0),
                3,
                cv2.LINE_AA,
            )
            print("FPS: " + str(fps))

            stream.set_frame(res_plotted)

    cam_thread.join()
    print(f"CAMERA {cameraname} EXITING (detector thread)")


if enable_mjpeg:
    stream: Stream = Stream("Detectorator", size=(640, 480), quality=50, fps=10)
    server: MjpegServer = MjpegServer("0.0.0.0", 5090)
    server.add_stream(stream)
    server.start()
else:
    stream = None

threads: list[Thread] = []
for i in range(len(cameras)):
    # Create the thread
    # daemon=True makes it shut down if something goes wrong
    thread = Thread(
        target=run_tracker_in_thread, args=(cameras[i], i, stream), daemon=True
    )
    # Add to the array to use later
    threads.append(thread)
    # Start the thread
    thread.start()

snapshot_thread: Thread = Thread(target=snapshotter.run_snapshotter_thread, daemon=True)
snapshot_thread.start()

try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit) as e:
    print(COLOR_BOLD, "INTERRUPT RECIEVED -- EXITING", COLOR_RESET, sep="")
    is_interrupted = True

# Wait for the tracker threads to finish
for thread in threads:
    thread.join()

if enable_mjpeg:
    # Clean up
    server.stop()
