#!/usr/bin/env python3
from threading import Thread
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from ultralytics.engine.results import Results # type: ignore
import time
import ntables
import snapshotter
import signal
import sys
from mjpeg_streamer import MjpegServer, Stream
from queue import Empty, Queue, Full

# ANSI colors
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"

# are we running interactively?
is_interactive: bool = sys.stderr.isatty()

# disable when not needed to improve performance
enable_mjpeg: bool = is_interactive

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
model = YOLO('models/best_Reefscape_2025_model.pt')

# exit gracefully on ^C
is_interrupted: bool = False
print("Hello World")

def run_cam_in_thread(cameraname: int, file_index: int, q: Queue) -> None:
    video: cv2.VideoCapture = cv2.VideoCapture(cameraname)  # Read the video file
    #video.set(cv2.CAP_PROP_BUFFERSIZE, 0) # doesn't work :(

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = video.read()  # Read the video frames

        # exit if no frames remain
        if not ret:
            break
    
        if is_interrupted:
            break

        # Empty the queue if it is full so the frame in it is the most recent one
        if q.full():
            # This should almost never happen, but it avoids any potential errors if it is emptied between calling full and get
            try:
                q.get_nowait()
            except Empty:
                pass
        try:
            q.put_nowait(frame.copy())
        except Full:
            pass

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
    cam_thread = Thread(target=run_cam_in_thread, args=(cameraname, file_index, q), daemon=False)
    cam_thread.start()

    snapshot_time: float = time.time()

    print(f"Camera {cameraname} activating")

    while True:
        if (is_interactive):
            print(f"Camera: {cameraname}") # For debugging 

        # Exit the loop if no more frames in either video
        if is_interrupted or (not cam_thread.is_alive()):
            print(f"CAMERA {cameraname} EXITING")
            break

        start_time: float = time.time()

        frame: np.ndarray = q.get(block=True)

        # Track objects in frames if available
        results: list[Results] = model.track(frame, persist=True, verbose=is_interactive)
        res_plotted: np.ndarray = results[0].plot()
        # Calculate offsets and add to NetworkTables
        ntables.add_results(results, file_index)
        end_time: float = time.time()

        if (time.time() - snapshot_time > 1): # snapshot every x seconds
            snapshotter.submit(frame.copy(), results[0].new())
            snapshot_time = time.time()

        if (enable_mjpeg):
            fps: float = round(1/(end_time-start_time), 2)
            cv2.putText(res_plotted, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 

            stream.set_frame(res_plotted)


if (enable_mjpeg):
    stream: Stream = Stream("Detectorator", size=(640, 480), quality=50, fps=10)
    server: MjpegServer = MjpegServer("localhost", 8080)
    server.add_stream(stream)
    server.start()
else:
    stream = None

threads: list[Thread] = []
for i in range(len(cameras)):
    # Create the thread
    # daemon=True makes it shut down if something goes wrong
    thread = Thread(target=run_tracker_in_thread, args=(cameras[i], i, stream), daemon=True)
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

if (enable_mjpeg):
    # Clean up and close windows
    server.stop()
    cv2.destroyAllWindows()

