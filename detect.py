#!/usr/bin/env python3
from threading import Thread
import cv2
import numpy as np
from ultralytics import YOLO # type: ignore
from ultralytics.engine.results import Results # type: ignore
import time
import ntables
import signal

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
model = YOLO('models/best.pt')

# exit gracefully on ^C
is_interrupted: bool = False


def run_tracker_in_thread(cameraname: int, file_index: int) -> None:
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
    video: cv2.VideoCapture = cv2.VideoCapture(cameraname)  # Read the video file
    #video.set(cv2.CAP_PROP_BUFFERSIZE, 0) # doesn't work :(

    while True:
        print(f"Camera: {cameraname}") # For debugging 
        
        while True:
            before_read: float = time.time()
            ret: bool
            frame: np.ndarray
            ret, frame = video.read()  # Read the video frames
            after_read: float = time.time()

            # exit if no frames remain
            if not ret:
                break

            # if the frame is read too quickly, it's probably from the buffer
            if after_read - before_read > 1/20 / 10: # assume 20fps and take a tenth of that
                break

        # Exit the loop if no more frames in either video
        if (not ret) or is_interrupted:
            print(f"CAMERA {cameraname} EXITING")
            break

        start_time: float = time.time()

        # Track objects in frames if available
        results: list[Results] = model.track(frame, persist=True)
        res_plotted: np.ndarray = results[0].plot()
        # Calculate offsets and add to NetworkTables
        ntables.add_results(results[0], file_index)
        end_time: float = time.time()

        fps = str(round(1/(end_time-start_time), 2))
        cv2.putText(res_plotted, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 

        cv2.imshow(f"Tracking_Stream_{cameraname}", res_plotted)

        key = cv2.waitKey(1)

    # Release video sources
    video.release()

threads: list[Thread] = []
for i in range(len(cameras)):
    # Create the thread
    # daemon=True makes it shut down if something goes wrong
    thread = Thread(target=run_tracker_in_thread, args=(cameras[i], i), daemon=True)
    # Add to the array to use later
    threads.append(thread)
    # Start the thread
    thread.start()

try:
    while True:
        time.sleep(1)
except (KeyboardInterrupt, SystemExit) as e:
    print(COLOR_BOLD, "INTERRUPT RECIEVED -- EXITING", COLOR_RESET, sep="")
    is_interrupted = True

# Wait for the tracker threads to finish
for thread in threads:
    thread.join()

# Clean up and close windows
cv2.destroyAllWindows()

