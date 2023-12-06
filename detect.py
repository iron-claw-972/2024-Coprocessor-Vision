#!/usr/bin/env python3
import threading
import cv2
from ultralytics import YOLO
import time

lock = threading.Lock()

def run_tracker_in_thread(filename, model, file_index, lock):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """
    video = cv2.VideoCapture(filename)  # Read the video file

    while True:
        lock.acquire()
        print(f"Camera: {filename}") # For debugging 
        ret, frame = video.read()  # Read the video frames
        start_time = time.time()
        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        end_time = time.time()

        fps = str(int(1/(end_time-start_time)))
        cv2.putText(res_plotted, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 

        cv2.imshow(f"Tracking_Stream_{filename}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        lock.release()

    # Release video sources
    video.release()


# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n.pt')

# Define the video files for the trackers
video_file1 = 0  # Path to video file, 0 for webcam
video_file2 = 1  # Path to video file, 0 for webcam, 1 for external camera

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1, lock), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2, lock), daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
