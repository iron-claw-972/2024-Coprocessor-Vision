import threading
import cv2
from ultralytics import YOLO
import time
import ntables

# Define the video files for the trackers
# Path to video files, 0 for webcam, 1 for external camera
# On a robot, these should be in the same order as the cameras in VisionConstants.java
video_files = [0, 1]

def run_tracker_in_thread(filename, model, file_index):
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
        print(f"Camera: {filename}") # For debugging 
        ret, frame = video.read()  # Read the video frames
        start_time = time.time()
        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        # Calculate offsets and add to NetworkTables
        ntables.add_results(results, file_index)
        end_time = time.time()

        fps = str(int(1/(end_time-start_time)))
        cv2.putText(res_plotted, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 

        cv2.imshow(f"Tracking_Stream_{filename}", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()

# Load the model
model = YOLO('yolov8n.pt')
threads = []
for i in range(len(video_files)):
    # Create the thread
    thread = threading.Thread(target=run_tracker_in_thread, args=(video_files[i], model, i), daemon=True)
    # Add to the array to use later
    threads += thread
    # Start the thread
    thread.start()

# Wait for the tracker threads to finish
for thread in threads:
    thread.join()

# Clean up and close windows
cv2.destroyAllWindows()