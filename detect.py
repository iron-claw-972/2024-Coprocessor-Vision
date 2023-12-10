#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import time
import pandas as pd

model = YOLO('yolov8n.pt')

video = cv2.VideoCapture(0)  # Read the video file

while True:
    ret, frame = video.read()  # Read the video frames
    start_time = time.time()

    if not ret:
        print(f"CAMERA {0} EXITING")
        break

    # Track objects in frames if available
    results = model.track(frame, persist=True)
    res_plotted = results[0].plot()
    # Calculate offsets and add to NetworkTables
    end_time = time.time()

    fps = str(int(1/(end_time-start_time)))
    cv2.putText(res_plotted, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
    cv2.imshow("result", res_plotted)
# Release video sources
video.release()

# Clean up and close windows
cv2.destroyAllWindows()