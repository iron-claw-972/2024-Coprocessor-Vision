#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import time
import pandas as pd
import util
import time
model = YOLO('yolov8n.pt')

video = cv2.VideoCapture(0)  # Read the video file

list_of_x_offsets = []
list_of_y_offsets = []
list_of_pixel_widths = []

start_time = time.time()

while True:
    ret, frame = video.read()  # Read the video frames

    if not ret:
        print(f"CAMERA {0} EXITING")
        break

    # Track objects in frames if available
    results = model.track(frame, persist=True)
    res_plotted = results[0].plot()

    #Calculate offsets and add to NetworkTables

    #put text code from: 
    cv2.imshow("Tracking_Stream", res_plotted)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    if key == ord("s"): 
        #https://github.com/ultralytics/ultralytics/issues/5150
        if(len(results[0])): 
            boxes = results[0].boxes.xyxy.tolist()
            cone = boxes[0]
            distance = input("What is the distance?: ")
            starting_time = time.time()

            while True: 
                x_offset_deg = util.get_x_offset_deg(cone)
                y_offset_deg = util.get_y_offset_deg(cone)
                pixel_width = util.get_pixel_width(cone)
                
                list_of_x_offsets.append(x_offset_deg)
                list_of_y_offsets.append(y_offset_deg)
                list_of_pixel_widths.append(pixel_width)

                quit_key = cv2.waitKey(1)
                if quit_key == ord("q"): 
                    #avg calculations from: 
                    avg_x_offset = sum(list_of_x_offsets)/len(list_of_y_offsets)
                    avg_y_offset = sum(list_of_y_offsets)/len(list_of_y_offsets)
                    avg_pixel_width = sum(list_of_pixel_widths)/len(list_of_pixel_widths)
                    break
                else: 
                    starting_time = time.time()

                
            #https://www.w3schools.com/python/python_file_write.asp
            csv = open("data.csv", "a")
            csv.write(f"{avg_x_offset},{avg_y_offset},{avg_pixel_width},{distance}\n")
            csv.close()

# Release video sources
video.release()

# Clean up and close windows
cv2.destroyAllWindows()