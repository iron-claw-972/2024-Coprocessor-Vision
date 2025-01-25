import cv2
import time
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow with your API key
rf = Roboflow(api_key="kPLRt1EV7ib8qfAAjk9n")
project = rf.workspace().project("2025-reefscape-tv2dj")
model = project.version("2").model

# Initialize webcam
video = cv2.VideoCapture(0)  # Use 0 for the default webcam
if not video.isOpened():
    raise RuntimeError("Could not access the webcam. Please check your device.")

# Resize the window to make it bigger
cv2.namedWindow('Webcam Detection', cv2.WINDOW_NORMAL)  # Resizable window
cv2.resizeWindow('Webcam Detection', 800, 600)  # Set initial window size

# Function to predict using the Roboflow API
def infer(frame):
    try:
        # Perform inference on the current frame
        prediction = model.predict(frame, confidence=30, overlap=30).json()  # Reduce confidence threshold for speed

        # Draw the prediction results on the frame
        for item in prediction["predictions"]:
            x1, y1, x2, y2 = item["x"] - item["width"] / 2, item["y"] - item["height"] / 2, item["x"] + item["width"] / 2, item["y"] + item["height"] / 2
            label = item["class"]
            confidence = item["confidence"]

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence*100:.1f}%", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    except Exception as e:
        print(f"Inference error: {e}")
        return frame

# Main loop to capture frames and display inference results
start_time = time.time()  # Start time for FPS calculation
frame_count = 0
frame_skip = 2  # Skip frames to improve FPS

try:
    while True:
        # Capture a frame from the webcam
        ret, frame = video.read()
        if not ret or frame is None:
            print("Failed to capture frame from webcam.")
            break

        # Skip frames (to achieve higher FPS)
        if frame_count % frame_skip == 0:
            # Perform inference on the frame
            result_frame = infer(frame)

            # Increment the frame count
            frame_count += 1

            # Calculate FPS (frames per second)
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0

            # Display the FPS on the frame
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame with inference results
            cv2.imshow('Webcam Detection', result_frame)

        else:
            frame_count += 1  # Increment frame count but skip processing

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break
finally:
    # Release resources when finished
    video.release()
    cv2.destroyAllWindows()
