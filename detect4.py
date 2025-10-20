import cv2
import base64
import numpy as np
import requests
import json
import time
from roboflow import Roboflow

# Load Roboflow configuration
try:
    with open('roboflow_config.json') as f:
        config = json.load(f)
    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = int(config["ROBOFLOW_SIZE"])
except FileNotFoundError:
    print("roboflow_config.json not found. Please create the configuration file.")
    exit(1)
except KeyError as e:
    print(f"Missing key in configuration: {e}")
    exit(1)

# Test API connection
test_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}?api_key={ROBOFLOW_API_KEY}"
try:
    response = requests.get(test_url)
    print(f"API Test Response: {response.status_code}")
    print(f"Response content: {response.text}")
except Exception as e:
    print(f"API Test Error: {e}")



# Initialize Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(ROBOFLOW_MODEL.split("/")[0])
model = project.version(int(ROBOFLOW_MODEL.split("/")[1])).model

# Initialize webcam
video = cv2.VideoCapture(0)


# Function to predict using the Roboflow API
def infer(frame):
    try:
        # Perform inference on the current frame
        prediction = model.predict(frame, confidence=30, overlap=30).json()  # Reduce confidence threshold for speed

        # Draw the prediction results on the frame
        for item in prediction["predictions"]:
            x1, y1, x2, y2 = item["x"] - item[""] / 2, item["y"] - item[""] / 2, item["x"] + item[""] / 2, item["y"] + item[""] / 2
            label = item["class"]
            confidence = item["confidence"]

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence*100:.1f}%", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    except Exception as e:
        print(f"Inference error: {e}")
        return frame


# Main loop
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    start_time = time.time()
    
    processed_image = infer(frame)
    
    # Calculate and display FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(processed_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Roboflow Inference', processed_image)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

