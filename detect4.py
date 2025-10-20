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

def infer(frame):
    try:
        # Convert frame to bytes
        retval, buffer = cv2.imencode('.jpg', frame)
        img_bytes = buffer.tobytes()
        
        # Predict using Roboflow SDK
        result = model.predict(img_bytes, confidence=40, overlap=30).json()
        print(f"Prediction result: {result}")
        
        # Draw bounding boxes on the frame
        for prediction in result['predictions']:
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{prediction['class']} {prediction['confidence']:.2f}", 
                        (int(x-w/2), int(y-h/2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return frame
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return frame