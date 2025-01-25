import json
import cv2
import base64
import numpy as np
import requests
import time

# Load config
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Construct the Roboflow Infer URL
upload_url = "".join([
    "https://detect.roboflow.com/",
    ROBOFLOW_MODEL,
    "?api_key=",
    ROBOFLOW_API_KEY,
    "&format=image",
    "&stroke=5"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()
    if not ret:
        print("Error: Failed to capture frame.")
        return None

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    if not retval:
        print("Error: Failed to encode image.")
        return None
    img_str = base64.b64encode(buffer)

    # Send image to Roboflow API for inference
    try:
        response = requests.post(upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        response.raise_for_status()  # Check for HTTP errors

        # Parse result image from response
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Main loop; infers sequentially until you press "q"
frame_count = 0
start_time = time.time()

while True:
    # Capture start time to calculate fps
    frame_count += 1

    # On "q" keypress, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()

    if image is not None:
        # Display the inference results
        cv2.imshow('image', image)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0
            print(f"FPS: {fps:.2f}")

# Release resources when finished
video.release()
cv2.destroyAllWindows()
