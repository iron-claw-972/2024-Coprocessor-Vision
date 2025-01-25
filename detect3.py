# load config
from asyncio import threads
import json

from detect import COLOR_BOLD, COLOR_RESET
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]



from roboflow import Roboflow
rf = Roboflow(api_key="kPLRt1EV7ib8qfAAjk9n")
project = rf.workspace("2025-reefscape-tv2dj").project("2025-reefscape-tv2dj/1")
version = project.version(1)
dataset = version.download("yolov11")

import cv2
import base64
import numpy as np
import requests
import time

# Construct the Roboflow Infer URL
# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
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

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw

    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    return image

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