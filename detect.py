# Based off of: https://github.com/jveitchmichaelis/edgetpu-yolo/blob/main/detect.py

"""
Instalation for librarys: ( can skip steps like pytorch if already done)
sudo apt-get update && sudo apt-get -y upgrade
sudo apt-get install -y git curl gnupg

# Install PyCoral (you don't need to do this on a Coral Board)
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
sudo apt-get update
sudo apt-get install -y gasket-dkms libedgetpu1-std python3-pycoral

# Get Python dependencies
sudo apt-get install -y python3 python3-pip
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy
python3 -m pip install opencv-python-headless
python3 -m pip install tqdm pyyaml

# Clone this repository
git clone https://github.com/jveitchmichaelis/edgetpu-yolo
cd edgetpu-yolo

CHANGE python3 TO BE python3.9 AS PYTORCH REQUIRES IT
"""

from edgetpumodel import  EdgeTPUModel
from utils import get_image_tensor

import numpy as np
import cv2

# get model and labels
model_name = input("Model File (Weights in Tflite):")
names_yaml = input("Path to <names>.yaml file:")

# setup edge tpu model
model = EdgeTPUModel(model_name, names_yaml, conf_thresh=0.25, iou_thresh=0.45)
input_size = model.get_image_size()

# run with random data (dont know why, included in original repo code)
x = (255*np.random.random((3,*input_size))).astype(np.uint8)
model.forward(x)

# setup cv2 camera
cam = cv2.VideoCapture(0)
        
while True:
    res, image = cam.read()
    
    if res is False:
        # failed to capture
        break
    
    # format image and get base predictions
    full_image, net_image, pad = get_image_tensor(image, input_size[0])
    pred = model.forward(net_image)
    
    # format preditctions and plot onto full_image
    model.process_predictions(pred[0], full_image, pad)
    
    # display image with annotation
    cv2.imshow("Frame", full_image)
    
    # plot the time it took to process the predictions (dosen't include cv2.imshow time)
    # GOAL: 50 FPS (to match with robot periodic)
    tinference, tnms = model.get_last_inference_time()
    print(f"Frame done in {tinference+tnms}")
        
# cv2 closing process
