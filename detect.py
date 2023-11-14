#!/usr/bin/env python3.9

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
python3.9 -m pip install --upgrade pip setuptools wheel
python3 -m pip install numpy
python3 -m pip install opencv-python-headless
python3.9 -m pip install tqdm pyyaml

# Clone this repository
git clone https://github.com/jveitchmichaelis/edgetpu-yolo
cd edgetpu-yolo

CHANGE python3 TO BE python3.9 AS PYTORCH REQUIRES IT
"""

from edgetpumodel import  EdgeTPUModel
from utils import get_image_tensor
import numpy as np
import cv2
import ntables
from pathlib import Path
import time
import multiprocessing

def process1():
	# get model and labels 	
	cap = cv2.VideoCapture(0)
	while True:

		start_time = time.time() 

		res, image = cap.read()
    	
		if res is False:
        	# failed to capture
			break

    	# format image and get base predictions
		full_image, net_image, pad = get_image_tensor(image, input_size[0])

		pred = model.forward(net_image)
    	
    	# format preditctions and plot onto full_image
		model.process_predictions(pred[0], image, pad)
		    	
    	# display image with annotation
		end_time = time.time()
		fps = str(int(1/(end_time-start_time)))

		cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
		cv2.imshow("FRAME",image)
    	
		if(cv2.waitKey(1) & 0xFF == ord("q")): 
			break
	
    	# plot the time it took to process the predictions (dosen't include cv2.imshow time)
    	# GOAL: 50 FPS (to match with robot periodic)
		tinference, tnms = model.get_last_inference_time()


		#y_offset_deg_nt = model.get_y_offset_deg(pred[0])

		#ntables.publish_distance(distance_nt)
		#ntables.publish_y_offset_deg(y_offset_deg_nt)

		if(len(pred[0])): #check if prediction array is empty or not 
			distances = []
			for prediction in pred: 
				distance = model.get_distance(prediction,actual_obj_width_meters=0.059,actual_obj_dist_meters=0.15)
				distances.append(distance)
			
			pred_index_min_dist = distances.index(min(distances))

			distance_nt = model.get_distance(pred[pred_index_min_dist],actual_obj_width_meters=0.059,actual_obj_dist_meters=6.04)
			x_offset_deg_nt = model.get_x_offset_deg(pred[pred_index_min_dist])
			y_offset_deg_nt = model.get_y_offset_deg(pred[pred_index_min_dist])

			ntables.publish_distance(distance_nt)
			ntables.publish_x_angle_offset(x_offset_deg_nt)
			ntables.publish_y_angle_offset(y_offset_deg_nt)

    # cv2 closing process
	cap.release()
	cv2.destroyAllWindows()

def process2():
	# get model and labels 	
	cap = cv2.VideoCapture(2)
	while True:

		start_time = time.time() 

		res, image = cap.read()
    	
		if res is False:
        	# failed to capture
			break

    	# format image and get base predictions
		full_image, net_image, pad = get_image_tensor(image, input_size[0])

		pred = model.forward(net_image)
    	
    	# format preditctions and plot onto full_image
		model.process_predictions(pred[0], image, pad)
		    	
    	# display image with annotation
		end_time = time.time()
		fps = str(int(1/(end_time-start_time)))

		cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
		cv2.imshow("FRAME",image)
    	
		if(cv2.waitKey(1) & 0xFF == ord("q")): 
			break
	
    	# plot the time it took to process the predictions (dosen't include cv2.imshow time)
    	# GOAL: 50 FPS (to match with robot periodic)
		tinference, tnms = model.get_last_inference_time()


		#y_offset_deg_nt = model.get_y_offset_deg(pred[0])

		#ntables.publish_distance(distance_nt)
		#ntables.publish_y_offset_deg(y_offset_deg_nt)

		if(len(pred[0])): #check if prediction array is empty or not 
			distances = []
			for prediction in pred: 
				distance = model.get_distance(prediction,actual_obj_width_meters=0.059,actual_obj_dist_meters=0.15)
				distances.append(distance)
			
			pred_index_min_dist = distances.index(min(distances))

			distance_nt = model.get_distance(pred[pred_index_min_dist],actual_obj_width_meters=0.059,actual_obj_dist_meters=6.04)
			x_offset_deg_nt = model.get_x_offset_deg(pred[pred_index_min_dist])
			y_offset_deg_nt = model.get_y_offset_deg(pred[pred_index_min_dist])

			ntables.publish_distance(distance_nt)
			ntables.publish_x_angle_offset(x_offset_deg_nt)
			ntables.publish_y_angle_offset(y_offset_deg_nt)

    # cv2 closing process
	cap.release()
	cv2.destroyAllWindows()
	
if __name__ == "__main__": 
	model_name = "yolov5s-int8-224_edgetpu.tflite"#input("Model File (Weights in Tflite):")
	names_yaml = "data/coco.yaml"#input("Path to <names>.yaml file:")
	
	# setup edge tpu model
	model = EdgeTPUModel(model_name, names_yaml, conf_thresh=0.7, iou_thresh=0.45)
	input_size = model.get_image_size()
	
	# run with random data (dont know why, included in original repo code)
	x = (255*np.random.random((3,*input_size))).astype(np.uint8)
	model.forward(x)
	
	# setup cv2 camerayolov5s-int8-96_edgetpu.tflite
	process1()
	process2()
