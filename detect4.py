import cv2
import base64
import numpy as np
import requests
import json
import time
from roboflow import Roboflow

# Load Roboflow configuration

ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
ROBOFLOW_SIZE = int(config["ROBOFLOW_SIZE"])