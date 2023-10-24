# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a program to run a retrained detector model
To run type: python examples/detect_image_webcam.py --model test_data/cone_cube_detector_v4_edgetpu.tflite --labels test_data/cone_cube_labels.txt
ADAPTED FROM https://github.com/google-coral/examples-camera/tree/master/opencv/detect.py AND https://github.com/google-coral/pycoral/tree/master/examples/detect_image.py
"""

import argparse
import time

from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

import cv2

def main():
  #create argument parser class instance
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  
  #add model argument(allows us to specify what model to use) 
  parser.add_argument('-m', '--model', required=True,
                      help='File path of .tflite file')
  
  #add labels argument(allows us to specify what labels.txt file to use) 
  parser.add_argument('-l', '--labels', help='File path of labels file')

  parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
  
  #add camera_id argument(allows us to specify which camera to use)
  parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)

  #add threshold argument(allows us to only display objects that are detected with a confidence level of 75% or above)
  parser.add_argument('--threshold', type=float, default=0.7,
                        help='detector confidence threshold')
  
  args = parser.parse_args()

  #read labels.txt file and turn it into dictionary format({id:label})
  labels = read_label_file(args.labels) if args.labels else {}

  #create interpreter(used to)
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  #get the input image size that the model uses
  inference_size = input_size(interpreter)

  #create a video capture object to access USB camera or webcam  
  cap = cv2.VideoCapture(args.camera_idx)

  previous_frame_timestamp = 0; 
  current_frame_timestamp = 0; 

  font = cv2.FONT_HERSHEY_SIMPLEX 

  while cap.isOpened():
      
      #get frames from the video capture 
      ret, frame = cap.read()

      if not ret:
          break
      
      cv2_im = frame

      cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

      #resize input frames to the size that the model uses
      cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)

      run_inference(interpreter, cv2_im_rgb.tobytes())

      objs = detect.get_objects(interpreter, args.threshold)[:args.top_k]
      cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)

      current_frame_timestamp = time.time()
      fps = 1/(current_frame_timestamp-previous_frame_timestamp) 
      fps = str(int(fps))

      cv2.putText(cv2_im, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
      
      cv2.imshow('frame', cv2_im)
      previous_frame_timestamp = time.time()

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
      


  cap.release()
  cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]

    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
  main()
