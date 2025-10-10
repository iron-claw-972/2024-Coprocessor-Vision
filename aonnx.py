# export_onnx.py
from ultralytics import YOLO
model = YOLO('models/best_Reefscape_2025_model.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print("Done: check model.onnx in current directory")
