from ultralytics import YOLO
model = YOLO("models/best_Reefscape_2025_model.pt")
model.export(format="engine", half=True, nms=True)
# half=True -> FP16; nms=True tries to include NMS in the engine (faster)
print("Done: check model.onnx in current directory")

