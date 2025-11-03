#!/usr/bin/env python3
"""
detect.py — optimized for Jetson Orin Nano + TensorRT engine + NVMM camera + MJPEG streamer.
Drop-in replacement: set paths & names as needed and run on the Jetson.
"""

import cv2
import time
import threading
from queue import Queue, Empty, Full
import numpy as np
from ultralytics import YOLO
from flask import Flask, Response

# === USER / ENV CONFIG ===
ENGINE_PATH = "models/best_Reefscape_2025_model.engine"   # your built engine
MODEL_NAMES = ["cone", "cube"]                            # restore your class names here
IMG_SZ = 640                                             # model input size (tune if needed)
USE_NVARGUS = True                                        # set False for USB/v4l2 Arducam
CAM_SENSOR_ID = 0                                         # nvarguscamerasrc sensor-id (if CSI)
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 60
HTTP_PORT = 5090
HTTP_ROUTE = "/detectorator"

# === Globals ===
frame_q = Queue(maxsize=1)     # only keep newest frame to avoid backlog
display_frame = None           # frame used by MJPEG streamer
stop_event = threading.Event()

# === Helper: Build GStreamer pipeline for NVMM (Jetson) or fallback v4l2 ===
def make_gst_pipeline(width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS, sensor_id=CAM_SENSOR_ID, use_nvargus=True):
    if use_nvargus:
        # NVMM path -> zero-copy on Jetson (preferred)
        gst = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
            "nvvidconv ! video/x-raw, format=(string)BGRx ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        )
    else:
        # USB/camera via v4l2 (may be slower; adjust device if needed)
        gst = (
            f"v4l2src device=/dev/video0 ! video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
            "videoconvert ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        )
    return gst

# === Capture thread: fills frame_q with (frame, timestamp) and drops old frames ===
def capture_loop(cap, q):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            # small sleep to avoid tight loop if capture fails
            time.sleep(0.005)
            continue
        ts = time.time()
        # keep only the newest frame: if queue full, remove oldest
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait((frame, ts))
        except Full:
            # should not happen due to get_nowait, but safe-guard
            pass

# === Lightweight detector drawing helper (works with typical Ultralytics results) ===
def extract_detections(results):
    """
    Attempt to extract boxes as (x1,y1,x2,y2,class_idx,conf) list from ultralytics Results.
    Returns empty list on failure.
    """
    dets = []
    if results is None or len(results) == 0:
        return dets
    try:
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            return dets

        # try several access patterns (works for cpu/tensor/engine outputs)
        try:
            xy = boxes.xyxy.cpu().numpy()
        except Exception:
            try:
                xy = boxes.xyxy.numpy()
            except Exception:
                try:
                    xy = np.array(boxes.xyxy)   # fallback
                except Exception:
                    xy = None

        if xy is None:
            return dets

        # class indices
        try:
            cls = boxes.cls.cpu().numpy()
        except Exception:
            try:
                cls = boxes.cls.numpy()
            except Exception:
                # try to read per-box attr
                cls = [int(getattr(b, "cls", 0)) for b in boxes]

        # confidences
        try:
            conf = boxes.conf.cpu().numpy()
        except Exception:
            try:
                conf = boxes.conf.numpy()
            except Exception:
                conf = [1.0] * len(xy)

        for i, b in enumerate(xy):
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            c = int(cls[i]) if isinstance(cls, (list, np.ndarray)) else int(cls)
            cv = float(conf[i]) if isinstance(conf, (list, np.ndarray)) else float(conf)
            dets.append((x1, y1, x2, y2, c, cv))
    except Exception as e:
        # graceful fallback: no detections parsed
        # print("extract_detections parse error:", e)
        return []
    return dets

def draw_detections(frame, dets, names):
    for (x1, y1, x2, y2, ci, conf) in dets:
        label = names[ci] if ci < len(names) else f"class{ci}"
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# === Inference loop: gets newest frame and runs model.track/predict ===
def inference_loop(q, model, names, imgsz=IMG_SZ):
    global display_frame
    last_report = time.time()
    infer_count = 0
    infer_accum = 0.0

    while not stop_event.is_set():
        try:
            frame, ts = q.get(timeout=0.5)
        except Empty:
            continue

        # inference start
        t0 = time.time()
        try:
            # prefer track (keeps object IDs across frames) — fall back to predict if track not supported
            results = model.track(frame, imgsz=imgsz, device=0, persist=True)
        except Exception:
            results = model.predict(frame, imgsz=imgsz, device=0)

        t1 = time.time()
        infer_time = t1 - t0
        infer_accum += infer_time
        infer_count += 1

        # parse boxes and draw
        dets = extract_detections(results)
        drawn = frame.copy()
        draw_detections(drawn, dets, names)

        # annotate with timing
        total_latency = time.time() - ts
        cv2.putText(drawn, f"latency {total_latency*1000:.0f} ms | infer {infer_time*1000:.0f} ms",
                    (10, drawn.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # publish to MJPEG display frame (atomic swap)
        display_frame = drawn

        # FPS & instrumentation print every 1s
        if time.time() - last_report >= 1.0:
            avg_infer = (infer_accum / infer_count) if infer_count else 0.0
            print(f"[INFO] infer_fps ~ {1.0/avg_infer:.1f} | avg_infer={avg_infer*1000:.1f} ms | queue_size={q.qsize()}")
            infer_count = 0
            infer_accum = 0.0
            last_report = time.time()

# === MJPEG streaming via Flask ===
app = Flask(__name__)

def mjpeg_generator():
    global display_frame
    # simple generator for MJPEG
    while not stop_event.is_set():
        if display_frame is None:
            time.sleep(0.01)
            continue
        # encode
        ret, jpg = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue
        frame_bytes = jpg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    # when stopping, yield nothing

@app.route(HTTP_ROUTE)
def stream_route():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === Main ===
def main():
    print("[INFO] Loading model (TensorRT engine)...")
    model = YOLO(ENGINE_PATH, task="detect")
    # restore names
    model.names = MODEL_NAMES
    print(f"[INFO] Model loaded. names={model.names}")

    # build gst pipeline and open capture
    gst = make_gst_pipeline(use_nvargus=USE_NVARGUS)
    print(f"[INFO] Opening camera with pipeline:\n{gst[:200]}...")  # show partial for brevity
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Failed to open camera with GStreamer pipeline. Try use_nvargus=False or check OpenCV build.")
        return

    # start Flask server in its own thread
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=HTTP_PORT, threaded=True, use_reloader=False), daemon=True)
    flask_thread.start()
    print(f"[INFO] MJPEG stream available at: http://<jetson-ip>:{HTTP_PORT}{HTTP_ROUTE}")

    # start capture and inference threads
    cap_thread = threading.Thread(target=capture_loop, args=(cap, frame_q), daemon=True)
    inf_thread = threading.Thread(target=inference_loop, args=(frame_q, model, MODEL_NAMES, IMG_SZ), daemon=True)
    cap_thread.start()
    inf_thread.start()

    print("[INFO] Running. Press Ctrl+C to stop. Use tegrastats to watch GPU utilization.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        stop_event.set()
        cap.release()
        time.sleep(0.5)

if __name__ == "__main__":
    main()
