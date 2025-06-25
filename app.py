from flask import Flask, request, Response, render_template
import numpy as np
import cv2
from detection_face_recognition import detection_func  # Теперь тут вся логика
from image_segmentation import get_masks, apply_masks_to_frame
import torchvision.transforms as T
import torchvision
import torch
from threading import Thread, Lock
from image_segmentation import model as segmentation_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

app = Flask(__name__)

use_segmentation = False
segmentation_model.eval()
segmentation_model.to(device)
enable_detection = False
frame_lock = Lock()
latest_frame = None
active_detectors = []
allowed_classes = [0, 14, 15, 16]

detectors = {
    'Human Detector': None,
    'Cats Detector': None,
    'Dogs Detector': None,
    'Bird Detector': None,
}

@app.route("/", methods=["GET", "POST"])
def index():
    global use_segmentation, enable_detection, active_detectors, allowed_classes
    selected_detectors = []
    if request.method == "POST":
        selected_detectors = request.form.getlist("detectors")
        use_segmentation = "Segmentation" in selected_detectors
        active_detectors = [det for det in detectors if det in selected_detectors]
        enable_detection = len(active_detectors) > 0
        class_map = {
            'Human Detector': 0,
            'Bird Detector': 14,
            'Cats Detector': 15,
            'Dogs Detector': 16,
        }
        allowed_classes = [class_map[d] for d in active_detectors if d in class_map]
    options = list(detectors.keys()) + ['Segmentation']
    return render_template("index.html",
                           options=options,
                           selected_detectors=selected_detectors)

def capture_frames():
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (640, 480))
        with frame_lock:
            latest_frame = frame.copy()

capture_thread = Thread(target=capture_frames, daemon=True)
capture_thread.start()

def gen_frames():
    global latest_frame, enable_detection
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
        try:
            if enable_detection:
                detection_frame = frame.copy()
                frame = detection_func(detection_frame, allowed_classes)
            if use_segmentation:
                seg_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
                binary_masks, labels = get_masks(seg_frame)
                frame = apply_masks_to_frame(frame, binary_masks, labels)
        except Exception as e:
            print("Frame processing error:", e)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)