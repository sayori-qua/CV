<div align="center">
  <h3>Written by @sayori_qua</h3>
</div>

# 🖼️ Real-Time Object Detection with Face & Animal Recognition
This project is a real-time object detection and classification system that uses webcams to detect people, cats, dogs, and birds while also recognizing facial attributes like gender and age , and applying semantic segmentation to identify various objects in the scene.

**It combines powerful models from:**
- YOLOv8 for object detection
- DeepFace for face analysis
- Hugging Face Transformers for breed/species classification
- Mask R-CNN for image segmentation

**🧠 Features**
🔍 Detects and tracks multiple objects: humans, dogs, cats, and birds.
👥 For humans:
- Recognizes gender
- Predicts age
- Estimates height
- 
🐶🐱 Classifies animal breeds:
- Dogs (via ViT)
- Cats (via custom model)
- Birds (via species classifier)
Semantic segmentation overlay using Mask R-CNN
🎮 Toggle between detectors via a Flask-based web interface

**📦 Requirements**
Ensure you have the following dependencies installed:

```bash 
pip install torch torchvision opencv-python deepface flask ultralytics pillow transformers
```

Make sure you're using a compatible version of CUDA if running on GPU.

**🚀 How to Run**
1. Start the Flask server:
```bash
python app.py
```
3. Open your browser and go to:
```bash
http://localhost:5000
```
5. Select desired detectors (e.g., Human Detector, Cats Detector, Segmentation).
6. View real-time results from your webcam stream!

**📸 Webcam Stream Preview**
The live video feed shows bounding boxes and labels around detected objects:

- Humans are labeled with ID, gender, age, and estimated height
- Animals show breed or species prediction with confidence score
- Segmented objects are color-coded with class names displayed
  
**📝 Notes**
- The first time you run it, some models will be downloaded automatically — this may take some time.
- Make sure your camera is accessible at /dev/video0 (Linux) or as default device (Windows/Mac).
- You can modify allowed classes and models inside detection_face_recognition.py.
  
**💡 Possible Improvements**
- Add support for more animal species
- Implement caching for faster inference
- Save logs or video recordings
- Add configuration file for easier customization
