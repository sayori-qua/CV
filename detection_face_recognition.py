from ultralytics import YOLO
from deepface import DeepFace
import cv2
from tensorflow.keras.models import load_model
from transformers import pipeline
from transformers import ViTImageProcessor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

processor = ViTImageProcessor.from_pretrained("skyau/dog-breed-classifier-vit", use_fast=True)
model = YOLO("yolov8m.pt").to(device)
pipe_for_dogs = pipeline("image-classification", model="skyau/dog-breed-classifier-vit", device=0)
pipe_for_cats = pipeline("image-classification", model="dima806/67_cat_breeds_image_detection", device=0)
pipe_for_birds = pipeline("image-classification", model="chriamue/bird-species-classifier", device=0)
pipe_for_face = pipeline("image-classification", model="dima806/fairface_age_image_detection", device=0)

frame_counter = 0
last_gender = {}
gender_confirmed = {}
classes = [0,14,15,16]

def detection_func(frame, allowed_classes=None):
    if allowed_classes is None:
        allowed_classes = [0, 14, 15, 16]
    global frame_counter, last_gender, gender_confirmed
    results = model.track(frame, persist=True, iou=0.5, conf=0.4)
    for result in results:
        for box in result.boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            track_id = int(box.id)
            obj_label = model.names[cls]
            if cls not in allowed_classes:
                continue
            if cls == 0:
                face_img = frame[y1:int(y2), x1:x2]
                if face_img.size == 0:
                    cv2.putText(frame, f"Person ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    continue
                try:
                    face_resized = cv2.resize(face_img, (224, 224))
                except:
                    cv2.putText(frame, f"Person ID:{track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    continue
                if track_id not in last_gender:
                    last_gender[track_id] = {"Man": 0, "Woman": 0}
                    gender_confirmed[track_id] = False
                if not gender_confirmed[track_id] or frame_counter % 10 == 0:
                    try:
                        result = DeepFace.analyze(face_resized, actions=['gender'], enforce_detection=False)
                        last_gender[track_id] = result[0]['gender']
                        gender_confirmed[track_id] = True
                    except Exception as e:
                        print(f"Ошибка анализа пола для ID {track_id}:", e)
                        gender_confirmed[track_id] = False
                if gender_confirmed[track_id]:
                    man_score = last_gender[track_id].get('Man', 0)
                    woman_score = last_gender[track_id].get('Woman', 0)
                    gender = 'Man' if man_score > woman_score else 'Woman'
                else:
                    gender = "Unknown"
                face_cropped = cv2.resize(face_img, (224, 224))
                face_cropped = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)
                pil_image_face = Image.fromarray(face_cropped)
                predictions = pipe_for_face(pil_image_face)
                if predictions:
                    top_prediction = predictions[0]
                    age = top_prediction['label']
                    average_height = 170
                    pixels = y2 - y1
                    final_height = int((pixels / frame.shape[0]) * average_height)
                    gender_label = f"{gender} ID:{track_id} Age:{age} Height {final_height} cm"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, gender_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    average_height = 170
                    pixels = y2 - y1
                    final_height = int((pixels / frame.shape[0]) * average_height)
                    gender_label = f"{gender} ID:{track_id} Age: Unknown Height {final_height} cm"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, gender_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif cls == 14:
                bird_img = frame[y1:y2, x1:x2]
                bird_cropped = cv2.resize(bird_img, (224, 224))
                bird_cropped = cv2.cvtColor(bird_cropped, cv2.COLOR_BGR2RGB)
                pil_image_birds = Image.fromarray(bird_cropped)
                predictions = pipe_for_birds(pil_image_birds)
                if predictions:
                    top_prediction = predictions[0]
                    breed_label = top_prediction['label']
                    confidence = top_prediction['score']
                    label = f"Bird species: {breed_label}, Confidence: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    cv2.putText(frame, "Bird detected (no species)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif cls == 15:
                cat_img = frame[y1:y2, x1:x2]
                cat_cropped = cv2.resize(cat_img, (224, 224))
                cat_cropped = cv2.cvtColor(cat_cropped, cv2.COLOR_BGR2RGB)
                pil_image_cats = Image.fromarray(cat_cropped)
                predictions = pipe_for_cats(pil_image_cats)
                if predictions:
                    top_prediction = predictions[0]
                    breed_label = top_prediction['label']
                    confidence = top_prediction['score']
                    label = f"Cat Breed: {breed_label}, Confidence: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, "Cat detected (no breed)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif cls == 16:
                dog_img = frame[y1:y2, x1:x2]
                dog_cropped = cv2.resize(dog_img, (224, 224))
                dog_cropped = cv2.cvtColor(dog_cropped, cv2.COLOR_BGR2RGB)
                pil_image_dogs = Image.fromarray(dog_cropped)
                predictions = pipe_for_dogs(pil_image_dogs)
                if predictions:
                    top_prediction = predictions[0]
                    breed_label = top_prediction['label']
                    confidence = top_prediction['score']
                    label = f"Dog Breed: {breed_label}, Confidence: {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    cv2.putText(frame, "Dog detected (no breed)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    frame_counter += 1
    return frame