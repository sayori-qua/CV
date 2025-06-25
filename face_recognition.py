from mtcnn import MTCNN
from deepface import DeepFace
import cv2
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

source = r"C:\Users\user\PycharmProjects\CV\videos_for_recognition\0001d815c0--6203fa2ad1a403449593c5cb.mp4"

def face_recognition_func(source):

    detector = MTCNN()

    open_video = cv2.VideoCapture(source)
    while open_video.isOpened():
        rating, frame = open_video.read()
        if not rating:
            break
        rgb_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_vid)
        for face in faces:
            x, y, width, height = face['box']
            face_img = rgb_vid[y:y + height, x:x + width]
            try:
                result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
                gender = result[0]['gender']
                info = f"{gender}"
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2) #объект, левый верхний, правый нижний, цвет и размер илнии выделения, система не декартова
                cv2.putText(frame, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                print("Error:", e)
                continue
        cv2.imshow("MTCNN and DeepFace", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    open_video.release()
    cv2.destroyAllWindows()

