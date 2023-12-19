import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def detect_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        return (x, y, w, h), face_roi
    else:
        return None, None

def load_and_preprocess_frame(frame):
    
    frame_resized = cv2.resize(frame, (48, 48))
    
    
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    img_array = np.expand_dims(gray_frame, axis=0)
    
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

def predict_emotion(model, frame):
    img_array = load_and_preprocess_frame(frame)
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions[0])
    
    emotions = ['Angry','Disgust', 'Fear','Happy', 'neutral', 'Sad', 'Surprise']
    predicted_emotion = emotions[emotion_index]
    
    return predicted_emotion

if __name__ == "__main__":
    model_path = "path_to_your_trained_model.h5"
    emotion_model = load_model(r"C:\Users\Koush\OneDrive\Desktop\virtusa_project\emoji_model_v3.h5")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        face_rect, face_roi = detect_face(frame, face_cascade)

        if face_roi is not None:
            x, y, w, h = face_rect
        
            predicted_emotion = predict_emotion(emotion_model, face_roi)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, f"Emotion: {predicted_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
