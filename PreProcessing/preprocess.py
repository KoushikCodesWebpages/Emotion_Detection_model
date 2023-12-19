import os
import cv2

def extract_faces_from_emotion_directory(emotion_directory, output_directory, face_cascade):
    for filename in os.listdir(emotion_directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            img_path = os.path.join(emotion_directory, filename)
            img = cv2.imread(img_path)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y + h, x:x + w]
                output_path = os.path.join(output_directory, f"{filename}_face_{i+1}.jpg")
                cv2.imwrite(output_path, face)

if __name__ == "__main__":
    In_dir = r"C:\Users\Koush\OneDrive\Desktop\virtusa_project\Emo_in"
    Out_dir = r"C:\Users\Koush\OneDrive\Desktop\virtusa_project\Emo_out"

    # Predefined list of emotions
    emotions = ['Sad', 'Happy', 'Angry', 'Surprise', 'Fear', 'Neutral', 'Disgust']

    # Cascade classifier loading
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Iterate through each emotion directory
    for emotion in emotions:
        emotion_directory = os.path.join(In_dir, emotion)
        output_directory = os.path.join(Out_dir, emotion)

        # Ensure the output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        extract_faces_from_emotion_directory(emotion_directory, output_directory, face_cascade)
