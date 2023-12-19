import cv2
import os
import time

def capture_and_save_frames(output_folder, max_frames=1000):
    cap = cv2.VideoCapture(0)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY
                     )
        image_path = os.path.join(output_folder, f"image_{frame_count + 1}.png")
        cv2.imwrite(image_path, frame)

        cv2.imshow('Capturing Frames', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_folder = r"Koushi_twist\NEUTRAL"
    capture_and_save_frames(output_folder, max_frames=1000)