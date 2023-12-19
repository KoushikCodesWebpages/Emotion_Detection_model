import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

emotions = ['angry', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'neutral']

for emotion in emotions:
    
    main_directory = os.path.join(r'EX_Data/', emotion)

    if not os.path.exists(main_directory):
        print(f"Directory not found: {main_directory}")
        continue
   
    emotion_images = os.listdir(main_directory)
    df = pd.DataFrame({'filename': emotion_images})

    train_ratio = 0.8

    train_df, test_df = train_test_split(df, test_size=1 - train_ratio, random_state=42)

    train_directory = os.path.join(r'EX_TRAIN/', emotion)
    test_directory = os.path.join(r'EX_VAL', emotion)

    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(test_directory, exist_ok=True)

    for _, row in train_df.iterrows():
        src_path = os.path.join(main_directory, row['filename'])
        dest_path = os.path.join(train_directory, row['filename'])
        shutil.move(src_path, dest_path)

    for _, row in test_df.iterrows():
        src_path = os.path.join(main_directory, row['filename'])
        dest_path = os.path.join(test_directory, row['filename'])
        shutil.move(src_path, dest_path)

print("Dataset splitting complete.")
