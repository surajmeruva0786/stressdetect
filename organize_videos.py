import os
import shutil

# Define the mapping of emotion codes to emotion names
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Define which emotions are stress and non-stress
stress_emotions = ["sad", "angry", "fearful", "disgust", "surprised"]
non_stress_emotions = ["neutral", "calm", "happy"]

# Get the absolute path of the script's directory
base_dir = "d:\\github_projects\\stressdetect"
content_path = os.path.join(base_dir, "content")

# Create stress and non-stress folders at the root
stress_path = os.path.join(base_dir, "stress")
non_stress_path = os.path.join(base_dir, "non-stress")
os.makedirs(stress_path, exist_ok=True)
os.makedirs(non_stress_path, exist_ok=True)

# Create emotion folders inside stress/non-stress folders
for code, emotion in emotion_map.items():
    if emotion in stress_emotions:
        emotion_folder_path = os.path.join(stress_path, emotion)
    elif emotion in non_stress_emotions:
        emotion_folder_path = os.path.join(non_stress_path, emotion)
    else:
        continue
    os.makedirs(emotion_folder_path, exist_ok=True)

# Get a list of all files in the content directory
files = [f for f in os.listdir(content_path) if os.path.isfile(os.path.join(content_path, f))]

# Move files to emotion folders
for file in files:
    try:
        parts = file.split("-")
        if len(parts) == 7:
            emotion_code = parts[2]
            if emotion_code in emotion_map:
                emotion_name = emotion_map[emotion_code]
                
                if emotion_name in stress_emotions:
                    dest_folder_path = os.path.join(stress_path, emotion_name)
                elif emotion_name in non_stress_emotions:
                    dest_folder_path = os.path.join(non_stress_path, emotion_name)
                else:
                    continue

                src_path = os.path.join(content_path, file)
                dest_path = os.path.join(dest_folder_path, file)
                shutil.move(src_path, dest_path)
    except IndexError:
        print(f"Could not process file: {file}. Filename does not match expected format.")

print("Files organized successfully.")
