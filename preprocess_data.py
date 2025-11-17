import os
import cv2
import numpy as np
import mediapipe as mp
import librosa
import soundfile as sf
from tqdm import tqdm
import shutil
import random
import warnings
import subprocess
warnings.filterwarnings('ignore')

# --- Constants ---
# Video
FRAME_RATE = 30
WINDOW_SECONDS = 2
STRIDE_SECONDS = 0.5
IMG_SIZE = (224, 224)

# Audio
SAMPLE_RATE = 16000
AUDIO_WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
AUDIO_STRIDE_SAMPLES = int(STRIDE_SECONDS * SAMPLE_RATE)
VAD_TOP_DB = 30  # Threshold for VAD

# --- Paths ---
BASE_DIR = r"D:\github_projects\stressdetect"
DATA_DIR = BASE_DIR
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed_data")
FFMPEG_EXE = os.path.join(BASE_DIR, "ffmpeg_portable", "ffmpeg.exe")

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5
)

def check_ffmpeg():
    """Check if FFmpeg is available."""
    if not os.path.exists(FFMPEG_EXE):
        print("\n" + "="*60)
        print("ERROR: FFmpeg not found!")
        print(f"Expected location: {FFMPEG_EXE}")
        print("\nPlease run the PowerShell commands to download FFmpeg.")
        print("="*60 + "\n")
        return False
    
    # Test FFmpeg
    try:
        result = subprocess.run(
            [FFMPEG_EXE, "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ FFmpeg found: {FFMPEG_EXE}\n")
            return True
    except:
        pass
    
    print(f"\n✗ FFmpeg found but not working: {FFMPEG_EXE}\n")
    return False

def align_face(image, landmarks):
    """
    Aligns the face by rotating the image so the eyes are horizontal.
    """
    # Using keypoints for the eyes to calculate angle
    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]
    
    # Get coordinates
    left_eye_x = int(left_eye.x * image.shape[1])
    left_eye_y = int(left_eye.y * image.shape[0])
    right_eye_x = int(right_eye.x * image.shape[1])
    right_eye_y = int(right_eye.y * image.shape[0])
    
    # Calculate angle
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # clockwise
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # counter-clockwise
        
    a = np.linalg.norm(np.array([left_eye_x, left_eye_y]) - np.array([point_3rd[0], point_3rd[1]]))
    b = np.linalg.norm(np.array([right_eye_x, right_eye_y]) - np.array([point_3rd[0], point_3rd[1]]))
    c = np.linalg.norm(np.array([right_eye_x, right_eye_y]) - np.array([left_eye_x, left_eye_y]))
    
    if b != 0 and c != 0:
        cos_a = (b**2 + c**2 - a**2) / (2 * b * c)
        cos_a = np.clip(cos_a, -1.0, 1.0)  # Avoid numerical errors
        angle = np.arccos(cos_a)
        angle = (angle * 180) / np.pi
        
        if direction == -1:
            angle = 90 - angle
            
        # Rotate image
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, (direction * angle), 1.0)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated_image
    return image  # Return original if alignment fails

def augment_video_window(window):
    """Applies random augmentations to a video window."""
    # Random horizontal flip
    if random.random() > 0.5:
        window = np.array([cv2.flip(frame, 1) for frame in window])
    
    # Color Jitter (simple version)
    if random.random() > 0.5:
        # Ensure the input is BGR uint8
        if window.dtype != np.uint8:
            window = (window * 255).astype(np.uint8)

        hsv = cv2.cvtColor(window[0], cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        hue_shift = random.uniform(-10, 10)
        sat_shift = random.uniform(0.8, 1.2)
        val_shift = random.uniform(0.8, 1.2)
        
        h = cv2.add(h, hue_shift)
        s = np.clip(cv2.multiply(s.astype(np.float32), sat_shift), 0, 255).astype(np.uint8)
        v = np.clip(cv2.multiply(v.astype(np.float32), val_shift), 0, 255).astype(np.uint8)
        
        final_hsv = cv2.merge((h, s, v))
        jittered_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        
        # Apply jitter to all frames in the window for consistency
        jittered_window = np.array([jittered_frame] * len(window))
        return jittered_window.astype(np.float32) / 255.0

    return window

def augment_audio_window(window):
    """Applies random augmentations to an audio window."""
    # Add noise
    if random.random() > 0.5:
        noise_amp = 0.005 * np.random.uniform() * np.amax(np.abs(window))
        if noise_amp > 0:
            window = window + noise_amp * np.random.normal(size=window.shape[0])
        
    # Time stretch
    if random.random() > 0.5:
        rate = np.random.uniform(0.8, 1.2)
        window = librosa.effects.time_stretch(y=window, rate=rate)
        # Ensure correct length after stretch
        if len(window) > AUDIO_WINDOW_SAMPLES:
            window = window[:AUDIO_WINDOW_SAMPLES]
        elif len(window) < AUDIO_WINDOW_SAMPLES:
            window = np.pad(window, (0, AUDIO_WINDOW_SAMPLES - len(window)), mode='constant')
        
    # Pitch shift
    if random.random() > 0.5:
        n_steps = np.random.randint(-2, 3)
        if n_steps != 0:
            window = librosa.effects.pitch_shift(y=window, sr=SAMPLE_RATE, n_steps=n_steps)
        
    return window

def extract_audio_ffmpeg(video_path, output_dir):
    """Extract audio from video using FFmpeg."""
    try:
        temp_audio_path = os.path.join(output_dir, f"temp_audio_{os.getpid()}_{random.randint(1000,9999)}.wav")
        
        # FFmpeg command to extract audio
        command = [
            FFMPEG_EXE,
            "-i", video_path,
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # Audio codec
            "-ar", str(SAMPLE_RATE),  # Sample rate
            "-ac", "1",  # Mono
            "-y",  # Overwrite
            temp_audio_path
        ]
        
        # Run FFmpeg
        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        if result.returncode != 0 or not os.path.exists(temp_audio_path):
            return None
        
        # Check if file has content
        if os.path.getsize(temp_audio_path) == 0:
            os.remove(temp_audio_path)
            return None
        
        # Load audio with librosa
        audio_waveform, _ = librosa.load(temp_audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Clean up temp file
        try:
            os.remove(temp_audio_path)
        except:
            pass
        
        return audio_waveform
        
    except Exception as e:
        return None

def process_file(video_path, label, output_dir):
    """Main processing function for a single video file."""
    try:
        # Check if video file exists
        if not os.path.exists(video_path):
            return
            
        # --- Video Processing ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            return

        # Process frames: face detection, alignment, cropping, normalization
        processed_frames = []
        for frame in frames:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Align face
                aligned_frame = align_face(frame, face_landmarks)
                
                # Get bounding box
                h, w, _ = aligned_frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                # Add padding to bounding box
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Crop face
                face_crop = aligned_frame[y_min:y_max, x_min:x_max]
                if face_crop.size == 0:
                    processed_frames.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))
                    continue

                # Resize to 224x224
                resized_face = cv2.resize(face_crop, IMG_SIZE)
                # Normalize to [0, 1]
                normalized_face = resized_face.astype(np.float32) / 255.0
                processed_frames.append(normalized_face)
            else:
                # No face detected, use zero frame
                processed_frames.append(np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32))

        # Create sliding windows for video (2s window, 0.5s stride)
        video_windows = []
        num_frames_in_window = int(WINDOW_SECONDS * FRAME_RATE)
        num_frames_in_stride = int(STRIDE_SECONDS * FRAME_RATE)
        
        for i in range(0, len(processed_frames) - num_frames_in_window + 1, num_frames_in_stride):
            window = processed_frames[i:i + num_frames_in_window]
            if len(window) == num_frames_in_window:
                video_windows.append(np.array(window))

        # --- Audio Processing ---
        audio_waveform = extract_audio_ffmpeg(video_path, output_dir)
        
        if audio_waveform is None or len(audio_waveform) == 0:
            return
        
        # Voice Activity Detection (VAD)
        clips = librosa.effects.split(audio_waveform, top_db=VAD_TOP_DB)
        if len(clips) > 0:
            vad_audio = np.concatenate([audio_waveform[start:end] for start, end in clips])
        else:
            vad_audio = audio_waveform  # Use original if VAD finds nothing

        # Create sliding windows for audio
        audio_windows = []
        for i in range(0, len(vad_audio) - AUDIO_WINDOW_SAMPLES + 1, AUDIO_STRIDE_SAMPLES):
            window = vad_audio[i:i + AUDIO_WINDOW_SAMPLES]
            if len(window) == AUDIO_WINDOW_SAMPLES:
                audio_windows.append(window)

        # --- Sync, Augment, and Save ---
        num_windows = min(len(video_windows), len(audio_windows))
        
        if num_windows == 0:
            return
        
        for i in range(num_windows):
            video_window = video_windows[i]
            audio_window = audio_windows[i]

            # Apply augmentations
            video_window = augment_video_window(video_window)
            audio_window = augment_audio_window(audio_window)
            
            # Extract audio features
            mel_spec = librosa.feature.melspectrogram(
                y=audio_window, 
                sr=SAMPLE_RATE, 
                n_mels=64
            )
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            mfccs = librosa.feature.mfcc(
                y=audio_window, 
                sr=SAMPLE_RATE, 
                n_mfcc=13
            )
            
            pitch, _ = librosa.piptrack(
                y=audio_window, 
                sr=SAMPLE_RATE
            )
            
            rms = librosa.feature.rms(y=audio_window)
            
            audio_features = {
                "log_mel": log_mel_spec, 
                "mfcc": mfccs, 
                "pitch": pitch, 
                "rms": rms
            }

            # Save preprocessed data
            output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_window_{i}"
            np.save(os.path.join(output_dir, f"{output_filename}_video.npy"), video_window)
            np.savez(os.path.join(output_dir, f"{output_filename}_audio.npz"), **audio_features)

    except Exception as e:
        pass  # Silent fail to avoid cluttering output

def main():
    """Main function to iterate through all videos and preprocess them."""
    print("="*60)
    print("DATA PREPROCESSING - STRESS DETECTION")
    print("="*60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Output directory: {PREPROCESSED_DIR}\n")
    
    # Check FFmpeg
    if not check_ffmpeg():
        return
    
    # Remove existing preprocessed data
    if os.path.exists(PREPROCESSED_DIR):
        print("Removing existing preprocessed data...")
        shutil.rmtree(PREPROCESSED_DIR)
    
    for label in ["stress", "non-stress"]:
        input_label_dir = os.path.join(DATA_DIR, label)
        output_label_dir = os.path.join(PREPROCESSED_DIR, label)
        os.makedirs(output_label_dir, exist_ok=True)
        
        # Collect all video files
        video_files = []
        for root, _, files in os.walk(input_label_dir):
            for file in files:
                if file.lower().endswith(".mp4"):
                    video_files.append(os.path.join(root, file))
        
        print(f"\nFound {len(video_files)} video files in '{label}' category")
        
        if len(video_files) == 0:
            print(f"WARNING: No MP4 files found in {input_label_dir}")
            continue
        
        # Process each video file
        successful = 0
        for video_file in tqdm(video_files, desc=f"Processing {label} videos"):
            before_count = len([f for f in os.listdir(output_label_dir) if f.endswith('.npy')])
            process_file(video_file, label, output_label_dir)
            after_count = len([f for f in os.listdir(output_label_dir) if f.endswith('.npy')])
            if after_count > before_count:
                successful += 1
        
        print(f"✓ Successfully processed {successful}/{len(video_files)} videos in '{label}'")
    
    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETED!")
    print(f"Preprocessed data saved to: {PREPROCESSED_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()