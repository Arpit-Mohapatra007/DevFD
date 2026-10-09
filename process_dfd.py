import os
import cv2
import random
from pathlib import Path
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Set reproducible seed for identical sampling
random.seed(42)

# Initialize MTCNN face detector
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
detector = MTCNN(keep_all=False, select_largest=True, device=device)

def extract_faces_from_video(video_path, output_dir, video_id, num_frames=20, img_size=(224, 224)):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return 0

    # Calculate the equal interval step
    interval = max(1, total_frames // num_frames)
    
    saved_count = 0
    current_idx = 0
    
    # Run until we get exactly 20 crops or run out of frames
    while saved_count < num_frames and current_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb_frame)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]
            h, w, _ = frame.shape
            
            # Pad bounding box by 15% to capture full head context
            pad_x = (box[2] - box[0]) * 0.15
            pad_y = (box[3] - box[1]) * 0.15

            x1 = max(0, int(box[0] - pad_x))
            y1 = max(0, int(box[1] - pad_y))
            x2 = min(w, int(box[2] + pad_x))
            y2 = min(h, int(box[3] + pad_y))

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                face_resized = cv2.resize(face_crop, img_size)
                out_path = os.path.join(output_dir, f"{video_id}_frame_{saved_count:02d}.jpg")
                cv2.imwrite(out_path, face_resized)
                
                saved_count += 1
                current_idx += interval
                continue
        
        # Fallback: No face found at this timestamp, check the immediate next frame
        current_idx += 1
        
    cap.release()
    return saved_count

if __name__ == "__main__":
    # Correct paths based on your terminal logs
    real_dir = Path("datasets/DFD/DFD_original sequences")
    fake_dir = Path("datasets/DFD/DFD_manipulated_sequences/DFD_manipulated_sequences")
    
    output_base = Path("processed_data/DFD")
    out_real = output_base / "real"
    out_fake = output_base / "fake"
    
    out_real.mkdir(parents=True, exist_ok=True)
    out_fake.mkdir(parents=True, exist_ok=True)
    
    # Grab all .mp4 files
    real_videos = list(real_dir.glob("*.mp4"))
    fake_videos = list(fake_dir.glob("*.mp4"))
    
    num_videos = 50
    num_frames = 20
    
    # Randomly sample exactly 50 real and 50 fake videos
    selected_real = random.sample(real_videos, min(num_videos, len(real_videos)))
    selected_fake = random.sample(fake_videos, min(num_videos, len(fake_videos)))
    
    print(f"\nProcessing DFD: Extracting faces from {len(selected_real)} REAL videos...")
    for v_path in tqdm(selected_real, desc="Real Videos"):
        video_id = v_path.stem 
        extract_faces_from_video(v_path, out_real, video_id, num_frames=num_frames)
        
    print(f"\nProcessing DFD: Extracting faces from {len(selected_fake)} FAKE videos...")
    for v_path in tqdm(selected_fake, desc="Fake Videos"):
        video_id = v_path.stem
        extract_faces_from_video(v_path, out_fake, video_id, num_frames=num_frames)
            
    print("\nExtraction complete for DFD dataset!")
