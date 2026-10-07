import os, random
from pathlib import Path
import cv2, numpy as np, torch
from tqdm import tqdm
from facenet_pytorch import MTCNN

ORIG_DIR = Path("datasets/FF++/FaceForensics++_C23/original")
OUT_DIR = Path("pretrain_faces")
USED_DIRS = ["processed_data/FF++/real"]
FRAMES_PER_VIDEO, PAD, IMG_SIZE = 10, 0.15, (224, 224)   # same pad / size as process_all_categories.py

device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(keep_all=False, select_largest=True, device=device)   # same settings as your script


def used_ids():
    ids = set()
    for d in USED_DIRS:
        if not os.path.isdir(d):
            print(f"  (skip, not found) {d}")
            continue
        for f in os.listdir(d):
            stem = f.split("_frame")[0]          # "005_010_frame_03.jpg" -> "005_010"
            ids |= set(stem.split("_"))          # {"005", "010"}
    return ids


def crop_face(frame):
    boxes, _ = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if boxes is None or len(boxes) == 0:
        return None
    box = boxes[0]
    h, w, _ = frame.shape
    px, py = (box[2] - box[0]) * PAD, (box[3] - box[1]) * PAD
    x1, y1 = max(0, int(box[0] - px)), max(0, int(box[1] - py))
    x2, y2 = min(w, int(box[2] + px)), min(h, int(box[3] + py))
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, IMG_SIZE) if crop.size > 0 else None


def extract(video_path, vid):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return 0
    saved = 0
    for k, idx in enumerate(np.linspace(0, total - 1, FRAMES_PER_VIDEO + 2)[1:-1].astype(int)):
        for j in range(5):                       # fallback: try next frames if no face
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx + j, total - 1))
            ok, frame = cap.read()
            if not ok:
                break
            face = crop_face(frame)
            if face is not None:
                cv2.imwrite(str(OUT_DIR / f"{vid}_frame_{saved:02d}.jpg"), face)
                saved += 1
                break
    cap.release()
    return saved


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    used = used_ids()
    print(f"excluded ids: {len(used)}")
    videos = sorted(ORIG_DIR.rglob("*.mp4"))
    todo = [v for v in videos if v.stem not in used]
    print(f"{len(videos)} original videos, {len(todo)} unused -> pre-training")
    total = 0
    for v in tqdm(todo):
        total += extract(v, v.stem)
    print(f"saved {total} face crops in {OUT_DIR}/")
