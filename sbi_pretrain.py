import os, sys, glob, random
import cv2, numpy as np, timm, torch, torch.nn as nn
import mediapipe as mp
from torch.utils.data import Dataset, DataLoader

cv2.setNumThreads(0)
FACE_DIR, CACHE = "pretrain_faces", "sbi_landmarks.npz"
MAX_REAL, EPOCHS, BATCH, LR, SIZE, PADLM = 8000, 10, 64, 1e-5, 224, 56
MEAN = np.array([0.4815, 0.4578, 0.4082], np.float32)
STD = np.array([0.2686, 0.2613, 0.2758], np.float32)


def real_paths():
    paths = sorted(glob.glob(f"{FACE_DIR}/*.jpg"))
    if len(paths) > MAX_REAL:
        paths = random.Random(0).sample(paths, MAX_REAL)
    return paths


def read_rgb(p):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (SIZE, SIZE))


def build_items(paths):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z['req']) == paths:
            print("loaded landmark cache")
            return list(z['paths']), z['lms']
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    ok_paths, lms = [], []
    for i, p in enumerate(paths):
        img = read_rgb(p)
        # MTCNN crops are tight: pad before landmark detection, then shift back
        pad = cv2.copyMakeBorder(img, PADLM, PADLM, PADLM, PADLM, cv2.BORDER_REPLICATE)
        res = fm.process(pad)
        if res.multi_face_landmarks:
            S = SIZE + 2 * PADLM
            pts = np.array([[q.x * S - PADLM, q.y * S - PADLM] for q in res.multi_face_landmarks[0].landmark], np.float32)
            pts = np.clip(pts, 0, SIZE - 1)
            ok_paths.append(p); lms.append(pts)
        if (i + 1) % 1000 == 0:
            print(f"  landmarks: {i+1}/{len(paths)}", flush=True)
    lms = np.stack(lms)
    np.savez(CACHE, req=np.array(paths, dtype=object), paths=np.array(ok_paths, dtype=object), lms=lms)
    return ok_paths, lms


# ---------- SBI (simplified Shiohara & Yamasaki) ----------
def color_aug(img):
    img = img.astype(np.float32)
    img *= random.uniform(0.85, 1.15)
    img += random.uniform(-20, 20)
    img += np.random.uniform(-15, 15, size=(1, 1, 3))
    return np.clip(img, 0, 255).astype(np.uint8)

def degrade(img):
    if random.random() < 0.5:
        s = random.uniform(0.5, 0.95); h, w = img.shape[:2]
        img = cv2.resize(cv2.resize(img, (int(w * s), int(h * s))), (w, h))
    elif random.random() < 0.5:
        k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        img = cv2.filter2D(img, -1, k)
    return img

def make_sbi(img, lm):
    h, w = img.shape[:2]
    src, tgt = img.copy(), img.copy()
    if random.random() < 0.5: src = degrade(color_aug(src))
    else:                     tgt = degrade(color_aug(tgt))
    s = random.uniform(0.95, 1.05); tx, ty = np.random.uniform(-0.03, 0.03, 2) * w
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s); M[:, 2] += (tx, ty)
    src = cv2.warpAffine(src, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    hull = cv2.convexHull(lm.astype(np.int32))
    mask = np.zeros((h, w), np.float32); cv2.fillConvexPoly(mask, hull, 1.0)
    k = random.choice([9, 15, 21, 31])
    mask = cv2.GaussianBlur(mask, (k, k), 0) * random.choice([0.25, 0.5, 0.75, 1, 1, 1])
    mask = mask[..., None]
    return (mask * src + (1 - mask) * tgt).astype(np.uint8)


class SBIDataset(Dataset):                      # real = 1, SBI fake = 0
    def __init__(self, paths, lms):
        self.paths, self.lms = paths, lms
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = read_rgb(self.paths[i])
        if random.random() < 0.5: x, y = img, 1.0
        else:                     x, y = make_sbi(img, self.lms[i]), 0.0
        if random.random() < 0.5: x = x[:, ::-1]
        x = ((x.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.float32)


if __name__ == "__main__":
    paths = real_paths()
    print(f"real crops found: {len(paths)}")
    ok_paths, lms = build_items(paths)
    print(f"landmarks found for {len(ok_paths)}/{len(paths)} images")
    ds = SBIDataset(ok_paths, lms)

    if len(sys.argv) > 1 and sys.argv[1] == "--preview":
        tiles = []
        for i in range(8):
            img = read_rgb(ok_paths[i])
            tiles.append(np.concatenate([img, make_sbi(img, lms[i])], axis=0))
        cv2.imwrite("sbi_preview.png", cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))
        print("saved sbi_preview.png (top row real, bottom row SBI fake)")
        sys.exit()

    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4, drop_last=True)
    dev = 'cuda'
    vit = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True, num_classes=0).to(dev)
    head = nn.Linear(768, 1).to(dev)
    opt = torch.optim.AdamW(list(vit.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()

    for ep in range(EPOCHS):
        vit.train(); tot = correct = n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                logit = head(vit(x)).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(logit.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(y); correct += ((logit > 0).float() == y).sum().item(); n += len(y)
        print(f"epoch {ep+1}/{EPOCHS} loss {tot/n:.4f} acc {correct/n*100:.1f}%", flush=True)
        torch.save({'vit': vit.state_dict(), 'head': head.state_dict()}, 'clip_sbi.pt')
    print("saved clip_sbi.pt")
