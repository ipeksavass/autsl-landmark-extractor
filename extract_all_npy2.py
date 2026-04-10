#Hücre 1
from google.colab import drive
drive.mount('/content/drive')

!pip install mediapipe
#-------------------------------------
#Hücre 2
%%writefile extract_all_npy.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import os
import urllib.request
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

MAX_FRAMES  = 30    
MIN_FRAMES  = 8     
N_WORKERS   = 8    
POSE_UPPER_BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24] 

HAND_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_PATH = "hand_landmarker.task"
POSE_MODEL_PATH = "pose_landmarker.task"

def ensure_models():
    for url, path in [(HAND_MODEL_URL, HAND_MODEL_PATH), (POSE_MODEL_URL, POSE_MODEL_PATH)]:
        if not os.path.exists(path): urllib.request.urlretrieve(url, path)

def resample_frames(frames, target=MAX_FRAMES):
    n = len(frames)
    if n == 0: return np.zeros((target, 153), dtype=np.float32)
    indices = np.linspace(0, n - 1, target)
    resampled = []
    for idx in indices:
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        t = idx - lo
        frame_vector = []
        
        pa, pb = frames[lo].get("pose"), frames[hi].get("pose")
        if pa is None and pb is None: frame_vector.extend([0.0] * 27)
        elif pa is None: frame_vector.extend(pb)
        elif pb is None: frame_vector.extend(pa)
        else: frame_vector.extend((np.array(pa) * (1 - t) + np.array(pb) * t).tolist())

        for hand in ("left_hand", "right_hand"):
            a, b = frames[lo].get(hand), frames[hi].get(hand)
            if a is None and b is None: frame_vector.extend([0.0] * 63)
            elif a is None: frame_vector.extend(b)
            elif b is None: frame_vector.extend(a)
            else: frame_vector.extend((np.array(a) * (1 - t) + np.array(b) * t).tolist())
                
        resampled.append(frame_vector)
    return np.array(resampled, dtype=np.float32)

def process_video(video_path):
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.5, min_tracking_confidence=0.5)

    pose_det = mp_vision.PoseLandmarker.create_from_options(pose_opts)
    hand_det = mp_vision.HandLandmarker.create_from_options(hand_opts)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, frame_idx = [], 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts_ms = int(frame_idx * 1000 / fps)

        pose_res = pose_det.detect_for_video(mp_img, ts_ms)
        pose_data = None
        if pose_res.pose_landmarks:
            pose_data = []
            for idx in POSE_UPPER_BODY: pose_data.extend([pose_res.pose_landmarks[0][idx].x, pose_res.pose_landmarks[0][idx].y, pose_res.pose_landmarks[0][idx].z])

        hand_res = hand_det.detect_for_video(mp_img, ts_ms)
        left_hand, right_hand = None, None
        if hand_res.hand_landmarks and hand_res.handedness:
            for lm_list, hd_list in zip(hand_res.hand_landmarks, hand_res.handedness):
                pts = [c for p in lm_list for c in (p.x, p.y, p.z)]
                if hd_list[0].category_name == "Left": left_hand = pts
                else: right_hand = pts

        frames.append({"pose": pose_data, "left_hand": left_hand, "right_hand": right_hand})
        frame_idx += 1

    cap.release()
    pose_det.close()
    hand_det.close()
    return frames

def extract_worker(args):
    video_path, output_path = args
    if Path(output_path).exists(): return video_path.name, True, "Zaten işlenmiş (Atlandı)"
    try:
        raw_frames = process_video(video_path)
        if raw_frames is None or len(raw_frames) < MIN_FRAMES: return video_path.name, False, "Çok kısa"
        np.save(output_path, resample_frames(raw_frames, MAX_FRAMES))
        return video_path.name, True, "Başarılı"
    except Exception as e: return video_path.name, False, f"Hata: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/content/drive/MyDrive/AUTSL")
    parser.add_argument("--output",  default="/content/drive/MyDrive/AUTSL_landmarks_npy")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    
    # İŞTE BURASI: Colab'in gizli argümanlarını (ör: -f kernel.json) göz ardı etmek için parse_known_args() kullanıyoruz
    args, unknown = parser.parse_known_args()

    dataset_path = Path(args.dataset)
    output_base_dir = Path(args.output)
    tasks, bulunan_kelimeler = [], set()

    for split in ["train", "val", "test"]:
        split_in_dir = dataset_path / split
        if not split_in_dir.exists(): continue
        for word_dir in split_in_dir.iterdir():
            if not word_dir.is_dir(): continue
            bulunan_kelimeler.add(word_dir.name)
            word_out_dir = output_base_dir / split / word_dir.name
            word_out_dir.mkdir(parents=True, exist_ok=True)
            for video_file in word_dir.glob("*_color.mp4"):
                tasks.append((video_file, word_out_dir / video_file.name.replace("_color.mp4", ".npy")))

    ensure_models()
    print(f"\n  Bulunan Kelime Sayısı  : {len(bulunan_kelimeler)}")
    print(f"  Toplam İşlenecek Video : {len(tasks)}\n")
    if not tasks: return

    start, success, fail, skipped = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, future in enumerate(as_completed({executor.submit(extract_worker, t): t for t in tasks}), 1):
            vid_name, ok, msg = future.result()
            if "Atlandı" in msg: skipped += 1
            else: success += 1 if ok else 0; fail += 1 if not ok else 0
            print(f"[{i:4}/{len(tasks)}] {'✓' if ok else '✗'} {vid_name:<30} {msg}")

    print(f"\nSüre: {(time.time() - start)/60:.1f} dk | Yeni: {success} | Atlanan: {skipped} | Hata: {fail}")

if __name__ == "__main__":
    main()
#---------------------------
#Hücre 3
!python extract_all_npy.py



#AUTSL_landmarks_npy klasörü boş bir şekilde driveda oluşturulmalı 
