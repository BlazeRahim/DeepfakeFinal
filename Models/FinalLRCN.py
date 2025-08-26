# FinalLRCN.py
# Deepfake video classifier (face -> VGG16 features -> LRCN) with caching, versioned cache, and detailed debug logging.

import os
import time
from pathlib import Path
from typing import List

import numpy as np
import cv2
import tensorflow as tf

# ----------------------------
# Quick-run / debug toggles
# ----------------------------
SAMPLE_LIMIT_REAL = 0   # set >0 to cap how many real videos to process (e.g., 50). 0 = no cap
SAMPLE_LIMIT_FAKE = 0   # set >0 to cap how many fake videos to process (e.g., 200). 0 = no cap
SKIP_FEATURE_EXTRACTION = False  # True = use existing cache only; fail if missing
SKIP_TRAINING = False            # True = only prepare data and exit after split

# ----------------------------
# Config
# ----------------------------
REAL_DIR = r"Videos/Real"
FAKE_DIR = r"Videos/Fake"

SEQUENCE_LENGTH = 15             # frames per video (consistent across all samples)
FRAME_STRIDE = 15                # detect face every N frames
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 1e-4
SEED = 42
USE_MIXED_PRECISION = True

MODEL_PATH = "lrcn_deepfake_model.h5"
CACHE_ROOT = Path(f"cache/features/seq{SEQUENCE_LENGTH}_stride{FRAME_STRIDE}_vgg16avg")
CACHE_REAL = CACHE_ROOT / "real"
CACHE_FAKE = CACHE_ROOT / "fake"
CACHE_REAL.mkdir(parents=True, exist_ok=True)
CACHE_FAKE.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Headless plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Progress bars
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x  # fallback: no progress bar

# Safer OpenCV (avoid rare hangs due to OpenCL/threads)
cv2.ocl.setUseOpenCL(False)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

# ----------------------------
# TensorFlow / GPU setup
# ----------------------------
if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for _g in gpus:
            tf.config.experimental.set_memory_growth(_g, True)
        print(f"✅ TF sees GPU(s): {[d.name for d in gpus]}")
    except RuntimeError as e:
        print("GPU memory growth could not be set:", e)

# ----------------------------
# Keras / Sklearn imports
# ----------------------------
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import to_categorical, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix

# ----------------------------
# Utility logging helpers
# ----------------------------
def banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def stamp() -> str:
    return time.strftime("%H:%M:%S")

# ----------------------------
# Utilities
# ----------------------------
def list_videos(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    vids = [str(Path(folder, f)) for f in os.listdir(folder)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    vids.sort()
    return vids

def maybe_limit(videos: List[str], limit: int) -> List[str]:
    if limit and limit > 0:
        return videos[:limit]
    return videos

def cache_path_for(video_path: str, class_name: str) -> str:
    fname = Path(video_path).stem + ".npy"
    sub = CACHE_REAL if class_name == "real" else CACHE_FAKE
    return str(sub / fname)

# ----------------------------
# Face extraction
# ----------------------------
def extract_faces_from_video(video_path: str,
                             sequence_length: int = SEQUENCE_LENGTH,
                             frame_stride: int = FRAME_STRIDE) -> np.ndarray:
    """
    Returns (sequence_length, 224, 224, 3), uint8 RGB frames.
    Uses Haar cascade (CPU). If no face, fallback to center-crop.
    Pads with black frames if needed.
    """
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{stamp()}] ⚠️ Cannot open video: {video_path}")
        return np.zeros((sequence_length, 224, 224, 3), dtype=np.uint8)

    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(face_xml):
        print(f"[{stamp()}] ⚠️ Haar cascade missing at {face_xml}")
    face_cascade = cv2.CascadeClassifier(face_xml)

    faces = []
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(detected) > 0:
                (x, y, w, h) = max(detected, key=lambda bb: bb[2] * bb[3])
                crop_bgr = frame_bgr[y:y + h, x:x + w]
            else:
                h, w, _ = frame_bgr.shape
                sz = min(h, w)
                y0 = (h - sz) // 2
                x0 = (w - sz) // 2
                crop_bgr = frame_bgr[y0:y0 + sz, x0:x0 + sz]

            crop_rgb = cv2.cvtColor(cv2.resize(crop_bgr, (224, 224)), cv2.COLOR_BGR2RGB)
            faces.append(crop_rgb)

        frame_idx += 1
        if len(faces) >= sequence_length:
            break

    cap.release()

    while len(faces) < sequence_length:
        faces.append(np.zeros((224, 224, 3), dtype=np.uint8))

    t1 = time.perf_counter()
    return np.asarray(faces[:sequence_length], dtype=np.uint8)

# ----------------------------
# VGG16 feature extraction
# ----------------------------
def build_vgg16_feature_extractor():
    banner(f"[{stamp()}] Building VGG16 feature extractor (pooling='avg')")
    return VGG16(weights="imagenet", include_top=False, pooling="avg")

def faces_to_features(faces_rgb: np.ndarray, vgg_model) -> np.ndarray:
    """
    faces_rgb: (T, 224, 224, 3) uint8 RGB
    returns: (T, F) float32
    """
    T = faces_rgb.shape[0]
    feats = []
    for t in range(T):
        img = faces_rgb[t]
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = vgg_model.predict(x, verbose=0)  # (1, 512)
        feats.append(feat[0])
    return np.asarray(feats, dtype=np.float32)

def load_or_compute_features_for_video(video_path: str, class_name: str, vgg_model) -> np.ndarray:
    """
    Returns (SEQUENCE_LENGTH, feature_dim).
    Loads from per-video cache if valid; otherwise recomputes and writes cache.
    Includes detailed per-video timing/logging.
    """
    cpath = cache_path_for(video_path, class_name)

    # Attempt to load cache
    if os.path.exists(cpath):
        try:
            arr = np.load(cpath)
            if arr.ndim == 2 and arr.shape[0] == SEQUENCE_LENGTH:
                return arr
            else:
                print(f"[{stamp()}] ♻️ Rebuild cache (shape mismatch) -> {Path(video_path).name}, cached {arr.shape}")
        except Exception as e:
            print(f"[{stamp()}] ♻️ Rebuild corrupt cache -> {Path(video_path).name}: {e}")

    if SKIP_FEATURE_EXTRACTION:
        raise RuntimeError(f"Cache missing but SKIP_FEATURE_EXTRACTION=True: {cpath}")

    t0 = time.perf_counter()
    faces = extract_faces_from_video(video_path, SEQUENCE_LENGTH, FRAME_STRIDE)
    t1 = time.perf_counter()
    features = faces_to_features(faces, vgg_model)
    t2 = time.perf_counter()
    np.save(cpath, features)
    t3 = time.perf_counter()

    print(f"[{stamp()}] ✔ {Path(video_path).name} | faces {t1 - t0:.2f}s | vgg {t2 - t1:.2f}s | save {t3 - t2:.2f}s")
    return features

# ----------------------------
# Plotting helpers
# ----------------------------
def plot_roc_curve_safe(y_true_bin: np.ndarray, y_score_pos: np.ndarray, out_path="ROC_curve.png"):
    classes_present = set(y_true_bin.tolist())
    if classes_present == {0, 1}:
        fpr, tpr, _ = roc_curve(y_true_bin, y_score_pos)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(out_path)
        plt.close()
        print(f"📈 ROC saved -> {out_path}")
    else:
        print("⚠️ Skipping ROC: test set has a single class only.")

def plot_confusion_matrix_safe(y_true: np.ndarray, y_pred: np.ndarray,
                               labels=("Fake", "Real"), out_path="confusion_matrix.png"):
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Confusion matrix saved -> {out_path}")
    except Exception as e:
        print("⚠️ Skipping confusion matrix:", e)

# ----------------------------
# Model
# ----------------------------
def build_lrcn(sequence_length: int, feature_dim: int) -> Sequential:
    model = Sequential()
    model.add(TimeDistributed(Dense(256, activation='relu'),
                              input_shape=(sequence_length, feature_dim)))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax', dtype='float32'))  # ensure float32 output under mixed precision
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=['accuracy'])
    return model

# ----------------------------
# Main
# ----------------------------
def main():
    banner(f"[{stamp()}] Stage 1: Listing videos")
    real_videos = list_videos(REAL_DIR)
    fake_videos = list_videos(FAKE_DIR)

    if len(real_videos) == 0 or len(fake_videos) == 0:
        print("❌ No videos found. Check REAL_DIR/FAKE_DIR paths.")
        return

    real_videos = maybe_limit(real_videos, SAMPLE_LIMIT_REAL)
    fake_videos = maybe_limit(fake_videos, SAMPLE_LIMIT_FAKE)

    print(f"Found {len(real_videos)} real, {len(fake_videos)} fake videos")

    banner(f"[{stamp()}] Stage 2: Build VGG16")
    vgg = build_vgg16_feature_extractor()

    banner(f"[{stamp()}] Stage 3: Feature extraction (cached)")
    X_real = []
    for vp in tqdm(real_videos, desc="Real videos"):
        X_real.append(load_or_compute_features_for_video(vp, "real", vgg))

    X_fake = []
    for vp in tqdm(fake_videos, desc="Fake videos"):
        X_fake.append(load_or_compute_features_for_video(vp, "fake", vgg))

    banner(f"[{stamp()}] Stage 4: Stack arrays")
    try:
        X_real = np.stack(X_real, axis=0)
        X_fake = np.stack(X_fake, axis=0)
    except ValueError as e:
        print("♻️ Inconsistent cache shapes; forcing rebuild...")
        for p in list(CACHE_REAL.glob("*.npy")) + list(CACHE_FAKE.glob("*.npy")):
            try:
                p.unlink()
            except Exception:
                pass
        X_real = np.stack([load_or_compute_features_for_video(vp, "real", vgg) for vp in real_videos], axis=0)
        X_fake = np.stack([load_or_compute_features_for_video(vp, "fake", vgg) for vp in fake_videos], axis=0)

    y_real = np.ones((len(X_real),), dtype=np.int32)   # 1 = Real
    y_fake = np.zeros((len(X_fake),), dtype=np.int32)  # 0 = Fake

    X = np.concatenate([X_real, X_fake], axis=0)
    y = np.concatenate([y_real, y_fake], axis=0)

    print(f"✅ Features: X={X.shape} (N, T, F)  y={y.shape}  classes={np.unique(y)}")
    seq_len, feat_dim = X.shape[1], X.shape[2]

    banner(f"[{stamp()}] Stage 5: Train/Test split")
    try:
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        print("⚠️ Stratify disabled due to class counts.")

    y_train = to_categorical(y_train_int, num_classes=2)
    y_test  = to_categorical(y_test_int,  num_classes=2)

    if SKIP_TRAINING:
        print("⏭️ SKIP_TRAINING=True — stopping before model training.")
        return

    banner(f"[{stamp()}] Stage 6: Build model")
    model = build_lrcn(seq_len, feat_dim)
    model.summary()

    banner(f"[{stamp()}] Stage 7: Train")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    banner(f"[{stamp()}] Stage 8: Save model")
    model.save(MODEL_PATH)
    print(f"💾 Model saved -> {MODEL_PATH}")

    banner(f"[{stamp()}] Stage 9: Evaluate")
    y_pred_proba = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred_proba, axis=1)

    print("\n=== Classification Report (0=Fake, 1=Real) ===")
    print(classification_report(y_test_int, y_pred_cls, digits=4))

    plot_roc_curve_safe(y_true_bin=y_test_int, y_score_pos=y_pred_proba[:, 1], out_path="ROC_curve.png")
    plot_confusion_matrix_safe(y_true=y_test_int, y_pred=y_pred_cls, labels=("Fake", "Real"), out_path="confusion_matrix.png")

    print("✅ Done.")

# ----------------------------
# Entry
# ----------------------------
if __name__ == "__main__":
    banner(f"[{stamp()}] START")
    main()
    banner(f"[{stamp()}] END")
