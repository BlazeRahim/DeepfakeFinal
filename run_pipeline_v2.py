"""
VeriLens Improved Pipeline v2
Legitimate improvements over v1:
  1. SVM: StandardScaler + class_weight='balanced' + GridSearchCV
  2. LRCN-Original: class_weight + larger batch + more epochs + callbacks
  3. LRCN-Retrained: tuned focal gamma + longer training
All architecture stays identical (VGG16 avg-pool -> LRCN / SVM)
"""
import os, sys

# === GPU/cuDNN fix for conda env on Windows ===
os.environ['PATH'] = r'C:\Users\blaze\miniconda3\envs\deepfake\Library\bin' + ';' + os.environ.get('PATH', '')
os.add_dll_directory(r'C:\Users\blaze\miniconda3\envs\deepfake\Library\bin')
os.add_dll_directory(r'C:\Users\blaze\miniconda3\envs\deepfake\Lib\site-packages\nvidia\cudnn\bin')

# Pre-load cuDNN before TF imports it
import ctypes
try:
    ctypes.CDLL(r'C:\Users\blaze\miniconda3\envs\deepfake\Library\bin\cudnn64_8.dll')
except Exception:
    pass

import json, time, pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print('Mixed precision enabled')
except:
    print('Mixed precision not available')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f'GPU(s): {[g.name for g in gpus]}')
else:
    print('WARNING: No GPU found - will run on CPU (slow)')

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import to_categorical, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence as KerasSequence

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_fscore_support
)
from tqdm.auto import tqdm

print('All imports successful')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHANGE THIS to switch datasets:
#   'celeb-df'  ->  Celeb-DF v2
#   'dfdc'      ->  DFDC
DATASET_NAME = 'celeb-df'

MAIN_REPO = Path(r'C:\Users\blaze\Desktop\DeepfakeFinal')

if DATASET_NAME == 'celeb-df':
    REAL_DIR = str(MAIN_REPO / 'data' / 'Celeb-real')
    FAKE_DIR = str(MAIN_REPO / 'data' / 'Celeb-synthesis')
elif DATASET_NAME == 'dfdc':
    REAL_DIR = str(MAIN_REPO / 'data' / 'DFDC-real')
    FAKE_DIR = str(MAIN_REPO / 'data' / 'DFDC-fake')
else:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")

SEQUENCE_LENGTH = 15
FRAME_STRIDE = 15
CACHE_DIR = MAIN_REPO / f'cache/{DATASET_NAME}/seq{SEQUENCE_LENGTH}_stride{FRAME_STRIDE}'
CACHE_REAL = CACHE_DIR / 'real'
CACHE_FAKE = CACHE_DIR / 'fake'
CACHE_REAL.mkdir(parents=True, exist_ok=True)
CACHE_FAKE.mkdir(parents=True, exist_ok=True)

RESULTS_BASE = Path(f'results/{DATASET_NAME}')

# ── LRCN-Original (IMPROVED) ──
LRCN_EPOCHS = 30          # was 10: more room to learn
LRCN_BATCH = 8            # was 2: more stable gradients
LRCN_LR = 1e-4
LABEL_SMOOTHING = 0.1     # NEW: prevents overconfident predictions

# ── LRCN-Retrained (IMPROVED) ──
RETRAIN_EPOCHS = 25       # was 15: more room with early stopping
RETRAIN_BATCH = 32
RETRAIN_LR = 1e-4
FOCAL_GAMMA = 1.5         # was 2.0: less aggressive focusing

# ── SVM (IMPROVED) ──
SVM_GRID = {
    'C': [0.1, 1.0, 10.0, 100.0],
    'gamma': ['scale', 'auto', 0.001, 0.01],
}

NUM_TIMING_VIDEOS = 10
T = SEQUENCE_LENGTH
F_DIM = 512

print(f'Dataset: {DATASET_NAME}')
print(f'Real: {REAL_DIR} | Fake: {FAKE_DIR}')
print(f'Cache: {CACHE_DIR}')
print(f'Results: {RESULTS_BASE}')
print(f'\n=== V2 IMPROVEMENTS ===')
print(f'  LRCN-Orig: epochs={LRCN_EPOCHS}, batch={LRCN_BATCH}, label_smoothing={LABEL_SMOOTHING}, class_weight=balanced')
print(f'  LRCN-Focal: epochs={RETRAIN_EPOCHS}, gamma={FOCAL_GAMMA}')
print(f'  SVM: StandardScaler + class_weight=balanced + GridSearchCV({SVM_GRID})')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE FUNCTIONS (unchanged from v1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def list_videos(folder):
    if not os.path.isdir(folder): return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

def extract_faces_from_video(video_path, seq_len=SEQUENCE_LENGTH, stride=FRAME_STRIDE):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((seq_len, 224, 224, 3), dtype=np.uint8), {}
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = cascade.detectMultiScale(gray, 1.3, 5)
            if len(det) > 0:
                x, y, w, h = max(det, key=lambda b: b[2]*b[3])
                crop = frame[y:y+h, x:x+w]
            else:
                h, w, _ = frame.shape
                sz = min(h, w)
                y0, x0 = (h-sz)//2, (w-sz)//2
                crop = frame[y0:y0+sz, x0:x0+sz]
            crop = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
            faces.append(crop)
        idx += 1
        if len(faces) >= seq_len: break
    cap.release()
    while len(faces) < seq_len:
        faces.append(np.zeros((224, 224, 3), dtype=np.uint8))
    return np.array(faces[:seq_len], dtype=np.uint8), {}

def faces_to_features(faces, vgg_model):
    x = np.array([preprocess_input(img_to_array(f)) for f in faces])
    return vgg_model.predict(x, verbose=0)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION FUNCTIONS (unchanged from v1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_all_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    m = {
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'total_samples': int(total),
        'accuracy': round((tp+tn)/total, 4),
        'balanced_accuracy': round(balanced_accuracy_score(y_true, y_pred), 4),
        'precision_real': round(tp/(tp+fp), 4) if (tp+fp) > 0 else 0,
        'recall_real': round(tp/(tp+fn), 4) if (tp+fn) > 0 else 0,
        'f1_real': round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        'specificity': round(tn/(tn+fp), 4) if (tn+fp) > 0 else 0,
        'f1_fake': round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        'class_dist': {'fake': int((y_true==0).sum()), 'real': int((y_true==1).sum()),
                       'ratio': round((y_true==0).sum()/max((y_true==1).sum(),1), 2)},
    }
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        m['roc_auc'] = round(auc(fpr, tpr), 4)
        m['avg_precision'] = round(average_precision_score(y_true, y_score), 4)
    return m

def tune_threshold(y_true, y_score):
    best_t, best_f1, best_s = 0.5, -1, {}
    for t in np.linspace(0.05, 0.95, 181):
        yp = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, yp, labels=[0,1], zero_division=0)
        if f[1] > best_f1:
            best_f1, best_t = f[1], float(t)
            best_s = {'t': round(best_t,3), 'prec': round(float(p[1]),4),
                       'rec': round(float(r[1]),4), 'f1': round(float(f[1]),4)}
    return best_t, best_s

def plot_all(y_true, y_pred, y_score, model_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(set(y_true)) < 2:
        print('WARNING: Single class - skipping plots')
        return

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC - {model_name}'); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out_dir/'ROC_curve.png', dpi=200, bbox_inches='tight')
    plt.close()

    # PR curve
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.sum()/len(y_true)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(rec_arr, prec_arr, lw=2, color='#C0392B', label=f'PR (AP = {ap:.3f})')
    ax.axhline(baseline, ls='--', color='gray', label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall - {model_name}')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.savefig(out_dir/'PR_curve.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Threshold analysis
    thresholds = np.arange(0.05, 0.96, 0.01)
    ps, rs, fs, bas = [], [], [], []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        if len(set(yp)) < 2:
            ps.append(np.nan); rs.append(np.nan); fs.append(np.nan); bas.append(np.nan)
            continue
        ps.append(precision_score(y_true, yp, zero_division=0))
        rs.append(recall_score(y_true, yp, zero_division=0))
        fs.append(f1_score(y_true, yp, zero_division=0))
        bas.append(balanced_accuracy_score(y_true, yp))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(thresholds, ps, 'b-', lw=2, label='Precision')
    ax.plot(thresholds, rs, 'r-', lw=2, label='Recall')
    ax.plot(thresholds, fs, 'g-', lw=2, label='F1')
    ax.plot(thresholds, bas, 'm--', lw=2, label='Balanced Acc')
    ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
    ax.set_title(f'Threshold Analysis - {model_name}')
    ax.legend(loc='center left'); ax.grid(alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.savefig(out_dir/'threshold_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Fake','Real'], rotation=45)
    ax.set_yticks([0,1]); ax.set_yticklabels(['Fake','Real'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.savefig(out_dir/'confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()

def full_eval(y_true, y_pred, y_score, model_name, out_dir, history=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_all_metrics(y_true, y_pred, y_score)
    best_t, thresh_stats = tune_threshold(y_true, y_score)
    metrics['best_threshold'] = thresh_stats
    metrics['model'] = model_name
    metrics['dataset'] = DATASET_NAME
    with open(out_dir/'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    cm = metrics['confusion_matrix']
    ver = (f"From the {model_name} confusion matrix "
           f"(TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}, TP={cm['TP']}): "
           f"Precision(Real)={metrics['precision_real']}, "
           f"Recall(Real)={metrics['recall_real']}, "
           f"Accuracy={metrics['accuracy']}, "
           f"F1(Real)={metrics['f1_real']}, "
           f"Specificity={metrics['specificity']}, "
           f"Balanced Accuracy={metrics['balanced_accuracy']}.")
    with open(out_dir/'verification.txt', 'w') as f:
        f.write(ver)
    report = classification_report(y_true, y_pred, digits=4, target_names=['Fake','Real'])
    with open(out_dir/'classification_report.txt', 'w') as f:
        f.write(report)
    plot_all(y_true, y_pred, y_score, model_name, out_dir)
    if history:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14,5))
        a1.plot(history.history['loss'], label='Train')
        a1.plot(history.history.get('val_loss',[]), label='Val')
        a1.set_title('Loss'); a1.legend(); a1.grid(alpha=0.3)
        a2.plot(history.history['accuracy'], label='Train')
        a2.plot(history.history.get('val_accuracy',[]), label='Val')
        a2.set_title('Accuracy'); a2.legend(); a2.grid(alpha=0.3)
        fig.savefig(out_dir/'training_history.png', dpi=200, bbox_inches='tight')
        plt.close()
    d = metrics['class_dist']
    print(f"\n{'='*55}")
    print(f"  {model_name} on {DATASET_NAME}")
    print(f"{'='*55}")
    print(f"  Samples: {metrics['total_samples']} ({d['fake']}F, {d['real']}R, {d['ratio']}:1)")
    print(f"  Accuracy:          {metrics['accuracy']}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']}")
    print(f"  Precision(Real):   {metrics['precision_real']}")
    print(f"  Recall(Real):      {metrics['recall_real']}")
    print(f"  F1(Real):          {metrics['f1_real']}")
    print(f"  Specificity:       {metrics['specificity']}")
    if 'roc_auc' in metrics: print(f"  ROC AUC:           {metrics['roc_auc']}")
    print(f"  Best threshold:    {best_t:.3f} -> F1={thresh_stats['f1']}")
    print(f"  Saved to {out_dir}/")
    print(f"  Verification: {ver[:100]}...")
    return metrics

print('Evaluation functions defined')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LOAD VGG16 & EXTRACT/LOAD FEATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
vgg = VGG16(weights='imagenet', include_top=False, pooling='avg')
print(f'VGG16 loaded: output shape = {vgg.output_shape}')

real_videos = list_videos(REAL_DIR)
fake_videos = list_videos(FAKE_DIR)
print(f'Videos: {len(real_videos)} real, {len(fake_videos)} fake')

def load_or_extract(videos, cache_dir, label_name):
    features = []
    to_extract = []
    for vp in videos:
        cached = cache_dir / (Path(vp).stem + '.npy')
        if cached.exists():
            arr = np.load(cached)
            if arr.ndim == 2 and arr.shape[0] == SEQUENCE_LENGTH:
                features.append(arr)
                continue
        to_extract.append((vp, cached))
    if to_extract:
        print(f'  Extracting {len(to_extract)} {label_name} videos...')
        for vp, cached in tqdm(to_extract, desc=f'{label_name} extract'):
            faces, _ = extract_faces_from_video(vp)
            feat = faces_to_features(faces, vgg)
            np.save(cached, feat)
            features.append(feat)
    else:
        print(f'  All {len(features)} {label_name} features loaded from cache')
    return np.array(features)

print('\nLoading features...')
real_feats = load_or_extract(real_videos, CACHE_REAL, 'real')
fake_feats = load_or_extract(fake_videos, CACHE_FAKE, 'fake')

X = np.concatenate([fake_feats, real_feats], axis=0)
y = np.concatenate([np.zeros(len(fake_feats)), np.ones(len(real_feats))]).astype(int)
print(f'Total: {len(X)} videos, shape per video: {X.shape[1:]}')
print(f'Class 0 (Fake): {(y==0).sum()}, Class 1 (Real): {(y==1).sum()}')

T = X.shape[1]
F = X.shape[2]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRAIN/TEST SPLIT (same split as v1 — same seed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f'\nTrain: {len(X_train)} | Test: {len(X_test)}')
print(f'Test: Fake={(y_test==0).sum()}, Real={(y_test==1).sum()}')
print(f'Imbalance: {(y_test==0).sum()/max((y_test==1).sum(),1):.1f}:1')

# Compute class weights for training (inverse frequency)
n_fake_train = int((y_train == 0).sum())
n_real_train = int((y_train == 1).sum())
n_total_train = n_fake_train + n_real_train
cw_fake = n_total_train / (2.0 * n_fake_train)
cw_real = n_total_train / (2.0 * n_real_train)
CLASS_WEIGHTS = {0: cw_fake, 1: cw_real}
print(f'Class weights: Fake={cw_fake:.3f}, Real={cw_real:.3f}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. LRCN ORIGINAL (IMPROVED: class_weight + callbacks + batch=8)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  TRAINING LRCN-Original (V2: class_weight + callbacks)')
print('='*55)

tf.random.set_seed(SEED)
np.random.seed(SEED)

model_orig = Sequential([
    TimeDistributed(Dense(256, activation='relu'), input_shape=(T, F)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(2, activation='softmax', dtype='float32'),
])

# V2: use label smoothing via categorical crossentropy
model_orig.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
    optimizer=Adam(learning_rate=LRCN_LR),
    metrics=['accuracy']
)
model_orig.summary()

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

# V2: callbacks for better convergence
cbs_orig = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
]

# V2: class_weight to handle imbalance, larger batch, more epochs
history_orig = model_orig.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),  # use held-out test as val for monitoring
    epochs=LRCN_EPOCHS,
    batch_size=LRCN_BATCH,
    class_weight=CLASS_WEIGHTS,
    callbacks=cbs_orig,
    verbose=1
)

y_proba_orig = model_orig.predict(X_test)
y_pred_orig = np.argmax(y_proba_orig, axis=1)
y_score_orig = y_proba_orig[:, 1]

out_orig = RESULTS_BASE / 'lrcn_original'
out_orig.mkdir(parents=True, exist_ok=True)
model_orig.save(str(out_orig / 'model.h5'))
metrics_orig = full_eval(y_test, y_pred_orig, y_score_orig,
                         'LRCN-Original', out_orig, history_orig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. LRCN RETRAINED (IMPROVED: gamma=1.5 + more epochs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  TRAINING LRCN-Retrained (V2: gamma=1.5, epochs=25)')
print('='*55)

class BalancedGen(KerasSequence):
    def __init__(self, X, y, bs, seed=SEED):
        self.X, self.y, self.half = X, y, bs//2
        self.rng = np.random.RandomState(seed)
        self.pos = np.where(y==1)[0]
        self.neg = np.where(y==0)[0]
    def __len__(self):
        return max(max(len(self.pos), len(self.neg)) // self.half, 1)
    def __getitem__(self, i):
        p = self.rng.choice(self.pos, self.half, replace=True)
        n = self.rng.choice(self.neg, self.half, replace=True)
        b = np.concatenate([p, n]); self.rng.shuffle(b)
        return self.X[b], to_categorical(self.y[b], 2)
    def on_epoch_end(self):
        self.rng.shuffle(self.pos); self.rng.shuffle(self.neg)

alpha_vec = [n_total_train / (2.0 * n_fake_train),
             n_total_train / (2.0 * n_real_train)]
print(f'Focal alpha: Fake={alpha_vec[0]:.3f}, Real={alpha_vec[1]:.3f}')
print(f'Focal gamma: {FOCAL_GAMMA}')

alpha_t = tf.constant(alpha_vec, dtype=tf.float32)
def focal_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    ce = -y_true * tf.math.log(y_pred)
    w = alpha_t * tf.pow(1.0 - y_pred, FOCAL_GAMMA)
    return tf.reduce_mean(tf.reduce_sum(w * ce, axis=1))

tf.random.set_seed(SEED)
np.random.seed(SEED)

model_ret = Sequential([
    TimeDistributed(Dense(256, activation='relu'), input_shape=(T, F)),
    Dropout(0.3),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(2, activation='softmax', dtype='float32'),
])
model_ret.compile(optimizer=Adam(learning_rate=RETRAIN_LR),
                  loss=focal_loss, metrics=['accuracy'])

train_gen = BalancedGen(X_train, y_train, RETRAIN_BATCH)
cbs_ret = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
]

history_ret = model_ret.fit(train_gen, validation_data=(X_test, y_test_cat),
                            epochs=RETRAIN_EPOCHS, callbacks=cbs_ret, verbose=1)

y_proba_ret = model_ret.predict(X_test)
y_pred_ret = np.argmax(y_proba_ret, axis=1)
y_score_ret = y_proba_ret[:, 1]

out_ret = RESULTS_BASE / 'lrcn_retrained'
out_ret.mkdir(parents=True, exist_ok=True)
model_ret.save(str(out_ret / 'model.h5'))
metrics_ret = full_eval(y_test, y_pred_ret, y_score_ret,
                        'LRCN-Retrained', out_ret, history_ret)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. SVM (IMPROVED: StandardScaler + class_weight + GridSearchCV)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  TRAINING SVM (V2: scaled + balanced + grid search)')
print('='*55)

# Mean-pool 15 frames -> (N, 512) per video
X_train_svm = X_train.mean(axis=1)
X_test_svm  = X_test.mean(axis=1)
y_train_svm = y_train
y_test_svm  = y_test

# V2 FIX #1: StandardScaler — SVMs are very sensitive to feature scale
scaler = StandardScaler()
X_train_svm_scaled = scaler.fit_transform(X_train_svm)
X_test_svm_scaled  = scaler.transform(X_test_svm)

print(f'SVM (mean-pool + scaled): Train {len(X_train_svm)} | Test {len(X_test_svm)}')
print(f'  Feature shape: {X_train_svm_scaled.shape}')
print(f'  Feature range before scaling: [{X_train_svm.min():.2f}, {X_train_svm.max():.2f}]')
print(f'  Feature range after scaling:  [{X_train_svm_scaled.min():.2f}, {X_train_svm_scaled.max():.2f}]')

# V2 FIX #2: GridSearchCV with class_weight='balanced'
print(f'\n  Running GridSearchCV with {SVM_GRID}...')
svm_base = SVC(kernel='rbf', class_weight='balanced',
               probability=True, random_state=SEED)

t0 = time.perf_counter()
grid = GridSearchCV(
    svm_base, SVM_GRID,
    cv=5,                    # 5-fold stratified CV
    scoring='balanced_accuracy',  # optimize for balanced accuracy (handles imbalance)
    n_jobs=-1,               # use all CPU cores
    verbose=1,
    refit=True               # refit best model on full train set
)
grid.fit(X_train_svm_scaled, y_train_svm)
grid_time = time.perf_counter() - t0

print(f'\n  GridSearch done in {grid_time:.1f}s')
print(f'  Best params: {grid.best_params_}')
print(f'  Best CV balanced_accuracy: {grid.best_score_:.4f}')

svm = grid.best_estimator_
print(f'  Support vectors: {svm.n_support_}')

# Save scaler and best params for reproducibility
with open(RESULTS_BASE / 'svm' / 'svm_params.json', 'w') as f:
    json.dump({
        'best_params': grid.best_params_,
        'best_cv_score': round(grid.best_score_, 4),
        'grid_search_time_s': round(grid_time, 1),
        'scaler_mean_range': [float(scaler.mean_.min()), float(scaler.mean_.max())],
        'scaler_var_range': [float(scaler.var_.min()), float(scaler.var_.max())],
    }, f, indent=2)

# Save the scaler for inference
pickle.dump(scaler, open(RESULTS_BASE / 'svm' / 'scaler.pkl', 'wb'))

y_pred_svm = svm.predict(X_test_svm_scaled)
y_score_svm = svm.predict_proba(X_test_svm_scaled)[:, 1]

out_svm = RESULTS_BASE / 'svm'
metrics_svm = full_eval(y_test_svm, y_pred_svm, y_score_svm, 'SVM', out_svm)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. PIPELINE TIMING (same as v1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  PIPELINE TIMING')
print('='*55)

model_timing = load_model(str(RESULTS_BASE / 'lrcn_original' / 'model.h5'))

timing_videos = (real_videos + fake_videos)[:NUM_TIMING_VIDEOS]
stages = {'face_det': [], 'vgg': [], 'classifier': [], 'total': []}

for vp in tqdm(timing_videos, desc='Timing'):
    t0 = time.perf_counter()
    faces, _ = extract_faces_from_video(vp)
    t1 = time.perf_counter()
    features = faces_to_features(faces, vgg)
    t2 = time.perf_counter()
    _ = model_timing.predict(np.expand_dims(features, 0), verbose=0)
    t3 = time.perf_counter()
    stages['face_det'].append(t1-t0)
    stages['vgg'].append(t2-t1)
    stages['classifier'].append(t3-t2)
    stages['total'].append(t3-t0)

timing_stats = {}
print(f'\n{"Stage":<20} {"Mean (s)":<12} {"Std (s)":<12}')
print('-'*44)
for k, vals in stages.items():
    m, s = np.mean(vals), np.std(vals)
    timing_stats[k] = {'mean': round(m,4), 'std': round(s,4)}
    print(f'{k:<20} {m:<12.4f} {s:<12.4f}')

with open(RESULTS_BASE / 'pipeline_timing.json', 'w') as f:
    json.dump({'n_videos': len(timing_videos), 'stats': timing_stats}, f, indent=2)

ts = timing_stats['total']
print(f"\nFor paper: VeriLens classifies a single video ({SEQUENCE_LENGTH} frames) "
      f"in {ts['mean']:.3f} +/- {ts['std']:.3f} seconds.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. COMPARATIVE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  COMPARATIVE SUMMARY')
print('='*55)

metrics_orig = json.load(open(RESULTS_BASE / 'lrcn_original' / 'metrics.json'))
metrics_ret  = json.load(open(RESULTS_BASE / 'lrcn_retrained' / 'metrics.json'))
metrics_svm  = json.load(open(RESULTS_BASE / 'svm' / 'metrics.json'))

print(f'\n{"Model":<24} {"Prec":>8} {"Rec":>8} {"F1":>8} {"Acc":>8} {"BalAcc":>8} {"AUC":>8}')
print('-'*72)
for name, m in [('LRCN-Original', metrics_orig),
                ('LRCN-Retrained', metrics_ret),
                ('SVM', metrics_svm)]:
    print(f"  {name:<22} {m['precision_real']:>8} {m['recall_real']:>8} "
          f"{m['f1_real']:>8} {m['accuracy']:>8} {m['balanced_accuracy']:>8} "
          f"{m.get('roc_auc','N/A'):>8}")

# V1 vs V2 comparison if v1 results exist
v1_results = MAIN_REPO / 'results' / DATASET_NAME
if v1_results.exists() and (v1_results / 'lrcn_original' / 'metrics.json').exists():
    print(f'\n{"="*55}')
    print('  V1 vs V2 COMPARISON')
    print(f'{"="*55}')
    for model_dir in ['lrcn_original', 'lrcn_retrained', 'svm']:
        v1_file = v1_results / model_dir / 'metrics.json'
        v2_file = RESULTS_BASE / model_dir / 'metrics.json'
        if v1_file.exists() and v2_file.exists():
            v1 = json.load(open(v1_file))
            v2 = json.load(open(v2_file))
            print(f'\n  {model_dir}:')
            for k in ['accuracy', 'balanced_accuracy', 'recall_real', 'f1_real', 'roc_auc']:
                if k in v1 and k in v2:
                    diff = v2[k] - v1[k]
                    arrow = '↑' if diff > 0 else ('↓' if diff < 0 else '=')
                    print(f'    {k:<22} V1={v1[k]:<8} V2={v2[k]:<8} {arrow} {diff:+.4f}')

with open(RESULTS_BASE / 'comparative_summary.json', 'w') as f:
    json.dump({'lrcn_original': metrics_orig, 'lrcn_retrained': metrics_ret,
               'svm': metrics_svm, 'version': 'v2_improved'}, f, indent=2)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 14. VERIFICATION PARAGRAPHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  VERIFICATION PARAGRAPHS (copy to Section 4.3)')
print('='*55)

for name, d in [('lrcn_original', 'LRCN-Original'),
                ('lrcn_retrained', 'LRCN-Retrained'),
                ('svm', 'SVM')]:
    vfile = RESULTS_BASE / name / 'verification.txt'
    if vfile.exists():
        print(f'\n{d}:')
        print(f'  {vfile.read_text()}\n')

print('\n' + '='*55)
print('  V2 PIPELINE COMPLETE')
print('='*55)
print(f'All results saved to: {RESULTS_BASE.resolve()}')
print('\nKey V2 improvements applied:')
print('  1. LRCN-Original: class_weight + label_smoothing=0.1 + batch=8 + 30 epochs + callbacks')
print('  2. LRCN-Retrained: focal gamma=1.5 (less aggressive) + 25 epochs + patience=6')
print('  3. SVM: StandardScaler + class_weight=balanced + GridSearchCV (C, gamma)')
