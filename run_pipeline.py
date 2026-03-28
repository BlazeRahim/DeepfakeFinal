"""
VeriLens Complete Pipeline Runner
Runs the full evaluation pipeline: LRCN-Original, LRCN-Retrained, SVM (mean-pooled)
Generates all reviewer-requested outputs: metrics, plots, verification paragraphs
"""
import os, sys

# === GPU/cuDNN fix for conda env on Windows ===
os.environ['PATH'] = r'C:\Users\blaze\miniconda3\envs\deepfake\Library\bin' + ';' + os.environ.get('PATH', '')
os.add_dll_directory(r'C:\Users\blaze\miniconda3\envs\deepfake\Library\bin')

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

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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
DATASET_NAME = 'celeb-df'
# Use main repo paths (where data and cache live)
MAIN_REPO = Path(r'C:\Users\blaze\Desktop\DeepfakeFinal')
REAL_DIR = str(MAIN_REPO / 'data' / 'Celeb-real')
FAKE_DIR = str(MAIN_REPO / 'data' / 'Celeb-synthesis')

SEQUENCE_LENGTH = 15
FRAME_STRIDE = 15
CACHE_DIR = MAIN_REPO / f'cache/{DATASET_NAME}/seq{SEQUENCE_LENGTH}_stride{FRAME_STRIDE}'
CACHE_REAL = CACHE_DIR / 'real'
CACHE_FAKE = CACHE_DIR / 'fake'
CACHE_REAL.mkdir(parents=True, exist_ok=True)
CACHE_FAKE.mkdir(parents=True, exist_ok=True)

# Output results to worktree
RESULTS_BASE = Path(f'results/{DATASET_NAME}')

# LRCN Original
LRCN_EPOCHS = 10
LRCN_BATCH = 2
LRCN_LR = 1e-4

# LRCN Retrained
RETRAIN_EPOCHS = 15
RETRAIN_BATCH = 32
RETRAIN_LR = 1e-4
FOCAL_GAMMA = 2.0

NUM_TIMING_VIDEOS = 10
T = SEQUENCE_LENGTH
F_DIM = 512  # VGG16 avg pool output

print(f'Dataset: {DATASET_NAME}')
print(f'Real: {REAL_DIR} | Fake: {FAKE_DIR}')
print(f'Cache: {CACHE_DIR}')
print(f'Results: {RESULTS_BASE}')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CORE FUNCTIONS
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
# EVALUATION FUNCTIONS
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
# TRAIN/TEST SPLIT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f'\nTrain: {len(X_train)} | Test: {len(X_test)}')
print(f'Test: Fake={(y_test==0).sum()}, Real={(y_test==1).sum()}')
print(f'Imbalance: {(y_test==0).sum()/max((y_test==1).sum(),1):.1f}:1')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. LRCN ORIGINAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  TRAINING LRCN-Original')
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
model_orig.compile(loss='categorical_crossentropy',
                   optimizer=Adam(learning_rate=LRCN_LR), metrics=['accuracy'])
model_orig.summary()

y_train_cat = to_categorical(y_train, 2)
y_test_cat = to_categorical(y_test, 2)

history_orig = model_orig.fit(X_train, y_train_cat, validation_split=0.2,
                              epochs=LRCN_EPOCHS, batch_size=LRCN_BATCH, verbose=1)

y_proba_orig = model_orig.predict(X_test)
y_pred_orig = np.argmax(y_proba_orig, axis=1)
y_score_orig = y_proba_orig[:, 1]

out_orig = RESULTS_BASE / 'lrcn_original'
out_orig.mkdir(parents=True, exist_ok=True)
model_orig.save(str(out_orig / 'model.h5'))
metrics_orig = full_eval(y_test, y_pred_orig, y_score_orig,
                         'LRCN-Original', out_orig, history_orig)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. LRCN RETRAINED (Focal Loss + Balanced Batches)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  TRAINING LRCN-Retrained (Focal Loss)')
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

n0 = int((y_train==0).sum())
n1 = int((y_train==1).sum())
alpha_vec = [(n0+n1)/(2.0*n0), (n0+n1)/(2.0*n1)]
print(f'Focal alpha: Fake={alpha_vec[0]:.3f}, Real={alpha_vec[1]:.3f}')

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
cbs = [
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
]

history_ret = model_ret.fit(train_gen, validation_data=(X_test, y_test_cat),
                            epochs=RETRAIN_EPOCHS, callbacks=cbs, verbose=1)

y_proba_ret = model_ret.predict(X_test)
y_pred_ret = np.argmax(y_proba_ret, axis=1)
y_score_ret = y_proba_ret[:, 1]

out_ret = RESULTS_BASE / 'lrcn_retrained'
out_ret.mkdir(parents=True, exist_ok=True)
model_ret.save(str(out_ret / 'model.h5'))
metrics_ret = full_eval(y_test, y_pred_ret, y_score_ret,
                        'LRCN-Retrained', out_ret, history_ret)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. SVM (per-video, mean-pooled)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print('\n' + '='*55)
print('  TRAINING SVM (mean-pooled)')
print('='*55)

# Mean-pool: average 15 VGG16 feature vectors per video -> (N, 512)
X_train_svm = X_train.mean(axis=1)
X_test_svm  = X_test.mean(axis=1)
y_train_svm = y_train
y_test_svm  = y_test

print(f'SVM (mean-pool): Train {len(X_train_svm)} | Test {len(X_test_svm)}')
print(f'  Feature shape: {X_train_svm.shape}')

svm = SVC(kernel='rbf', C=1.0, gamma='scale',
          probability=True, random_state=SEED)
t0 = time.perf_counter()
svm.fit(X_train_svm, y_train_svm)
print(f'Trained in {time.perf_counter()-t0:.1f}s | SVs: {svm.n_support_}')

y_pred_svm = svm.predict(X_test_svm)
y_score_svm = svm.predict_proba(X_test_svm)[:, 1]

out_svm = RESULTS_BASE / 'svm'
metrics_svm = full_eval(y_test_svm, y_pred_svm, y_score_svm, 'SVM', out_svm)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. PIPELINE TIMING (R#4 Point 7)
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
for name, m in [('LRCN-Original', metrics_orig),
                ('LRCN-Retrained', metrics_ret),
                ('SVM', metrics_svm)]:
    print(f"  {name:<22} {m['precision_real']:>8} {m['recall_real']:>8} "
          f"{m['f1_real']:>8} {m['accuracy']:>8} {m['balanced_accuracy']:>8} "
          f"{m.get('roc_auc','N/A'):>8}")

with open(RESULTS_BASE / 'comparative_summary.json', 'w') as f:
    json.dump({'lrcn_original': metrics_orig, 'lrcn_retrained': metrics_ret,
               'svm': metrics_svm}, f, indent=2)

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
print('  PIPELINE COMPLETE')
print('='*55)
print(f'All results saved to: {RESULTS_BASE.resolve()}')
