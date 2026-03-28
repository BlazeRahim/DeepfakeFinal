"""Run SVM per-frame evaluation matching original Final.py protocol.

Subsamples 5 evenly-spaced frames per video from the 15-frame sequences
to keep training feasible for RBF SVM. Tests on all 15 frames per video.
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             balanced_accuracy_score, f1_score, precision_score,
                             recall_score, precision_recall_fscore_support)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

os.chdir(r'C:\Users\blaze\Desktop\DeepfakeFinal\.claude\worktrees\amazing-heisenberg')

SEED = 42
SEQUENCE_LENGTH = 15
TRAIN_FRAMES = [0, 3, 7, 11, 14]  # 5 evenly-spaced frames for training
DATASET_NAME = 'celeb-df'
RESULTS_BASE = Path(f'results/{DATASET_NAME}')
CACHE_DIR = Path(r'C:\Users\blaze\Desktop\DeepfakeFinal\cache') / DATASET_NAME / f'seq{SEQUENCE_LENGTH}_stride{SEQUENCE_LENGTH}'

# ── Helper functions ─────────────────────────────────────────────

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
        'class_dist': {'fake': int((np.array(y_true)==0).sum()),
                       'real': int((np.array(y_true)==1).sum()),
                       'ratio': round(float((np.array(y_true)==0).sum())/max(float((np.array(y_true)==1).sum()),1), 2)},
    }
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        m['roc_auc'] = round(auc(fpr, tpr), 4)
        m['avg_precision'] = round(average_precision_score(y_true, y_score), 4)
    return m

def tune_threshold(y_true, y_score):
    y_score = np.array(y_score)
    lo, hi = y_score.min(), y_score.max()
    thresholds = np.linspace(lo + 0.01*(hi-lo), hi - 0.01*(hi-lo), 181)
    best_t, best_f1, best_s = float(np.median(y_score)), -1, {}
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, yp, labels=[0,1], zero_division=0)
        if f[1] > best_f1:
            best_f1, best_t = f[1], float(t)
            best_s = {'t': round(best_t, 4), 'prec': round(float(p[1]),4),
                       'rec': round(float(r[1]),4), 'f1': round(float(f[1]),4)}
    return best_t, best_s

def plot_all(y_true, y_pred, y_score, model_name, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.array(y_true); y_score = np.array(y_score)
    if len(set(y_true)) < 2:
        print('Single class — skipping plots'); return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC - {model_name}'); ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(out_dir/'ROC_curve.png', dpi=200, bbox_inches='tight'); plt.close()

    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = y_true.sum()/len(y_true)
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(rec_arr, prec_arr, lw=2, color='#C0392B', label=f'PR (AP = {ap:.3f})')
    ax.axhline(baseline, ls='--', color='gray', label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall - {model_name}')
    ax.legend(); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.savefig(out_dir/'PR_curve.png', dpi=200, bbox_inches='tight'); plt.close()

    lo, hi = y_score.min(), y_score.max()
    thresholds = np.linspace(lo, hi, 91)
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
    ax.set_xlabel('Threshold (decision function)'); ax.set_ylabel('Score')
    ax.set_title(f'Threshold Analysis - {model_name}')
    ax.legend(loc='center left'); ax.grid(alpha=0.3); ax.set_ylim(0,1)
    fig.savefig(out_dir/'threshold_analysis.png', dpi=200, bbox_inches='tight'); plt.close()

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {model_name}'); plt.colorbar(im)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Fake','Real'], rotation=45)
    ax.set_yticks([0,1]); ax.set_yticklabels(['Fake','Real'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    fig.savefig(out_dir/'confusion_matrix.png', dpi=200, bbox_inches='tight'); plt.close()

def full_eval(y_true, y_pred, y_score, model_name, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
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
    print(f"  Best threshold:    {best_t:.4f} -> F1={thresh_stats['f1']}")
    print(f"  Saved to {out_dir}/")
    return metrics

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
print('Loading cached features...', flush=True)
real_dir = CACHE_DIR / 'real'
fake_dir = CACHE_DIR / 'fake'

X_real, X_fake = [], []
for f in sorted(real_dir.glob('*.npy')):
    arr = np.load(f)
    if arr.ndim == 2 and arr.shape[0] == SEQUENCE_LENGTH:
        X_real.append(arr)
for f in sorted(fake_dir.glob('*.npy')):
    arr = np.load(f)
    if arr.ndim == 2 and arr.shape[0] == SEQUENCE_LENGTH:
        X_fake.append(arr)

X_real = np.stack(X_real)
X_fake = np.stack(X_fake)
y_real = np.ones(len(X_real), dtype=np.int32)
y_fake = np.zeros(len(X_fake), dtype=np.int32)
X = np.concatenate([X_real, X_fake])
y = np.concatenate([y_real, y_fake])
print(f'Total: {X.shape} | Real={len(X_real)}, Fake={len(X_fake)}', flush=True)

# Split at VIDEO level (same seed as notebook)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f'Video split: Train={len(X_train)} | Test={len(X_test)}', flush=True)

# ── SVM Training: subsample 5 frames per video for training ──
X_train_sub = X_train[:, TRAIN_FRAMES, :]  # (N, 5, 512)
X_train_svm = X_train_sub.reshape(-1, X_train_sub.shape[-1])  # (N*5, 512)
y_train_svm = np.repeat(y_train, len(TRAIN_FRAMES))

# ── SVM Testing: use ALL 15 frames per video ──
X_test_svm = X_test.reshape(-1, X_test.shape[-1])  # (N*15, 512)
y_test_svm = np.repeat(y_test, SEQUENCE_LENGTH)

print(f'\nSVM: Train={len(X_train_svm)} frames ({len(TRAIN_FRAMES)}/video)', flush=True)
print(f'SVM: Test ={len(X_test_svm)} frames ({SEQUENCE_LENGTH}/video)', flush=True)
print(f'  Train: Fake={(y_train_svm==0).sum()}, Real={(y_train_svm==1).sum()}', flush=True)
print(f'  Test:  Fake={(y_test_svm==0).sum()}, Real={(y_test_svm==1).sum()}', flush=True)

# ── Train ──
print('\nTraining SVM (RBF, C=1.0, gamma=scale)...', flush=True)
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=SEED)
t0 = time.perf_counter()
svm.fit(X_train_svm, y_train_svm)
train_time = time.perf_counter()-t0
print(f'Trained in {train_time:.1f}s | SVs: {svm.n_support_}', flush=True)

# ── Per-frame predictions ──
print('\nPredicting on test frames...', flush=True)
y_pred_svm  = svm.predict(X_test_svm)
y_score_svm = svm.decision_function(X_test_svm)
print(f'Predicted: Real={(y_pred_svm==1).sum()}, Fake={(y_pred_svm==0).sum()}', flush=True)

# ── Per-frame evaluation ──
out_svm = RESULTS_BASE / 'svm'
print(f'\n{"="*60}')
print(f'  SVM PER-FRAME EVALUATION (N={len(y_test_svm)})')
print(f'{"="*60}', flush=True)
metrics_svm = full_eval(y_test_svm, y_pred_svm, y_score_svm, 'SVM', out_svm)

# ── Calibrate + aggregate to per-video ──
print('\nCalibrating probabilities (Platt scaling)...', flush=True)
cal_svm = CalibratedClassifierCV(svm, method='sigmoid', cv='prefit')
cal_svm.fit(X_train_svm, y_train_svm)
y_proba_svm = cal_svm.predict_proba(X_test_svm)[:, 1]

y_proba_vid = y_proba_svm.reshape(-1, SEQUENCE_LENGTH).mean(axis=1)
y_pred_vid  = (y_proba_vid >= 0.5).astype(int)

out_svm_vid = RESULTS_BASE / 'svm_per_video'
print(f'\n{"="*60}')
print(f'  SVM PER-VIDEO AGGREGATION (N={len(y_test)}, mean over {SEQUENCE_LENGTH} frames)')
print(f'{"="*60}', flush=True)
metrics_svm_vid = full_eval(y_test, y_pred_vid, y_proba_vid, 'SVM (per-video)', out_svm_vid)

# ── Save models ──
joblib.dump(svm, str(out_svm / 'svm_model.pkl'))
joblib.dump(cal_svm, str(out_svm / 'svm_calibrated.pkl'))
print(f'\nSVM models saved to {out_svm}/')

# ── Comparative summary ──
metrics_orig = json.load(open(RESULTS_BASE / 'lrcn_original' / 'metrics.json'))
metrics_ret = json.load(open(RESULTS_BASE / 'lrcn_retrained' / 'metrics.json'))

print(f"\n{'='*80}")
print(f"  COMPARATIVE SUMMARY — {DATASET_NAME.upper()}")
print(f"{'='*80}")
header = f"  {'Model':<28} {'Eval':<10} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Acc':<8} {'BalAcc':<8} {'AUC':<8}"
print(f'\n{header}')
print('  ' + '-'*len(header))
for name, m, ev in [
    ('LRCN-Original',    metrics_orig,    'per-vid'),
    ('LRCN-Retrained',   metrics_ret,     'per-vid'),
    ('SVM (per-frame)',   metrics_svm,     'per-frm'),
    ('SVM (per-video)',   metrics_svm_vid, 'per-vid'),
]:
    print(f"  {name:<28} {ev:<10} "
          f"{m['precision_real']:<8} "
          f"{m['recall_real']:<8} "
          f"{m['f1_real']:<8} "
          f"{m['accuracy']:<8} "
          f"{m['balanced_accuracy']:<8} "
          f"{m.get('roc_auc','N/A'):<8}")

print(f"\n  LRCN: per-video ({metrics_orig['total_samples']} sequences)")
print(f"  SVM per-frame: {metrics_svm['total_samples']} test frames "
      f"(trained on {len(TRAIN_FRAMES)} frames/video, tested on {SEQUENCE_LENGTH} frames/video)")
print(f"  SVM per-video: mean calibrated probability over {SEQUENCE_LENGTH} frames")

with open(RESULTS_BASE / 'comparative_summary.json', 'w') as f:
    json.dump({
        'lrcn_original': metrics_orig,
        'lrcn_retrained': metrics_ret,
        'svm_per_frame': metrics_svm,
        'svm_per_video': metrics_svm_vid,
    }, f, indent=2)

print(f'\nAll results in {RESULTS_BASE}/')
print('ALL DONE.', flush=True)
