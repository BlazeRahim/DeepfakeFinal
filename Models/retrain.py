# retrain.py
# Retrain LRCN on cached VGG16 features with:
# - Balanced mini-batches (oversample minority each batch)
# - Focal loss (focus on hard/minority samples)
# - Threshold tuning for the Real class
# - Mixed precision safe final layer
# Saves: model_retrained.h5  and  model_retrained.threshold.txt

import os
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

# ----------------------------
# Config
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CACHE_ROOT = Path("cache/features/seq15_stride15_vgg16avg")  # <-- uses your cache folder
REAL_DIR = CACHE_ROOT / "real"
FAKE_DIR = CACHE_ROOT / "fake"

SEQUENCE_LENGTH = 15                 # should match your cache (T)
BATCH_SIZE = 32                      # per-step batch size (balanced within each batch)
EPOCHS = 15
LEARNING_RATE = 1e-4
MODEL_OUT = "model_retrained.h5"
THRESH_OUT = "model_retrained.threshold.txt"

# Mixed precision
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled")
except Exception:
    print("ℹ️ Mixed precision not available; continuing in float32.")

# Headless plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical, Sequence


# ----------------------------
# Small helpers
# ----------------------------
def banner(msg: str):
    print("\n" + "="*80)
    print(msg)
    print("="*80)

def load_class_folder(folder: Path, label: int) -> Tuple[List[np.ndarray], List[int]]:
    """Load all .npy sequences from a class folder. Skip wrong shapes."""
    X_list, y_list = [], []
    for p in sorted(folder.glob("*.npy")):
        try:
            arr = np.load(p)
        except Exception as e:
            print(f"⚠️ Skipping corrupt: {p.name} ({e})")
            continue
        if arr.ndim != 2:
            print(f"⚠️ Skipping {p.name}: expected 2D (T,F), got {arr.shape}")
            continue
        if arr.shape[0] != SEQUENCE_LENGTH:
            print(f"⚠️ Skipping {p.name}: expected T={SEQUENCE_LENGTH}, got {arr.shape[0]}")
            continue
        X_list.append(arr.astype(np.float32))
        y_list.append(label)
    return X_list, y_list


def load_cached_features(cache_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load all cached sequences from real/ and fake/ into X,y."""
    if not (cache_root.exists() and (cache_root / "real").exists() and (cache_root / "fake").exists()):
        raise FileNotFoundError(f"❌ Cache root not found or missing subfolders: {cache_root}")

    X_real, y_real = load_class_folder(cache_root / "real", label=1)  # 1=Real
    X_fake, y_fake = load_class_folder(cache_root / "fake", label=0)  # 0=Fake

    if len(X_real) == 0 or len(X_fake) == 0:
        raise RuntimeError("❌ Not enough cached sequences. Need both real and fake present.")

    X = np.stack(X_real + X_fake, axis=0)   # (N, T, F)
    y = np.array(y_real + y_fake, dtype=np.int32)

    return X, y


# ----------------------------
# Balanced batch generator
# ----------------------------
class BalancedSequence(Sequence):
    """
    Yields balanced mini-batches each step: half from Real(1), half from Fake(0).
    Oversamples the minority with replacement.
    """
    def __init__(self, X: np.ndarray, y_int: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 42):
        assert batch_size % 2 == 0, "BATCH_SIZE must be even for 50/50 sampling."
        self.X = X
        self.y = y_int
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        self.pos_idx = np.where(self.y == 1)[0]  # Real
        self.neg_idx = np.where(self.y == 0)[0]  # Fake

        self.steps = int(np.ceil(max(len(self.pos_idx), len(self.neg_idx)) / self.half))

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        pos = self.rng.choice(self.pos_idx, size=self.half, replace=(len(self.pos_idx) < self.half))
        neg = self.rng.choice(self.neg_idx, size=self.half, replace=(len(self.neg_idx) < self.half))
        batch_idx = np.concatenate([pos, neg])
        if self.shuffle:
            self.rng.shuffle(batch_idx)

        Xb = self.X[batch_idx]
        yb = to_categorical(self.y[batch_idx], num_classes=2)
        return Xb, yb

    def on_epoch_end(self):
        # reshuffle base indices for variety
        if self.shuffle:
            self.rng.shuffle(self.pos_idx)
            self.rng.shuffle(self.neg_idx)


# ----------------------------
# Focal loss for softmax (2-class)
# ----------------------------
def make_focal_loss(alpha_vec, gamma: float = 2.0):
    """
    alpha_vec: length-2 list/ndarray with per-class weights [alpha_fake(0), alpha_real(1)]
    gamma: focusing parameter
    y_true must be one-hot (N,2); y_pred are softmax probs (N,2)
    """
    alpha_vec = tf.constant(alpha_vec, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        # per-class focal
        cross_entropy = -y_true_f * tf.math.log(y_pred)
        weights = alpha_vec * tf.pow(1.0 - y_pred, gamma)
        fl = tf.reduce_sum(weights * cross_entropy, axis=1)
        return tf.reduce_mean(fl)
    return loss


# ----------------------------
# Model
# ----------------------------
def build_lrcn(sequence_length: int, feature_dim: int) -> tf.keras.Model:
    model = Sequential()
    model.add(TimeDistributed(Dense(256, activation='relu'), input_shape=(sequence_length, feature_dim)))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    # ensure numeric stability under mixed precision
    model.add(Dense(2, activation='softmax', dtype='float32'))
    return model


# ----------------------------
# Metrics & plots
# ----------------------------
def plot_roc_curve(y_true_int: np.ndarray, y_score_pos: np.ndarray, out_path="ROC_curve.png"):
    classes_present = set(y_true_int.tolist())
    if classes_present == {0, 1}:
        fpr, tpr, _ = roc_curve(y_true_int, y_score_pos)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC (Real=Positive)")
        plt.legend(loc="lower right")
        plt.savefig(out_path)
        plt.close()
        print(f"📈 ROC saved -> {out_path}")
    else:
        print("⚠️ Skipping ROC: validation set has a single class.")

def plot_confusion(y_true_int: np.ndarray, y_pred_int: np.ndarray, out_path="confusion_matrix.png"):
    try:
        cm = confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        ticks = np.arange(2)
        plt.xticks(ticks, ["Fake(0)", "Real(1)"], rotation=45)
        plt.yticks(ticks, ["Fake(0)", "Real(1)"])
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


def tune_threshold_for_real(y_true_int: np.ndarray, y_score_pos: np.ndarray) -> Tuple[float, dict]:
    """
    Grid search threshold on validation set to maximize F1 for class Real(1).
    Returns best_threshold and metrics at that threshold.
    """
    best_t, best_f1, best_stats = 0.5, -1.0, {}
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_score_pos >= t).astype(np.int32)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true_int, y_pred, labels=[0, 1], zero_division=0
        )
        f1_real = f1[1]
        if f1_real > best_f1:
            best_f1 = f1_real
            best_t = float(t)
            best_stats = {
                "precision_real": float(p[1]),
                "recall_real": float(r[1]),
                "f1_real": float(f1[1]),
                "precision_fake": float(p[0]),
                "recall_fake": float(r[0]),
                "f1_fake": float(f1[0]),
            }
    return best_t, best_stats


# ----------------------------
# Main
# ----------------------------
def main():
    banner("START " + time.strftime("%H:%M:%S"))
    banner("Stage 1: Locate cache")
    print(f"✅ Using cache root: {CACHE_ROOT}")

    banner("Stage 2: Load cached feature sequences")
    X, y_int = load_cached_features(CACHE_ROOT)
    print(f"✅ Loaded X={X.shape} (N,T,F), y={y_int.shape}, classes={np.unique(y_int)}")

    T, F = X.shape[1], X.shape[2]

    banner("Stage 3: Train/Val split (stratified)")
    X_train, X_val, y_train_int, y_val_int = train_test_split(
        X, y_int, test_size=0.20, random_state=SEED, stratify=y_int
    )

    # alpha for focal loss: inverse-frequency weighting per class
    n0 = int(np.sum(y_train_int == 0))
    n1 = int(np.sum(y_train_int == 1))
    total = n0 + n1
    alpha_0 = total / (2.0 * n0)  # Fake
    alpha_1 = total / (2.0 * n1)  # Real (larger when minority)
    alpha_vec = [alpha_0, alpha_1]
    print(f"⚖️ Class counts train: fake={n0}, real={n1}")
    print(f"⚖️ Focal alpha per class [fake, real] = {alpha_vec}")

    # Generators
    train_seq = BalancedSequence(X_train, y_train_int, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    y_val_onehot = to_categorical(y_val_int, num_classes=2)

    banner("Stage 4: Build model")
    model = build_lrcn(T, F)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss=make_focal_loss(alpha_vec=alpha_vec, gamma=2.0),
                  metrics=['accuracy'])
    model.summary()

    banner("Stage 5: Train")
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_OUT, monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
    ]

    history = model.fit(
        train_seq,
        validation_data=(X_val, y_val_onehot),
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1
    )

    banner("Stage 6: Evaluate + Save")
    # Ensure we save final weights even if early stopping hit before checkpoint
    model.save(MODEL_OUT)
    print(f"💾 Model saved as {MODEL_OUT}")

    # Probabilities for Real(1)
    y_val_proba = model.predict(X_val, verbose=0)
    y_val_pos = y_val_proba[:, 1]

    # Tune threshold for Real
    best_t, stats = tune_threshold_for_real(y_val_int, y_val_pos)
    with open(THRESH_OUT, "w") as f:
        f.write(str(best_t))
    print(f"🔧 Best threshold for class 'Real'(1): {best_t:.3f}  (saved to {THRESH_OUT})")
    print(f"   Stats at best T: {json.dumps(stats, indent=2)}")

    # Reports at 0.5 and best_t
    for name, thr in [("0.50 (default)", 0.5), (f"{best_t:.2f} (tuned)", best_t)]:
        y_pred_int = (y_val_pos >= thr).astype(np.int32)
        print(f"\n=== Classification Report @ threshold {name} ===")
        print(classification_report(y_val_int, y_pred_int, digits=4))
        plot_confusion(y_val_int, y_pred_int, out_path=f"confusion_matrix_{str(thr).replace('.','p')}.png")

    plot_roc_curve(y_val_int, y_val_pos, out_path="ROC_curve_retrained.png")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
