# VeriLens Paper Corrections — Section-by-Section

All numbers below are from actual pipeline runs on RTX 3050.
Copy-paste these into the relevant paper sections.

---

## Section 3.1.4 — SVM Configuration (R#4 Point 5)

**REPLACE** the existing SVM description with:

> The SVM baseline employs an RBF kernel with regularization parameter C = 1.0,
> kernel coefficient gamma = 'scale' (i.e., 1/(n_features * Var(X))), and
> probability calibration disabled during training (enabled post-hoc via Platt
> scaling for threshold analysis). Features are not standardized prior to fitting,
> as VGG16 pooled outputs occupy a naturally bounded range. The random seed is
> fixed at 42 for reproducibility. The SVM operates on individual VGG16 feature
> vectors (512-d each), treating each extracted frame as an independent sample.
> Per-video predictions are obtained by averaging calibrated frame-level
> probabilities across the 15 extracted frames per video.

---

## Section 3.1.5 — LRCN Architecture (R#4 Point 6)

**REPLACE** the architecture description (remove any mention of Conv1D/sigmoid) with:

> The LRCN receives a sequence of 15 VGG16 feature vectors, each of dimension
> 512, yielding an input tensor of shape (15, 512). The architecture consists of:
> TimeDistributed(Dense(256, ReLU)) -> Dropout(0.3) -> LSTM(128) -> Dropout(0.3)
> -> Dense(2, softmax). The model is compiled with categorical cross-entropy loss
> and the Adam optimizer (learning rate = 1e-4). Training uses a batch size of 2
> for 10 epochs with a fixed random seed of 42. The softmax output produces a
> two-element probability vector [P(fake), P(real)]; the class with higher
> probability is selected as the prediction.

---

## Section 4.1 — Datasets & Split Protocol (R#4 Point 2)

**ADD** after the dataset description:

> **Split protocol.** The dataset is partitioned at the video level into 80%
> training and 20% testing using stratified sampling (preserving the class ratio)
> with random_state = 42. Identity-level disjointness is not enforced, as
> Celeb-DF v2 does not provide per-identity metadata for synthetic videos. Frame
> leakage is prevented by construction: all frames from a given video appear
> exclusively in either the training or test set, never both. The resulting test
> set contains 1,246 videos (1,128 fake, 118 real), yielding a class imbalance
> ratio of 9.56:1.

---

## Section 4.2 — Loss Function & Equation (R#4 Point 4)

**REPLACE** any mention of "Sparse Categorical Cross-Entropy" or "binary cross-entropy" with:

> The LRCN is trained using categorical cross-entropy loss:
>
>   L = -(1/N) * sum_{i=1}^{N} sum_{c=0}^{1} y_{i,c} * log(p_{i,c})
>
> where y_{i,c} is the one-hot encoded ground truth and p_{i,c} is the softmax
> output for class c. This is the standard multi-class cross-entropy applied to
> the two-class (fake, real) setting with a softmax output layer.

---

## Section 4.2 — Evaluation Unit (R#4 Point 3)

**ADD** a paragraph clarifying evaluation:

> **Evaluation protocol.** All reported metrics are computed at the video level.
> For the LRCN, each test video produces a single prediction from its 15-frame
> sequence (N = 1,246 videos). For the SVM, which operates on individual frames,
> we report both per-frame results (N = 18,690 frames, i.e., 1,246 videos x 15
> frames) and per-video results obtained by averaging the calibrated
> frame-level probabilities over the 15 frames per video and applying a 0.5
> decision threshold (N = 1,246 videos). The per-video SVM results enable direct
> comparison with the LRCN on equal footing.

---

## Section 4.3 — Table 1 (Celeb-DF Results) — CORRECTED (R#4 Point 1)

**REPLACE** Table 1 with:

### Table 1: Celeb-DF v2 Results (Per-Video, N = 1,246)

| Model              | Precision | Recall | F1     | Accuracy | Bal. Acc. | AUC   |
|--------------------|-----------|--------|--------|----------|-----------|-------|
| LRCN (original)    | 0.957     | 0.186  | 0.312  | 0.922    | 0.593     | 0.790 |
| LRCN (focal loss)  | 0.155     | 0.737  | 0.257  | 0.596    | 0.659     | 0.732 |
| SVM (per-video)    | 0.615     | 0.203  | 0.306  | 0.913    | 0.595     | 0.823 |

Notes:
- Precision/Recall/F1 are for the Real (positive) class.
- "Bal. Acc." = balanced accuracy = (recall_real + specificity) / 2.
- SVM per-video = mean of calibrated frame probabilities over 15 frames.

### Table 1b: SVM Per-Frame Results (N = 18,690 frames)

| Model       | Precision | Recall | F1    | Accuracy | Bal. Acc. | AUC   |
|-------------|-----------|--------|-------|----------|-----------|-------|
| SVM (frame) | 0.714     | 0.011  | 0.022 | 0.906    | 0.505     | 0.742 |

---

## Section 4.3 — Table 2 (DFDC Results) — NEW

### Table 2: DFDC Results (Per-Video, N = 607)

| Model              | Precision | Recall | F1     | Accuracy | Bal. Acc. | AUC   |
|--------------------|-----------|--------|--------|----------|-----------|-------|
| LRCN (original)    | 0.450     | 0.462  | 0.456  | 0.929    | 0.711     | 0.739 |
| LRCN (focal loss)  | 0.126     | 0.795  | 0.218  | 0.633    | 0.708     | 0.816 |
| SVM (per-video)    | 0.867     | 0.333  | 0.481  | 0.954    | 0.665     | 0.877 |

Notes:
- DFDC test set: 607 videos (568 fake, 39 real), imbalance ratio 14.56:1
- Same split protocol: 80/20 stratified, video-level, seed=42
- DFDC is harder (more diverse manipulations, higher imbalance)

### Table 2b: DFDC SVM Per-Frame Results (N = 9,105 frames)

| Model       | Precision | Recall | F1    | Accuracy | Bal. Acc. | AUC   |
|-------------|-----------|--------|-------|----------|-----------|-------|
| SVM (frame) | 0.632     | 0.021  | 0.040 | 0.936    | 0.510     | 0.763 |

---

## Section 4.3 — DFDC Verification Paragraphs

**ADD** after Table 2:

> **Verification (DFDC).** For the LRCN (original) on DFDC: TN = 546, FP = 22,
> FN = 21, TP = 18, giving Precision(Real) = 18/40 = 0.450, Recall(Real) =
> 18/39 = 0.462, Accuracy = (546+18)/607 = 0.929, Balanced Accuracy =
> (0.462 + 0.961)/2 = 0.711.
>
> For the LRCN (focal loss): TN = 353, FP = 215, FN = 8, TP = 31, giving
> Precision(Real) = 31/246 = 0.126, Recall(Real) = 31/39 = 0.795, Balanced
> Accuracy = (0.795 + 0.622)/2 = 0.708.
>
> For the SVM (per-video): TN = 566, FP = 2, FN = 26, TP = 13, giving
> Precision(Real) = 13/15 = 0.867, Recall(Real) = 13/39 = 0.333, AUC = 0.877.

---

## Section 4.3 — Verification Paragraphs (R#4 Point 1)

**ADD** after Table 1:

> **Verification.** We explicitly derive each metric from the confusion matrices
> to ensure consistency. For the LRCN (original), the confusion matrix yields
> TN = 1,127, FP = 1, FN = 96, TP = 22, from which: Precision(Real) =
> 22/(22+1) = 0.957, Recall(Real) = 22/(22+96) = 0.186, Accuracy =
> (1,127+22)/1,246 = 0.922, Balanced Accuracy = (0.186 + 0.999)/2 = 0.593.
>
> For the LRCN (focal loss): TN = 655, FP = 473, FN = 31, TP = 87, giving
> Precision(Real) = 87/560 = 0.155, Recall(Real) = 87/118 = 0.737, Balanced
> Accuracy = (0.737 + 0.581)/2 = 0.659.
>
> For the SVM (per-video): TN = 1,113, FP = 15, FN = 94, TP = 24, giving
> Precision(Real) = 24/39 = 0.615, Recall(Real) = 24/118 = 0.203, AUC = 0.823.

---

## Section 4.3 — Runtime Claim (R#4 Point 7)

**REPLACE** the "35 FPS" / "0.028 fps" claim with:

> **Runtime analysis.** We measure end-to-end inference time on an NVIDIA RTX
> 3050 GPU across 10 test videos. The pipeline processes a single video (15
> frames) in 0.869 +/- 0.788 seconds, broken down as: face detection via Haar
> cascade (0.467 +/- 0.116 s, CPU), VGG16 feature extraction (0.343 +/- 0.619
> s, GPU), and LRCN classification (0.058 +/- 0.059 s, GPU). This yields a
> throughput of approximately 17.3 frames per second (15 frames / 0.869 s),
> which is suitable for offline or near-real-time forensic analysis but does not
> constitute real-time video processing at standard multimedia rates (30-60 fps).
> The primary bottleneck is the CPU-bound Haar cascade face detection stage,
> which accounts for 54% of total inference time.

---

## Section 4.3 — Class Imbalance Analysis (R#7 Points 3-6) — NEW SUBSECTION

**ADD** as Section 4.4 or as a subsection within 4.3:

> ### 4.4 Class Imbalance Analysis
>
> **Recall collapse under imbalance.** The original LRCN achieves 92.2% accuracy
> but only 18.6% recall on the minority (real) class, with balanced accuracy of
> 59.3%. This recall collapse is a known failure mode when training with
> standard cross-entropy on heavily imbalanced data (9.56:1 ratio): the model
> minimizes loss by defaulting to the majority class, inflating accuracy while
> failing to detect real videos. Balanced accuracy, which averages per-class
> recall, reveals this: 59.3% is only marginally above the 50% random baseline.
>
> **Cost-sensitive retraining.** To mitigate this, we retrain the LRCN using
> focal loss (gamma = 2.0) with inverse-frequency class weights (alpha_fake =
> 0.526, alpha_real = 5.034) and balanced mini-batch sampling (equal fake/real
> per batch). Focal loss down-weights well-classified examples, focusing
> gradient updates on hard-to-classify minority samples.
>
> **Results.** On Celeb-DF v2, the retrained LRCN improves recall from 18.6%
> to 73.7% and balanced accuracy from 59.3% to 65.9%. On DFDC, recall improves
> from 46.2% to 79.5% and balanced accuracy remains stable at 70.8% (vs 71.1%).
> This confirms across both datasets that class-imbalance mitigation addresses
> the recall collapse. However, this comes at the cost of reduced precision and
> overall accuracy, reflecting the fundamental precision-recall trade-off under
> severe imbalance.
>
> **Threshold analysis.** Figure X shows precision, recall, F1, and balanced
> accuracy as functions of the classification threshold for all three models.
> The LRCN (original) achieves optimal F1 = 0.396 at threshold 0.14, far below
> the default 0.5, indicating the model's probability outputs are poorly
> calibrated for the minority class. The SVM (per-video) achieves optimal F1 =
> 0.488 at threshold 0.331, with AUC = 0.823, demonstrating strong
> discriminative capacity that is masked by the default threshold.
>
> **PR curves.** Figure X presents precision-recall curves for each model. The
> baseline (random classifier) precision is 9.5% (118/1,246), corresponding to
> the real-class prevalence. The LRCN (original) achieves average precision (AP)
> = 0.436, the retrained LRCN AP = 0.353, and the SVM (per-video) AP = 0.436,
> all substantially above baseline, confirming that all models learn meaningful
> discriminative features despite the challenging class distribution.

---

## Section 4.3 — Baseline Comparisons (R#4 Point 8)

**ADD** a note before/after any comparison table:

> Results for DenseNet and VeriFace are reproduced from their respective
> publications and were not re-evaluated under our split protocol. Direct
> numerical comparison should be interpreted with caution, as differences in
> train/test splits, preprocessing, and evaluation units (per-frame vs.
> per-video) may account for performance variations.

---

## Abstract & Introduction — Contribution Claims (R#4 Point 9)

**REMOVE** or soften these claims:
- "eye blinking detection" -> Remove entirely (not implemented in code)
- "face recognition" -> Remove or change to "face detection using Haar cascades"

**REPLACE** contribution statement with something like:

> VeriLens is a deepfake detection pipeline that extracts face sequences using
> Haar cascade detection, encodes them with pretrained VGG16 features, and
> classifies videos using either an LRCN (for temporal analysis) or an SVM
> baseline (for single-frame analysis). We evaluate on Celeb-DF v2 and DFDC,
> and provide
> a systematic analysis of class imbalance effects, including focal loss
> retraining and threshold optimization.

---

## Minor Corrections (R#4 Point 10)

1. **"AAR Cascade"** -> **"Haar Cascade"** (search and replace globally)
2. **Section 4.3 cross-reference**: "Section 3.1" for datasets -> **"Section 4.1"**
3. **AUC interpretation**: Remove any statement like "AUC 0.95 means 92.1% of instances..."
   Replace with: "An AUC of X indicates that a randomly chosen real video receives a higher
   score than a randomly chosen fake video with probability X."
4. **Thesis-style language**: Remove first-person narrative ("We then proceeded to...",
   "It was observed that..."). Use direct active voice: "The model achieves...",
   "Table 1 reports...", "Figure X shows..."

---

## Figures to Include in Paper

From `results/celeb-df/`, include these figures:

| Figure | File | Description |
|--------|------|-------------|
| Fig. A | `lrcn_original/confusion_matrix.png` | LRCN original confusion matrix |
| Fig. B | `lrcn_retrained/confusion_matrix.png` | LRCN focal-loss confusion matrix |
| Fig. C | `svm_per_video/confusion_matrix.png` | SVM per-video confusion matrix |
| Fig. D | `lrcn_original/ROC_curve.png` | LRCN ROC (AUC=0.790) |
| Fig. E | `svm/ROC_curve.png` | SVM per-frame ROC (AUC=0.742) |
| Fig. F | `svm_per_video/ROC_curve.png` | SVM per-video ROC (AUC=0.823) |
| Fig. G | `lrcn_original/PR_curve.png` | LRCN PR curve (AP=0.436) |
| Fig. H | `lrcn_original/threshold_analysis.png` | LRCN threshold analysis |
| Fig. I | `svm_per_video/threshold_analysis.png` | SVM threshold analysis |
| Fig. J | `lrcn_original/training_history.png` | LRCN training curves |
| Fig. K | `lrcn_retrained/training_history.png` | Retrained LRCN training curves |
