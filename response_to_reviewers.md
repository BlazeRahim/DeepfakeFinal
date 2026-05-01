# Response to Reviewers — VeriLens Revision
# Journal: Multimedia Tools and Applications (Springer)
# Manuscript: VeriLens — Deepfake Detection via VGG16 + SVM/LRCN

---

## General Statement

We thank all reviewers for their detailed and constructive feedback. One reviewer accepted the previous revision; we address the remaining concerns from Reviewers #4 and #7 below. All changes are summarized in a revised manuscript with tracked changes. Experimental results have been re-run with full reproducibility documentation.

---

# REVIEWER #4

---

## Point 1 — Celeb-DF Result Inconsistency (Table 1 Column Swap)

**Reviewer comment**: Table 1's precision/recall/F1 values do not match the confusion matrices for the model labels shown in Figs. 11–12. When the metrics are recomputed from the displayed counts, the values align with the opposite model column.

**Response**: The reviewer is correct. We identified a column header labeling error in Table 1: the LRCN and SVM column headers were swapped. The underlying numerical values were correct; only the headers were mislabeled. We have corrected the column headers in the revised Table 1. To make this explicit, we have added the following verification paragraph to Section 4.3:

> From the LRCN confusion matrix (TN=**1126**, FP=**2**, FN=**97**, TP=**21**): Precision(Real) = TP/(TP+FP) = **0.913**, Recall(Real) = TP/(TP+FN) = **0.178**, F1(Real) = **0.2979**, Accuracy = **0.9205**. These values now appear under the LRCN column in the corrected Table 1. From the SVM confusion matrix (TN=**1128**, FP=**0**, FN=**118**, TP=**0**): Precision(Real) = **0.0**, Recall(Real) = **0.0**, F1(Real) = **0.0**.

**[ACTION: Fill values from results/celeb-df/lrcn_original/verification.txt and results/celeb-df/svm/verification.txt after running the notebook]** - DONE.

---

## Point 2 — Split Protocol and Leakage Controls

**Reviewer comment**: The manuscript does not state train/validation/test ratios, whether splitting is at video level, whether subjects/identities are disjoint, and how frame leakage is prevented.

**Response**: We have added the following to Section 4.1:

> All dataset splits are performed at the **video level** using a stratified 80/20 train/test partition (scikit-learn train_test_split, random_state=42, stratify=y). For LRCN training, an additional internal 20% of the training set is reserved for validation during training (no separate held-out validation set), resulting in an effective 64%/16%/20% train/internal-val/test split. The SVM is trained on the full 80% training split with no internal validation.
>
> **Leakage prevention**: Each video is processed independently. All 15 frames sampled from a given video appear exclusively in the split to which that video is assigned. No frame-level shuffling is performed across videos. Because Celeb-DF v2 and DFDC do not publish identity metadata in a form usable for subject-disjoint splitting, we note this as a limitation; however, the video-level split prevents the primary form of train-test leakage (frame overlap).

---

## Point 3 — Evaluation Unit and Frame Aggregation

**Reviewer comment**: It is unclear whether metrics are per-frame or per-video. Dataset counts in Section 4.1 do not align with confusion matrix totals.

**Response**: We clarify that the **original paper's Celeb-DF confusion matrices (Figs. 11–12) reflect per-frame evaluation** (which explains the inflated sample counts: 8,599 and 16,555 vs. 6,229 total videos). This is now explicitly disclosed. All new experimental results presented in the revised paper use **per-video evaluation**:

> Each video is represented by a fixed-length sequence of T=15 frames, sampled at a stride of 15 frames from the beginning of the video. A single prediction is produced per video:
> - **LRCN**: the softmax output of the LSTM over the 15-frame sequence gives one prediction per video.
> - **SVM**: the VGG16 feature vector of the first frame in the sequence is used to produce one prediction per video.
>
> We have updated Sections 4.1 and 4.3 and the captions of all figures to state the evaluation unit explicitly.

---

## Point 4 — Loss Function Contradiction

**Reviewer comment**: Section 4.2 states "Sparse Categorical Cross-Entropy"; Section 3.1.5 describes binary cross-entropy with sigmoid. These are contradictory and neither matches the actual implementation.

**Response**: Both descriptions were incorrect. The actual implementation uses **categorical cross-entropy** with a Dense(2, softmax) output layer. We have:

1. Corrected Section 3.1.5 to describe the actual architecture (see Point 6 below).
2. Updated Section 4.2 and Equation 6 to read:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{1} y_{i,c} \log(\hat{y}_{i,c})$$

where $y_{i,c} \in \{0,1\}$ is the one-hot encoded ground truth for sample $i$ and class $c$, and $\hat{y}_{i,c}$ is the predicted softmax probability.

3. Removed all mentions of "binary cross-entropy", "sparse categorical cross-entropy", and "sigmoid output" from the LRCN description.

---

## Point 5 — SVM Configuration

**Reviewer comment**: The SVM baseline cannot be replicated without kernel, C, γ, class weights, probability calibration, and feature standardization details.

**Response**: We have added the following to Section 3.1.4:

> The SVM classifier is configured with: RBF kernel, regularization parameter C = 1.0, kernel coefficient γ = 'scale' (computed as 1/(n_features × Var(X)), i.e., ≈ 1/(512 × Var(X))), probability calibration enabled (Platt scaling via scikit-learn's probability=True), no class weighting, and random_state = 42. **No feature normalization or standardization is applied** to the VGG16 feature vectors prior to SVM training. The SVM operates on the 512-dimensional VGG16 feature vector from a single frame per video (the first frame in the 15-frame sequence).

---

## Point 6 — LRCN Architecture Completeness

**Reviewer comment**: Section 3.1.5 gives a partial architecture (Conv1D/MaxPool/LSTM/Dense) but input tensor definition, sequence construction, learning rate, and source of mean±std are not reported.

**Response**: Section 3.1.5 has been rewritten to match the actual implementation:

> **Input**: A sequence of T=15 VGG16 feature vectors, each of dimension 512, giving input shape (15, 512). Sequences are padded with zero vectors if fewer than 15 face-detected frames are available (e.g., short videos or failed face detection).
>
> **Architecture**:
> 1. TimeDistributed(Dense(256, activation='relu')) — applies the same Dense layer to each of the 15 time steps independently.
> 2. Dropout(0.3)
> 3. LSTM(128, return_sequences=False) — produces a single 128-dimensional vector.
> 4. Dropout(0.3)
> 5. Dense(2, activation='softmax') — two-class output (Fake=0, Real=1).
>
> **Training**: Adam optimizer, learning rate = 1×10⁻⁴, batch size = 2, 10 epochs. Loss: categorical cross-entropy. Internal 20% validation split during training. Random seed = 42 (NumPy and TensorFlow).
>
> **Note on ±std in tables**: The ±standard deviation values in the original tables were not derived from multiple random seeds. In the revised paper, we either report results from a single definitive run or explicitly note when variance is computed across seeds.

---

## Point 7 — Runtime Claim

**Reviewer comment**: Section 4.3 contains an arithmetic error: 15/0.42 ≈ 35.7 fps, not 0.028 fps. Report as mean±std with stage breakdown.

**Response**: The original text contained an arithmetic error (0.028 is seconds-per-frame, not fps). More fundamentally, frame-processing rate does not imply real-time continuous video streaming capability. We have replaced the runtime section with:

> Pipeline latency was measured over **10** test videos. Results (mean ± std):
>
> | Stage | Mean (s) | Std (s) |
> |---|---|---|
> | Face detection (Haar, CPU) | **0.3516** | **0.0216** |
> | VGG16 feature extraction (GPU) | **0.1487** | **0.0779** |
> | LRCN classifier (GPU) | **0.0564** | **0.0591** |
> | **Total** | **0.5567** | **0.1381** |
>
> GPU acceleration (CUDA) is used for VGG16 and LRCN inference; Haar cascade face detection runs on CPU via OpenCV.
>
> No real-time claim is made; VeriLens is designed for post-hoc video analysis, not live-stream processing.

---

## Point 8 — Baseline Comparison Protocol

**Reviewer comment**: DenseNet and VeriFace numbers in Tables 1–2 may be from different evaluation protocols.

**Response**: We have added the following footnote to Tables 1 and 2:

> "† Results for DenseNet and VeriFace are reported from original publications [ref] and were not re-evaluated under our experimental protocol. Direct comparison should be interpreted with caution."

We have softened all comparative claims in the text to reflect this (e.g., replacing "outperforms" with "achieves competitive performance with").

---

# REVIEWER #7

---

## Point 1 — FPS / Real-Time Claim

**Reviewer comment**: "0.42 s for 15 frames extrapolated to ~35 FPS is confusing and will not result in real-time video processing."

**Response**: Agreed. The claim has been removed. See Reviewer #4, Point 7 for the replacement latency table. We no longer claim real-time performance.

---

## Point 2 — Thesis-Style Writing

**Reviewer comment**: Some sections exhibit thesis-style prose.

**Response**: We have revised the abstract, introduction, and Section 3 to use concise academic journal style. Specifically:
- Removed step-by-step narrative descriptions of well-known methods (VGG16, SVM)
- Condensed Section 3 methodology into focused descriptions
- Replaced lengthy motivational paragraphs with precise technical statements

---

## Point 3 — Balanced Accuracy, PR Curves, Threshold Analysis

**Reviewer comment**: Add balanced accuracy, PR curves, or cost-weighted loss analysis.

**Response**: We have added:

1. **Balanced accuracy** column to Tables 1 and 2. Values: **LRCN-Original: 0.5881, SVM: 0.5000, LRCN-Retrained: 0.6530**

2. **Precision-Recall curves** (new figures): Due to class imbalance (~9:1 Fake:Real in Celeb-DF v2), ROC AUC can be misleading. We now include PR curves with Average Precision scores for all models.

3. **Threshold analysis figure**: A plot of precision, recall, F1, and balanced accuracy as a function of decision threshold is provided for the LRCN model. The optimal threshold for the Real class (maximizing F1(Real)) is **0.095**, at which Precision(Real)=**0.4554**, Recall(Real)=**0.4322**, F1(Real)=**0.4435**.

4. **Retrained LRCN** with focal loss (γ=2.0) and balanced mini-batches as a class-imbalance mitigation strategy. This addresses the cost-sensitive learning concern.

---

## Point 4 — No Threshold Analysis

**Reviewer comment**: No threshold analysis is provided.

**Response**: Addressed above (Point 3). A threshold sensitivity plot is now included as a new figure in Section 4.3.

---

## Point 5 — No Class-Imbalance Mitigation

**Reviewer comment**: No cost-sensitive learning or class-imbalance mitigation is explored.

**Response**: We now present a second LRCN variant trained with:
- **Focal loss** (γ=2.0) with inverse-frequency class weighting: α_fake = (N)/(2·N_fake), α_real = (N)/(2·N_real)
- **Balanced mini-batch generator**: each training batch contains equal numbers of Real and Fake samples (oversampling the minority with replacement)

Results for LRCN-Retrained are reported alongside the original in the revised Tables 1–2. Key finding: balanced training substantially improves Recall(Real) and balanced accuracy, at the cost of overall accuracy.

---

## Point 6 — No Ablation for LRCN Recall Collapse

**Reviewer comment**: No ablation explains why LRCN collapses in recall while maintaining high accuracy.

**Response**: We have added an ablation/analysis paragraph to Section 4.3:

> The LRCN's low recall for the Real class despite high overall accuracy is a direct consequence of the 9:1 class imbalance (Fake:Real) in Celeb-DF v2. A classifier predicting Fake for all samples achieves ~90% accuracy trivially. The LRCN, trained with standard categorical cross-entropy on imbalanced batches, learns to minimize loss by biasing predictions toward the majority (Fake) class. This is evidenced by the confusion matrix: the LRCN achieves high TN (correctly rejecting fake videos) but low TP (failing to identify genuine videos). The retrained LRCN (focal loss + balanced batches) corrects this at the cost of introducing false positives, demonstrating the precision-recall trade-off inherent in imbalanced detection tasks.

---

## Summary of Changes

| Issue | Section Changed | Change Type |
|---|---|---|
| Table 1 column swap | Table 1 + Section 4.3 | Correction |
| Loss function contradiction | Eq. 6, Sec. 3.1.5, Sec. 4.2 | Correction |
| LRCN architecture mismatch | Section 3.1.5 | Rewrite |
| FPS / real-time claim | Section 4.3 | Removal + latency table |
| Evaluation unit | Section 4.1, 4.3, captions | Clarification |
| SVM hyperparameters | Section 3.1.4 | Addition |
| Train/test split protocol | Section 4.1 | Addition |
| Haar cascade typo | Throughout | Correction |
| Cross-reference fix | Section 4.3 | Correction |
| AUC description | Section 4.3 | Correction |
| Eye-blink / face-rec overclaims | Abstract, Introduction | Softened |
| Baseline comparison footnote | Tables 1–2 | Addition |
| Balanced accuracy | Tables 1–2, Section 4.3 | Addition |
| Threshold analysis | Section 4.3 + new figure | Addition |
| PR curves | Section 4.3 + new figures | Addition |
| Focal loss retrained model | Section 3, 4.3, Tables 1–2 | Addition |
| Ablation: LRCN recall collapse | Section 4.3 | Addition |

---
