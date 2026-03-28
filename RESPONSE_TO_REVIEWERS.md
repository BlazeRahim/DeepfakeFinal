# Response to Reviewers — VeriLens

Dear Editor and Reviewers,

We thank all reviewers for their constructive feedback. We have addressed every
point raised. Below we provide a point-by-point response with specific changes
made and the sections/pages affected.

---

## Response to Reviewer #4

### Point 1: Celeb-DF result inconsistency (Table 1 column swap)

**Reviewer concern:** Table 1's precision/recall/F1 values do not match the
confusion matrices; values align with the opposite model column.

**Response:** The reviewer is correct — the original Table 1 had the LRCN and
SVM column headers swapped. We have corrected Table 1 and now include explicit
confusion-matrix-derived verification in Section 4.3:

- **LRCN (original):** TN=1,127, FP=1, FN=96, TP=22 -> Precision=0.957,
  Recall=0.186, F1=0.312, Accuracy=0.922, Balanced Accuracy=0.593, AUC=0.790
- **SVM (per-video):** TN=1,113, FP=15, FN=94, TP=24 -> Precision=0.615,
  Recall=0.203, F1=0.306, Accuracy=0.913, Balanced Accuracy=0.595, AUC=0.823

Each metric can be independently verified from the confusion matrix values
shown in the updated figures. (See Section 4.3, Table 1, and the new
verification paragraph.)

---

### Point 2: Split protocol and leakage controls

**Reviewer concern:** The manuscript does not state train/test ratios, whether
splitting is at video level, or how leakage is prevented.

**Response:** We have added a detailed split protocol to Section 4.1:

- 80/20 stratified split at the **video level** (random_state=42)
- All frames from a given video appear exclusively in either train or test
- Test set: 1,246 videos (1,128 fake, 118 real), imbalance ratio 9.56:1
- Identity-level disjointness is not enforced, as Celeb-DF v2 does not provide
  per-identity metadata for synthetic videos — this is noted as a limitation

---

### Point 3: Evaluation unit clarity

**Reviewer concern:** Unclear whether metrics are per-frame or per-video;
confusion matrix totals don't match dataset counts.

**Response:** We now state explicitly in Section 4.2:

- **LRCN:** per-video evaluation (N=1,246 videos, one prediction per 15-frame
  sequence)
- **SVM (per-frame):** N=18,690 frames (1,246 videos x 15 frames); each VGG16
  feature vector is an independent sample
- **SVM (per-video):** calibrated frame probabilities are averaged over 15
  frames per video, then thresholded (N=1,246 videos)

The per-video SVM results are used for direct comparison with the LRCN. Both
evaluation units are reported in separate table rows.

---

### Point 4: Loss function contradiction

**Reviewer concern:** Section 4.2 says "Sparse Categorical Cross-Entropy" while
Section 3.1.5 describes binary cross-entropy with sigmoid output.

**Response:** The implemented loss function is **categorical cross-entropy**
with a **softmax** output layer (2-class). We have corrected both Section 3.1.5
and Section 4.2 to reflect this consistently. The equation in Section 4.2 has
been updated to the standard categorical cross-entropy formulation.

---

### Point 5: SVM configuration

**Reviewer concern:** SVM cannot be replicated without kernel, C, gamma, class
weights, and preprocessing details.

**Response:** We have added full SVM specification to Section 3.1.4:

| Parameter        | Value                          |
|-----------------|--------------------------------|
| Kernel          | RBF                            |
| C               | 1.0                            |
| Gamma           | 'scale' (1/(n_features*Var(X)))|
| Class weights   | None (uniform)                 |
| Probability     | Post-hoc Platt scaling         |
| Standardization | None (VGG16 outputs are bounded)|
| Random seed     | 42                             |

---

### Point 6: LRCN model definition

**Reviewer concern:** Architecture is incomplete (Conv1D/MaxPool mentioned but
not matching code); missing sequence construction, learning rate, etc.

**Response:** We have rewritten Section 3.1.5 with the correct architecture:

```
Input: (15, 512) — 15 frames x 512-d VGG16 features
TimeDistributed(Dense(256, ReLU))
Dropout(0.3)
LSTM(128)
Dropout(0.3)
Dense(2, softmax)
```

- Loss: categorical cross-entropy
- Optimizer: Adam (lr=1e-4)
- Batch size: 2, Epochs: 10, Seed: 42
- Sequence construction: 15 frames sampled every 15th frame from each video;
  if fewer than 15 faces detected, zero-padded
- No learning rate scheduling or early stopping for the original model

The incorrect reference to Conv1D/MaxPool has been removed.

---

### Point 7: Runtime claim

**Reviewer concern:** Arithmetic inconsistency (0.42s for 15 frames claimed as
35 FPS and 0.028 fps simultaneously); not real-time.

**Response:** We have replaced the incorrect claim with measured per-stage
timing on an RTX 3050 (mean +/- std over 10 videos):

| Stage              | Time (s)          |
|--------------------|-------------------|
| Haar face detection| 0.467 +/- 0.116   |
| VGG16 extraction   | 0.343 +/- 0.619   |
| LRCN classification| 0.058 +/- 0.059   |
| **Total**          | **0.869 +/- 0.788**|

Throughput: ~17.3 frames/sec. We now explicitly state this is suitable for
offline forensic analysis, not real-time processing (which requires 30+ fps).
The "35 FPS" claim has been removed.

---

### Point 8: Baseline comparisons protocol-matched

**Reviewer concern:** DenseNet and VeriFace numbers may not be from the same
protocol.

**Response:** We have added a note in Section 4.3 stating that DenseNet and
VeriFace results are reproduced from their original publications and were not
re-evaluated under our split protocol. We have softened comparative claims
accordingly, noting that differences in splits, preprocessing, and evaluation
units limit direct comparison.

---

### Point 9: Contribution claims (eye blinking, face recognition)

**Reviewer concern:** Abstract/introduction reference "eye blinking detection"
and "face recognition" but neither is implemented or ablated.

**Response:** We have removed the "eye blinking detection" claim entirely, as
it is not implemented in the pipeline. "Face recognition" has been changed to
"face detection using Haar cascades," which accurately describes the
implemented module. The contribution statement now reads:

> VeriLens extracts face sequences using Haar cascade detection, encodes them
> with pretrained VGG16 features, and classifies videos using either an LRCN
> (for temporal analysis) or an SVM baseline (for single-frame analysis).

---

### Point 10: Minor technical corrections

**Response:** All corrected:

1. "AAR Cascade" -> "Haar Cascade" (global find-and-replace)
2. Section 4.3 cross-reference "Section 3.1" -> "Section 4.1"
3. AUC interpretation corrected: "An AUC of 0.790 indicates that a randomly
   chosen real video receives a higher score than a randomly chosen fake video
   with probability 79.0%."
4. Thesis-style language has been revised to use direct active voice throughout.

---

## Response to Reviewer #7

### Point 1: FPS claim misleading

**Reviewer concern:** 0.42s for 15 frames extrapolated to "~35 FPS" is
confusing and does not constitute real-time processing.

**Response:** Addressed jointly with R#4 Point 7 above. The claim has been
replaced with measured per-stage timing (total: 0.869 +/- 0.788 s/video, ~17.3
fps). We explicitly state this is offline/near-real-time, not standard
real-time multimedia processing.

---

### Point 2: Thesis-style writing

**Reviewer concern:** Some sections still exhibit thesis-style writing.

**Response:** We have revised the manuscript to use direct, concise active
voice throughout. Narrative passages ("We then proceeded to investigate..."),
first-person accounts of the research process, and unnecessary hedging have
been replaced with factual statements ("Table 1 reports...", "The model
achieves...", "Figure X shows...").

---

### Point 3: Add balanced accuracy, PR curves, cost-weighted loss analysis

**Reviewer concern:** No balanced accuracy, PR curves, or cost-weighted loss
analysis provided.

**Response:** We now report:

- **Balanced accuracy** for all models in Tables 1-2:
  - Celeb-DF: LRCN 0.593, LRCN-retrained 0.659, SVM per-video 0.595
  - DFDC: LRCN 0.711, LRCN-retrained 0.708, SVM per-video 0.665
- **PR curves** with average precision (AP) and the random baseline for all
  models on both datasets (new figures)
- **Cost-weighted loss analysis** via focal loss (gamma=2.0) with
  inverse-frequency alpha weights, demonstrating the precision-recall trade-off
  under class imbalance (new Section 4.4)

---

### Point 4: Threshold analysis

**Reviewer concern:** No threshold analysis is provided.

**Response:** We now include threshold analysis plots (new Figure X) showing
precision, recall, F1, and balanced accuracy as functions of the decision
threshold for all three models. Key findings:

- LRCN (original): optimal F1=0.396 at threshold 0.14 (default 0.5 severely
  under-detects real videos)
- SVM (per-video): optimal F1=0.488 at threshold 0.331 (AUC=0.823 confirms
  strong discriminative capacity masked by default threshold)
- LRCN (focal loss): optimal F1=0.356 at threshold 0.765

---

### Point 5: Cost-sensitive learning / class-imbalance mitigation

**Reviewer concern:** No cost-sensitive learning or class-imbalance mitigation
explored.

**Response:** We now include an LRCN retrained with:

1. **Focal loss** (gamma=2.0) to down-weight easy majority-class examples
2. **Inverse-frequency class weights** (alpha_fake=0.526, alpha_real=5.034)
3. **Balanced mini-batch sampling** (equal fake/real per batch)
4. **ReduceLROnPlateau** (factor=0.5, patience=2) and **early stopping**
   (patience=4, restore best weights)

Results on Celeb-DF (per-video, N=1,246):
- Recall improved from 18.6% to 73.7% (+55.1 percentage points)
- Balanced accuracy improved from 59.3% to 65.9% (+6.6 pp)

Results on DFDC (per-video, N=607):
- Recall improved from 46.2% to 79.5% (+33.3 pp)
- Balanced accuracy stable at 70.8% (vs 71.1%)
- AUC improved from 0.739 to 0.816

This demonstrates that the recall collapse is directly caused by class
imbalance and can be substantially mitigated through cost-sensitive training.

---

### Point 6: No ablation for LRCN recall collapse

**Reviewer concern:** No ablation explains why LRCN collapses in recall while
maintaining high accuracy.

**Response:** The new Section 4.4 provides this analysis:

1. The 9.56:1 fake-to-real class imbalance means a trivial all-fake classifier
   achieves ~90.5% accuracy. The original LRCN (92.2% accuracy, 18.6% recall)
   is only marginally better than this baseline.

2. Balanced accuracy (59.3%) is only 9.3 points above random (50%), revealing
   the true discriminative performance masked by standard accuracy.

3. Threshold analysis shows the LRCN assigns systematically low probability to
   the real class — optimal threshold is 0.14, not 0.5.

4. Retraining with focal loss + balanced batches recovers 73.7% recall
   (Celeb-DF) and 79.5% recall (DFDC), confirming across both datasets that
   the collapse is caused by the loss landscape under imbalance, not
   architectural limitations.

---

## Summary of Changes

| Change | Section | Reviewer |
|--------|---------|----------|
| Corrected Table 1 (swapped headers) | 4.3 | R#4.1 |
| Added verification paragraphs | 4.3 | R#4.1 |
| Added split protocol | 4.1 | R#4.2 |
| Clarified evaluation unit | 4.2 | R#4.3 |
| Fixed loss function (categorical CE) | 3.1.5, 4.2 | R#4.4 |
| Full SVM specification | 3.1.4 | R#4.5 |
| Corrected LRCN architecture | 3.1.5 | R#4.6 |
| Corrected runtime claim | 4.3 | R#4.7, R#7.1 |
| Labeled baseline comparisons | 4.3 | R#4.8 |
| Removed overclaims | Abstract, 1 | R#4.9 |
| Fixed typos and cross-refs | Global | R#4.10 |
| Revised thesis-style writing | Global | R#7.2 |
| Added balanced accuracy | 4.3 Table 1 | R#7.3 |
| Added PR curves | 4.3 new fig | R#7.3 |
| Added focal loss analysis | 4.4 (new) | R#7.3, R#7.5 |
| Added threshold analysis | 4.3 new fig | R#7.4 |
| Added class-imbalance mitigation | 4.4 (new) | R#7.5 |
| Added recall collapse ablation | 4.4 (new) | R#7.6 |
| Added DFDC evaluation (Table 2) | 4.3 | All |
| DFDC verification paragraphs | 4.3 | R#4.1 |

We believe these revisions address all reviewer concerns comprehensively.
