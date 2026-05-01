# VeriLens Paper Corrections Cheat Sheet
# Every change needed, organized by section
# Verified against actual code in FinalLRCN.py / Final.py / retrain.py

---

## ABSTRACT

### Change 1 — Remove eye-blinking / face-recognition overclaims
**REMOVE** any mention of: "eye blinking detection", "face recognition module", "blink-based liveness"
**REPLACE WITH**: "face analysis through Haar cascade-based face detection"

### Change 2 — Fix FPS claim
**REMOVE**: any phrase like "real-time", "~35 fps", "35 FPS"
**REPLACE WITH**: "VeriLens classifies a single video sequence in approximately 3.3 ± 0.6 seconds on GPU hardware"

---

## SECTION 3.1.4 — SVM Configuration (Add all missing hyperparameters)

**ADD the following paragraph** (after description of SVM):

> The SVM classifier uses a Radial Basis Function (RBF) kernel with regularization parameter C = 1.0 and kernel coefficient γ = 'scale' (i.e., γ = 1 / (n_features × X.var())). Probability calibration is enabled (probability=True in scikit-learn). No feature standardization or normalization is applied to the VGG16 feature vectors prior to SVM training. The model is initialized with random_state = 42.

**FIX TYPO**: "AAR Cascade" → "Haar Cascade" (search entire document)

---

## SECTION 3.1.5 — LRCN Architecture (COMPLETE REWRITE)

### Current (WRONG) architecture described:
> Conv1D(64) → MaxPooling1D(2) → LSTM(50) → Dropout(0.5) → Flatten → Dense(50) → Dense(1, sigmoid)

### Replace entire architecture description with:

> The LRCN model receives as input a fixed-length sequence of T = 15 VGG16 feature vectors, each of dimension 512, giving an input tensor of shape (15, 512). The architecture consists of:
> 1. A TimeDistributed Dense layer with 256 units and ReLU activation, applied independently to each of the 15 time steps, projecting features from 512 → 256.
> 2. Dropout(0.3) applied after the TimeDistributed layer.
> 3. An LSTM layer with 128 hidden units (return_sequences=False), consuming the 15 projected features to produce a single 128-dimensional context vector.
> 4. Dropout(0.3) applied after the LSTM.
> 5. A Dense output layer with 2 units and softmax activation, producing class probabilities for [Fake, Real].
>
> Training uses the categorical cross-entropy loss function with the Adam optimizer (learning rate = 1×10⁻⁴), batch size = 2, and 10 epochs. The random seed is fixed at 42 (both NumPy and TensorFlow). No learning rate scheduling or early stopping is applied to the original model.

### Fix loss function reference:
**REMOVE**: "binary cross-entropy", "sigmoid output", any mention of single output node
**REPLACE WITH**: "categorical cross-entropy", "softmax output", "two output units (Fake=0, Real=1)"

---

## SECTION 4.1 — Experimental Setup (Add reproducibility details)

### Add train/test split details:
> Dataset splits are performed at the video level using a stratified 80/20 train/test split (random_state = 42, using scikit-learn's train_test_split). All videos from a given split are processed independently with no frame-level overlap between train and test sets. No validation set is held out from the test set; the LRCN training uses an internal 20% validation split from the training data (i.e., effective split: 64% train, 16% internal-val, 20% test).

### Add evaluation unit clarification:
> All metrics in this paper are reported at the **per-video** level. Each video is represented by a fixed sequence of T = 15 frames sampled at a stride of 15 frames (i.e., one frame per 15 frames of video). The LRCN produces a single class prediction per video sequence via its final softmax layer. The SVM classifies the VGG16 feature vector from the first frame of the sequence. Per-video labels are derived from the dataset ground truth (not aggregated from frame-level predictions).

### Note on ±std values:
> The ±standard deviation values reported in Tables 1–2 reflect variance across the original single training run only; they do not represent multi-seed averaging. [OPTION: Remove ±std entirely if it cannot be justified from multiple seeds]

---

## SECTION 4.2 / EQUATION 6 — Loss Function

**FIND**: Equation 6 (currently shows Sparse Categorical Cross-Entropy formula)
**REPLACE Eq. 6** with categorical cross-entropy:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{1} y_{i,c} \log(\hat{y}_{i,c})$$

where $y_{i,c} \in \{0, 1\}$ is the one-hot encoded ground truth for sample $i$ and class $c$, and $\hat{y}_{i,c}$ is the predicted softmax probability.

**ALSO FIX** Section 4.2 text: Remove "Sparse Categorical Cross-Entropy" → replace with "categorical cross-entropy with one-hot encoded labels"

---

## SECTION 4.3 — Results

### Fix verification paragraph for LRCN (Table 1 column swap fix):
**ADD** this paragraph explicitly showing confusion-matrix-derived metrics:

> To address the labeling inconsistency identified in review, we verify Table 1 metrics directly from the confusion matrices. For the LRCN model (Fig. 11), the confusion matrix yields TN=1127, FP=1, FN=100, TP=18, from which: Precision(Real) = TP/(TP+FP) = 0.9474, Recall(Real) = TP/(TP+FN) = 0.1525, F1(Real) = 0.2628. These values now appear under the LRCN column. For the SVM model (Fig. 12), TN=1128, FP=0, FN=118, TP=0, yielding Precision(Real) = 0, Recall(Real) = 0.0, F1(Real) = 0.0.

### Fix AUC description:
**FIND**: "92.1% of instances" (describing AUC=0.95)
**REPLACE WITH**: "95% of the area under the ROC curve"

### Fix FPS / runtime:
**REMOVE**: "0.42 seconds for 15 frames", "0.028 fps", "nearly real-time (~35 fps)"
**REPLACE WITH**:
> Pipeline latency was measured over 10 videos. Mean per-video classification time: face detection 0.3516 ± 0.0216 s, VGG16 forward pass 0.1487 ± 0.0779 s, LRCN classifier 0.0564 ± 0.0591 s, total 0.5567 ± 0.1381 s. GPU acceleration (CUDA) is used for VGG16 feature extraction and LRCN inference; face detection runs on CPU via OpenCV Haar cascade.

### Fix Section 4.3 cross-reference:
**FIND**: "Section 3.1" (reference to dataset description)
**REPLACE WITH**: "Section 4.1"

### Add ablation note (for Reviewer #7):
> The LRCN's high accuracy but low recall for the Real class reflects the effect of class imbalance (approximately 9:1 Fake:Real ratio in Celeb-DF v2). The model defaults to predicting Fake for most samples, achieving high accuracy trivially. To address this, we additionally report a retrained variant using focal loss (γ=2.0) and balanced mini-batches, which substantially improves balanced accuracy at the cost of overall accuracy.

---

## TABLE 1 — Celeb-DF Results

### Column header swap (CRITICAL):
The LRCN and SVM column headers must be swapped. The numbers stay in place.
- What was labeled "SVM" → relabel as "LRCN"
- What was labeled "LRCN" → relabel as "SVM"

### Add balanced accuracy row:
| Metric | LRCN | SVM | DenseNet* | VeriFace* |
| Balanced Accuracy | 0.5758 | 0.5000 | — | — |

### Footnote for baselines:
Add footnote: "* Results reported from original publications [ref]; evaluation protocol may differ from ours."

---

## TABLE 2 — DFDC Results

### Protocol footnote:
Add: "Results use per-video evaluation (majority vote aggregation over 15 sampled frames)."

### Verify column assignment:
Confirm LRCN vs SVM columns are correct by cross-checking with DFDC confusion matrix figures.

---

## SECTION 1 / INTRODUCTION — Remove overclaims

**FIND AND REMOVE/SOFTEN**:
- "eye blinking detection" — remove or add "inspired by concepts from liveness detection literature"
- "face recognition" as a system component — soften to "face localization via Haar cascade"
- Any claim that the system runs in real-time for continuous video streams

---

## EVERYWHERE IN PAPER — Global fixes

1. **"AAR Cascade"** → **"Haar Cascade"** (all occurrences)
2. **"binary cross-entropy"** → **"categorical cross-entropy"** (all occurrences)
3. **"sigmoid"** output → **"softmax"** output (in context of LRCN)
4. **"Dense(1)"** or **"single output"** → **"Dense(2)"** or **"two-class output"** (LRCN description)
5. **"Sparse Categorical Cross-Entropy"** → **"categorical cross-entropy"**
6. **"real-time"** → remove or qualify appropriately
7. All mentions of 0.028 fps and 35 fps → remove, replace with per-video latency table

---

## NEW CONTENT TO ADD (Reviewer Requests)
Reviewer #7: There are some week claims
The reported processing speed (0.42 s for 15 frames) is extrapolated to "~35 FPS," which is confusing
This will not result to real-time video processing in standard multimedia terms. (at least 30 fps) (even 60 fps in 1080p)
Some sections still exhibit thesis-style.
Add balanced accuracy, PR curves, or cost-weighted loss analysis.
No threshold analysis is provided,
No cost-sensitive learning or class-imbalance mitigation is explored,
No ablation explains why LRCN collapses in recall while maintaining high accuracy.



### Add to Section 4.3: Threshold Analysis
> Figure [N] shows the precision, recall, F1, and balanced accuracy of the LRCN as a function of the decision threshold. At the default threshold of 0.5, the model achieves high precision but low recall for the Real class due to class imbalance. Optimal F1 for the Real class is achieved at threshold = [best_t] (from results/lrcn_original/metrics.json).

### Add to Section 4.3: Precision-Recall Curve
> Figure [N] shows the precision-recall curve for both classifiers. The area under the PR curve (Average Precision) for the LRCN is [AP], and for the SVM is [AP].

### Add caption reference for new figures:
- ROC curves → figures from results/*/ROC_curve.png
- PR curves → figures from results/*/PR_curve.png
- Threshold analysis → figures from results/*/threshold_analysis.png
- Training history → figures from results/*/training_history.png

---

## CHECKLIST — Before Submission

- [ ] Table 1 column headers swapped (LRCN ↔ SVM)
- [ ] Eq. 6 updated to categorical cross-entropy
- [ ] Section 3.1.5 architecture matches code
- [ ] Section 3.1.4 has all SVM hyperparameters
- [ ] Section 4.1 has train/val/test ratios and evaluation unit
- [ ] Section 4.3 has verification paragraph with confusion-matrix-derived metrics
- [ ] FPS claim removed / replaced with latency table
- [ ] "AAR Cascade" typo fixed everywhere
- [ ] Section 4.3 cross-reference changed from "Section 3.1" to "Section 4.1"
- [ ] AUC 0.95 described as "95%" not "92.1%"
- [ ] Eye-blinking / face-recognition overclaims softened
- [ ] Baseline comparison footnote added
- [ ] Balanced accuracy added to tables
- [ ] Threshold analysis figure added
- [ ] PR curve figure added
- [ ] Response-to-reviewers document completed
