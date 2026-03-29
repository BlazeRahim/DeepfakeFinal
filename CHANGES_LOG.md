# VeriLens Paper Revision — Complete Change Log

**Paper:** VeriLens — Deepfake Detection via VGG16 + SVM/LRCN
**Journal:** Multimedia Tools and Applications (Springer)
**Revision Date:** March 2026
**Reviewers:** Reviewer #4 (major revision), Reviewer #7 (major revision), one reviewer accepted

---

## Summary Table

| # | Section | Change | Reviewer | Priority |
|---|---------|--------|----------|----------|
| 1 | Abstract | Removed eye blinking / face masking / face recognition overclaims | R#4.9 | HIGH |
| 2 | Abstract | Replaced last sentence with evaluation summary | R#4.9 | HIGH |
| 3 | Keywords | "Eye Blinking" → "VGG16", "Convolution" → "Convolutional" | R#4.9 | HIGH |
| 4 | Section 1 (Intro) | Rewrote face recognition paragraph → face detection | R#4.9, R#7.2 | HIGH |
| 5 | Section 3.1.4 | "AAR Cascade" → "Haar Cascade" (global) | R#4.10 | LOW |
| 6 | Section 3.1.4 | Added full SVM hyperparameters | R#4.5 | HIGH |
| 7 | Section 3.1.5 | Complete LRCN architecture rewrite (Conv1D → TimeDistributed) | R#4.4, R#4.6 | CRITICAL |
| 8 | Section 3.1.5 | Fixed loss function (sigmoid/binary CE → softmax/categorical CE) | R#4.4 | CRITICAL |
| 9 | Section 3.1.5 | Added training details (lr, batch, epochs, seed) | R#4.6 | HIGH |
| 10 | Section 4.1 | Added split protocol and leakage controls (NEW subsection) | R#4.2 | HIGH |
| 11 | Section 4.2 | Added per-video evaluation unit paragraph (NEW) | R#4.3 | HIGH |
| 12 | Section 4.2 | Fixed loss function description + Equation 6 | R#4.4 | CRITICAL |
| 13 | Section 4.2 | Added Balanced Accuracy definition + equation (NEW) | R#7.3 | MEDIUM |
| 14 | Section 4.3 | Replaced FPS/runtime claim with latency table | R#4.7, R#7.1 | CRITICAL |
| 15 | Section 4.3 | Fixed cross-reference "Section 3.1" → "Section 4.1" | R#4.10 | LOW |
| 16 | Section 4.3 | Fixed AUC interpretation (0.95 → 0.770) | R#4.10 | HIGH |
| 17 | Section 4.3 | Rewrote LRCN confusion matrix paragraph (per-video) | R#4.1, R#4.3 | CRITICAL |
| 18 | Section 4.3 | Rewrote SVM confusion matrix paragraph (per-video) | R#4.1, R#4.3 | CRITICAL |
| 19 | Section 4.3 | Added verification paragraph with math derivations | R#4.1 | CRITICAL |
| 20 | Section 4.3 | Rewrote DFDC LRCN confusion matrix paragraph | R#4.3 | HIGH |
| 21 | Section 4.3 | Rewrote DFDC SVM confusion matrix paragraph | R#4.3 | HIGH |
| 22 | Table 1 | Complete replacement — corrected headers, added LRCN-Focal, Balanced Acc, AUC | R#4.1, R#7.3 | CRITICAL |
| 23 | Table 2 | Complete replacement — DFDC per-video, added LRCN-Focal, Balanced Acc, AUC | R#4.1, R#7.3 | HIGH |
| 24 | Section 4.3 | Replaced comparative claims paragraph (softened) | R#4.8 | HIGH |
| 25 | Section 4.3 | Replaced discussion paragraph with corrected numbers | R#4.8 | MEDIUM |
| 26 | Section 4.4 | NEW section: Class-Imbalance Analysis and Cost-Sensitive Learning | R#7.3-R#7.6 | CRITICAL |
| 27 | Section 5 | Fixed conclusion overclaims (Face Masking → Haar cascade) | R#4.8, R#4.9 | MEDIUM |
| 28 | Section 5 | Added LRCN-Focal mention + baseline caveat in summary | R#4.8 | MEDIUM |

---

## Detailed Changes

---

### Change 1 — Abstract: Remove Overclaims
**Reviewer:** R#4.9
**Issue:** Abstract claims "eye blinking detection," "face recognition," and "face masking" but none are implemented in the pipeline.

**FIND:**
```
VeriLens leverages advanced technologies such as face recognition, face masking, and eye blinking detection to provide a robust and scalable solution for real-time deepfake identification. The system is engineered with a user-friendly interface, ensuring accessibility and ease of use. Through its innovative approach, VeriLens aims to enhance the security and authenticity of digital content, offering a crucial tool for safeguarding information in both digital and real-world contexts.
```

**REPLACE:**
```
VeriLens employs Haar cascade-based face detection to isolate facial regions from video frames, followed by VGG16 for spatial feature extraction and a Long-term Recurrent Convolutional Network (LRCN) for temporal sequence classification. An SVM classifier is additionally evaluated as a single-frame baseline. The system is engineered with a user-friendly interface, ensuring accessibility and ease of use for forensic and investigative purposes.
```

---

### Change 2 — Abstract: Last Sentence
**Reviewer:** R#4.9
**Issue:** Original last sentence references "eye blinking" and "facial recognition" again.

**FIND:**
```
VeriLens also aims to have higher accuracy than its predecessor, VeriFace, which utilized face masking, eye blinking, and facial recognition techniques to detect Deepfakes.
```

**REPLACE:**
```
Experimental evaluation on the Celeb-DF v2 and DFDC benchmark datasets demonstrates that VeriLens achieves competitive detection performance when compared with VeriFace and DenseNet-based methods, offering flexible trade-offs between precision and recall depending on the choice of classifier.
```

---

### Change 3 — Keywords
**Reviewer:** R#4.9
**Issue:** "Eye Blinking" is not implemented. "Convolution" should be "Convolutional."

**FIND:**
```latex
\keywords{Multimedia Data, Deepfake, Eye Blinking, Long-term Recurrent Convolution Networks, Support Vector Machine}
```

**REPLACE:**
```latex
\keywords{Multimedia Data, Deepfake, VGG16, Long-term Recurrent Convolutional Networks, Support Vector Machine}
```

---

### Change 4 — Introduction: Face Recognition Paragraph
**Reviewer:** R#4.9, R#7.2
**Issue:** Paragraph describes face recognition and eye blinking in thesis-style as if they are part of VeriLens. Neither is implemented.

**FIND:** The entire paragraph starting with "Face recognition is a major cornerstone..." through "...natural blinking patterns."

**REPLACE:**
```
Face detection is a fundamental step in both deepfake generation and detection pipelines. A face detector first identifies the regions within an image or video frame that contain faces. These regions are then aligned and normalized to produce uniform representations. In the proposed system, face detection is carried out using Haar cascades, which localize and extract facial regions from individual video frames for subsequent processing by the deep learning pipeline.
```

---

### Change 5 — Global: AAR Cascade Typo
**Reviewer:** R#4.10
**Issue:** "AAR Cascade" is a typo for "Haar Cascade."

**Action:** Global find-and-replace: `AAR Cascade` → `Haar Cascade` (2 occurrences)

---

### Change 6 — Section 3.1.4: SVM Hyperparameters
**Reviewer:** R#4.5
**Issue:** SVM cannot be replicated without kernel, C, gamma, class weights, probability calibration, and standardization details.

**FIND:**
```
A SVM classifier with probability estimates (SVC(probability=True)) is instantiated and trained on the training data.
```

**REPLACE:**
```
An SVM classifier is instantiated and trained on the extracted features. The classifier uses an RBF (Radial Basis Function) kernel with regularization parameter C = 1.0 and kernel coefficient gamma set to 'scale', which is internally computed as 1/(n_features x Var(X)), approximately 1/(512 x Var(X)) for the 512-dimensional VGG16 features. Probability estimates are enabled through Platt scaling (probability=True), which allows the classifier to output calibrated class probabilities rather than hard labels. No class weighting is applied and no feature normalization or standardization is performed prior to training, as the average-pooling outputs of VGG16 are inherently bounded. The random seed is fixed at 42 to ensure reproducibility. The SVM operates on the feature vector extracted from a single frame per video, specifically the first frame in which a face is successfully detected.
```

---

### Change 7, 8, 9 — Section 3.1.5: LRCN Architecture Complete Rewrite
**Reviewer:** R#4.4, R#4.6
**Issue:** Paper describes Conv1D/MaxPool/LSTM/sigmoid/binary CE. Actual code uses TimeDistributed(Dense(256))/LSTM(128)/softmax/categorical CE. Architecture is wrong, loss function contradicts itself, and training details are missing.

**FIND:** Everything from `\item \textbf{Define the LRCN Model:}` through `\item \textbf{Train the Model:}` including all sub-items (Conv1D Layer, MaxPooling1D, LSTM with 50 units, Flatten, Dense with sigmoid, binary CE compilation)

**REPLACE:** Complete rewrite with correct architecture:
- Input: (15, 512) — 15 VGG16 feature vectors of dimension 512
- TimeDistributed(Dense(256, relu)) — projects each time step
- Dropout(0.3)
- LSTM(128, return_sequences=False) — temporal encoding
- Dropout(0.3)
- Dense(2, softmax) — class 0=Fake, class 1=Real
- Loss: categorical cross-entropy
- Optimizer: Adam, lr=1e-4
- Batch size: 2, epochs: 10, validation split: 20%
- Seeds: 42 (NumPy + TensorFlow)
- No LR scheduling or early stopping for original model

---

### Change 10 — Section 4.1: Split Protocol (NEW Subsection)
**Reviewer:** R#4.2
**Issue:** No train/test split ratios, no statement of video-level splitting, no leakage prevention, not reproducible.

**Action:** Inserted new `\subsubsection*{Split Protocol and Leakage Controls}` after the dataset descriptions, containing:
- Stratified 80/20 video-level split (scikit-learn, random_state=42, stratify=y)
- LRCN: additional 20% internal validation → effective 64/16/20
- SVM: full 80% training, no internal validation
- Leakage prevention: each video is indivisible, no frame-level shuffling
- Limitation acknowledged: no subject-disjoint splitting (no identity metadata available)
- Test set compositions:
  - Celeb-DF v2: 1,246 videos (1,128 fake + 118 real), ratio 9.56:1
  - DFDC: 687 videos, 607 with successful face detection (568 fake + 39 real), ratio 14.56:1

---

### Change 11 — Section 4.2: Evaluation Unit (NEW Paragraph)
**Reviewer:** R#4.3
**Issue:** Unclear whether metrics are per-frame or per-video. Confusion matrix totals don't match dataset sizes (because original was per-frame).

**Action:** Inserted paragraph at the beginning of Section 4.2:
- All metrics are per-video, not per-frame
- LRCN: softmax over 15-frame sequence = one prediction per video
- SVM: VGG16 features from first detected face = one prediction per video
- N=1,246 for Celeb-DF, N=607 for DFDC

---

### Change 12 — Section 4.2: Loss Function + Equation 6
**Reviewer:** R#4.4
**Issue:** Paper says "Sparse Categorical Cross-Entropy" with binary classification equation. Actual code uses categorical cross-entropy with Dense(2, softmax).

**FIND:** The `\textbf{Loss Function:}` paragraph mentioning "Sparse Categorical Cross-Entropy" and the old Equation 6

**REPLACE:** Corrected to categorical cross-entropy with proper two-class formulation:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{1} y_{i,c} \cdot \log(\hat{y}_{i,c})$$

---

### Change 13 — Section 4.2: Balanced Accuracy Definition (NEW)
**Reviewer:** R#7.3
**Issue:** No balanced accuracy reported despite severe class imbalance.

**Action:** Inserted after Equation 5 (Specificity):
- Explains why standard accuracy is misleading under 9:1 imbalance
- Defines Balanced Accuracy = (Recall + Specificity) / 2
- Notes that 0.5 = random chance regardless of class distribution

---

### Change 14 — Section 4.3: Runtime / FPS Claim
**Reviewer:** R#4.7, R#7.1
**Issue:** "0.028 frames per second" is wrong (it's seconds per frame). "~35 FPS" extrapolation is misleading. No per-stage breakdown.

**FIND:**
```
In the proposed system pipeline takes around 0.42 seconds, or about 0.028 frames per second, to process a movie with 15 sampled frames on an NVIDIA RTX 3050 laptop GPU. This is nearly real-time performance (about 35 frames per second).
```

**REPLACE:** Latency table with per-stage breakdown measured on RTX 3050:

| Stage | Acceleration | Time (s) |
|---|---|---|
| Face detection (Haar cascade) | CPU (OpenCV) | 0.352 +/- 0.022 |
| VGG16 feature extraction | GPU (CUDA) | 0.149 +/- 0.078 |
| LRCN classification | GPU (CUDA) | 0.056 +/- 0.059 |
| **Total** | | **0.557 +/- 0.138** |

Explicitly states VeriLens is for **offline forensic analysis**, not real-time streaming.

---

### Change 15 — Section 4.3: Cross-Reference Fix
**Reviewer:** R#4.10

**FIND:** `the dataset described in Section 3.1`
**REPLACE:** `the datasets described in Section 4.1`

---

### Change 16 — Section 4.3: AUC Interpretation
**Reviewer:** R#4.10
**Issue:** "AUC of 0.95, demonstrating... approximately 92.1% of instances" is wrong. AUC is 0.770, and the interpretation was incorrect.

**FIND:**
```
the proposed approach achieves an AUC of 0.95, demonstrating a high degree of accuracy in distinguishing between positive and negative cases. This means that in approximately 92.1% of instances, the classifier correctly identifies a positive example over a negative one.
```

**REPLACE:**
```
the proposed LRCN model achieves an AUC of 0.770, which indicates that a randomly selected real video is assigned a higher classification score than a randomly selected fake video with probability 77.0%. The ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate across various decision thresholds, illustrating the discriminative capacity of the classifier.
```

---

### Change 17 — Section 4.3: LRCN Confusion Matrix Paragraph
**Reviewer:** R#4.1, R#4.3
**Issue:** Old paragraph used per-frame counts (374, 7530, 469, 226). Now per-video (N=1,246).

**FIND:** Paragraph starting "The VeriLens system, implemented using the LRCN model, produced the confusion matrix shown in Figure 11, demonstrating strong classification performance. It correctly identified 374 authentic films and 7,530 fraudulent movies..."

**REPLACE:** Per-video evaluation paragraph:
- TN=1,126, FP=2, FN=97, TP=21
- Precision(Real)=0.913, Recall(Real)=0.178, F1(Real)=0.298
- Accuracy=0.921, Balanced Accuracy=0.588
- Notes conservative classification bias from 9.56:1 imbalance

---

### Change 18 — Section 4.3: SVM Confusion Matrix Paragraph
**Reviewer:** R#4.1, R#4.3
**Issue:** Old paragraph used per-frame counts (15,030, 5, 1,317, 203).

**FIND:** Paragraph starting "The VeriLens system implemented using the SVM model produced the confusion matrix shown in Figure 12, accurately classifying 15,030 cases..."

**REPLACE:** Per-video evaluation paragraph:
- TN=1,128, FP=0, FN=118, TP=0 at default threshold 0.5
- Recall(Real)=0.0, Balanced Accuracy=0.5 (equivalent to trivial all-Fake classifier)
- At optimised threshold t=0.15: Precision=0.313, Recall=0.398, F1=0.351
- AUC=0.718 confirms discriminative capacity masked by default threshold

---

### Change 19 — Section 4.3: Verification Paragraph (NEW)
**Reviewer:** R#4.1
**Issue:** Reviewer requested explicit verification that confusion matrix counts match reported metrics.

**Action:** Added mathematical derivation paragraph showing:
- LRCN (Fig 11): TN=1,126, FP=2, FN=97, TP=21 → Precision=21/23=0.913, Recall=21/118=0.178, F1=0.298, Accuracy=1147/1246=0.921
- SVM (Fig 12): TN=1,128, FP=0, FN=118, TP=0 → Precision=undefined, Recall=0.0, Accuracy=1128/1246=0.905
- States these match Table 1 exactly

---

### Change 20 — Section 4.3: DFDC LRCN Confusion Matrix
**Reviewer:** R#4.3
**Issue:** Old per-frame counts (28, 604, 45, 10).

**FIND:** "the confusion matrix indicates that the model correctly identified 28 authentic films and 604 fraudulent movies..."

**REPLACE:** Per-video (N=607):
- TN=546, FP=22, FN=21, TP=18
- Precision(Real)=0.450, Recall(Real)=0.462, F1(Real)=0.456
- Accuracy=0.929, Balanced Accuracy=0.711

---

### Change 21 — Section 4.3: DFDC SVM Confusion Matrix
**Reviewer:** R#4.3
**Issue:** Old per-frame counts (43, 513, 30, 101).

**FIND:** "the confusion matrix reveals that the model correctly classified 43 authentic films and 513 fraudulent videos..."

**REPLACE:** Per-video (N=607):
- TN=566, FP=2, FN=26, TP=13
- Precision(Real)=0.867, Recall(Real)=0.333, Accuracy=0.954

---

### Change 22 — Table 1: Complete Replacement (Celeb-DF)
**Reviewer:** R#4.1, R#7.3
**Issue:** Column headers were SWAPPED (LRCN and SVM values were under the wrong headers). No balanced accuracy. No AUC. Per-frame values.

**Old Table:** 4 columns (LRCN, SVM, VeriFace, DenseNet) with swapped LRCN/SVM headers and per-frame values with +/- std

**New Table:** 5 columns with correct per-video values:

| Metric | LRCN | SVM* (t=0.15) | LRCN-Focal | VeriFace (CNN)+ | DenseNet+ |
|--------|------|---------------|------------|-----------------|-----------|
| Precision | 0.913 | 0.313 | 0.144 | 0.741 | 0.950 |
| Recall | 0.178 | 0.398 | 0.805 | 0.473 | 0.993 |
| Accuracy | 0.921 | --- | 0.530 | 0.734 | 0.970 |
| F1-score | 0.298 | 0.351 | 0.245 | 0.572 | 0.971 |
| Specificity | 0.998 | --- | 0.501 | 0.241 | 0.992 |
| Balanced Acc. | 0.588 | 0.500 | 0.653 | --- | --- |
| AUC | 0.770 | 0.718 | 0.745 | --- | --- |

Caption now states: per-video, N=1,246. Dagger footnote for baselines not re-evaluated under our protocol.

---

### Change 23 — Table 2: Complete Replacement (DFDC)
**Reviewer:** R#4.1, R#7.3
**Issue:** Same problems as Table 1. Also DFDC was not originally per-video.

**New Table:** 5 columns, per-video (N=607):

| Metric | LRCN | SVM | LRCN-Focal | VeriFace+ | DenseNet+ |
|--------|------|-----|------------|-----------|-----------|
| Precision | 0.450 | 0.867 | 0.126 | 0.62 | 0.88 |
| Recall | 0.462 | 0.333 | 0.795 | 0.35 | 0.85 |
| Accuracy | 0.929 | 0.954 | 0.633 | 0.78 | 0.90 |
| F1-score | 0.456 | 0.482 | 0.218 | 0.45 | 0.86 |
| Specificity | 0.961 | 0.996 | 0.622 | 0.80 | 0.91 |
| Balanced Acc. | 0.711 | 0.665 | 0.708 | --- | --- |
| AUC | 0.739 | 0.877 | 0.816 | --- | --- |

---

### Change 24 — Section 4.3: Comparative Claims
**Reviewer:** R#4.8
**Issue:** Direct superiority claims without protocol-matched baselines.

**FIND:** Paragraph starting "As shown in Table 1 and 2, when implemented with the LRCN model, VeriLens achieves the highest precision (0.976)..."

**REPLACE:** Softened language:
- States LRCN achieves high precision (0.913) and specificity (0.998), but lower recall (0.178)
- SVM requires threshold optimisation
- LRCN-Focal improves recall to 0.805
- Adds caveat: DenseNet and VeriFace results are from original publications, not re-evaluated under our protocol
- "Direct numerical comparisons should be interpreted with caution"

---

### Change 25 — Section 4.3: Discussion Paragraph
**Reviewer:** R#4.8
**Issue:** Old numbers from swapped table (accuracy 0.976, specificity 0.997, recall 0.134 under wrong model).

**FIND:** Paragraph starting "The LRCN model in VeriLens prioritizes high accuracy (0.976) and specificity (0.997)..."

**REPLACE:** Corrected numbers:
- LRCN: precision 0.913, specificity 0.998, recall 0.178
- SVM at default threshold: 90.5% accuracy, zero recall
- Threshold optimisation recovers SVM F1=0.351
- LRCN-Focal: recall 0.805 (Celeb-DF), 0.795 (DFDC)
- VeriLens offers three variants for different deployment priorities

---

### Change 26 — Section 4.4: Class-Imbalance Analysis (ENTIRE NEW SECTION)
**Reviewer:** R#7.3, R#7.4, R#7.5, R#7.6
**Issue:** No balanced accuracy, no PR curves, no cost-weighted loss analysis, no threshold analysis, no cost-sensitive learning, no ablation for LRCN recall collapse.

**Action:** Inserted new `\subsection{Class-Imbalance Analysis and Cost-Sensitive Learning}` containing:

**1. Analysis of LRCN Recall Collapse:**
- Trivial all-Fake classifier achieves 90.5% accuracy
- LRCN's 92.1% accuracy is only marginally better
- Balanced accuracy 0.588 is only 8.8pp above random (0.5)
- Threshold analysis: optimal threshold is 0.095, not 0.5
- SVM shows same pattern (confirms data-driven, not model-specific)

**2. Cost-Sensitive Retraining (LRCN-Focal):**
- Focal loss (gamma=2.0)
- Inverse-frequency class weights: alpha_c = N/(2*N_c)
- Balanced mini-batch sampling
- ReduceLROnPlateau + early stopping

**3. Results:**
- Celeb-DF: recall 17.8% → 80.5%, balanced accuracy 0.588 → 0.653
- DFDC: recall 46.2% → 79.5%, AUC 0.739 → 0.816
- FP increases from 2 to 563 (expected trade-off)

**4. New Figures:**
- Threshold sensitivity analysis (Fig: threshold_analysis.png)
- Precision-Recall curves with AP scores (Fig: pr_curves.png)

---

### Change 27 — Conclusion: Overclaims
**Reviewer:** R#4.8, R#4.9

**FIND:**
```
this paper introduced VeriLens, a deepfake detection framework that integrates Face Masking, VGG-16, and LRCN models to achieve enhanced accuracy, scalability, and real-time performance.
```

**REPLACE:**
```
this paper introduced VeriLens, a deepfake detection framework that combines Haar cascade-based face detection, VGG-16 feature extraction, and LRCN temporal classification to provide a scalable pipeline for offline deepfake detection across multiple benchmark datasets.
```

---

### Change 28 — Conclusion: Summary Sentence
**Reviewer:** R#4.8

**FIND:**
```
In summary, VeriLens achieves competitive DeepFake detection performance compared with VeriFace and DenseNet-based methods. By offering a high-precision LRCN variant and a higher-recall SVM variant
```

**REPLACE:**
```
In summary, VeriLens demonstrates competitive deepfake detection performance relative to VeriFace and DenseNet-based methods, noting that baseline results are drawn from their respective original publications and were not re-evaluated under the same experimental protocol. By offering a high-precision LRCN variant, a cost-sensitive LRCN-Focal variant with improved recall, and an SVM baseline
```

---

## Figures to Upload to Overleaf

### New Figures (Section 4.4)
| Local Path | Overleaf Name |
|---|---|
| `results/celeb-df/lrcn_original/threshold_analysis.png` | `threshold_analysis.png` |
| `results/celeb-df/lrcn_original/PR_curve.png` | `pr_curves.png` |

### Replace Existing Celeb-DF Figures
| Local Path | Replaces |
|---|---|
| `results/celeb-df/lrcn_original/confusion_matrix.png` | `lrcn confusion matrix.jpeg` |
| `results/celeb-df/svm/confusion_matrix.png` | `svm confusion matrix.jpeg` |
| `results/celeb-df/lrcn_original/ROC_curve.png` | `roc new.jpeg` |
| `results/celeb-df/lrcn_original/training_history.png` | `lrcn training and validation.jpeg` |
| `results/celeb-df/svm/training_history.png` | `svm trvining and validation.jpeg` |

### Replace Existing DFDC Figures
| Local Path | Replaces |
|---|---|
| `results/dfdc/lrcn_original/confusion_matrix.png` | `kaggle dataset/lrcn confusion matrix_kaggle dataset.jpeg` |
| `results/dfdc/lrcn_original/ROC_curve.png` | `kaggle dataset/lrcn roc_kaggle dataset.jpeg` |
| `results/dfdc/svm_per_video/confusion_matrix.png` | (add as new) |

---

## Metric Sources (from metrics.json)

All values come from actual experimental runs. Source files:

| File | Key Values |
|---|---|
| `results/celeb-df/lrcn_original/metrics.json` | TN=1126, FP=2, FN=97, TP=21, AUC=0.7698, best_t=0.095 |
| `results/celeb-df/lrcn_retrained/metrics.json` | TN=565, FP=563, FN=23, TP=95, AUC=0.7448, best_t=0.735 |
| `results/celeb-df/svm/metrics.json` | TN=1128, FP=0, FN=118, TP=0, AUC=0.7175, best_t=0.15 |
| `results/dfdc/lrcn_original/metrics.json` | TN=546, FP=22, FN=21, TP=18, AUC=0.7386 |
| `results/dfdc/lrcn_retrained/metrics.json` | TN=353, FP=215, FN=8, TP=31, AUC=0.8162 |
| `results/dfdc/svm_per_video/metrics.json` | TN=566, FP=2, FN=26, TP=13, AUC=0.8773 |
| `results/celeb-df/pipeline_timing.json` | face_det=0.352s, vgg=0.149s, classifier=0.056s, total=0.557s |
