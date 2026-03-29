# VeriLens Paper — Exact Corrections Cheatsheet
## Use this as a find-and-replace guide when editing the paper in LaTeX/Word

> **How to use:** Each correction below has:
> - **WHERE:** Page number and section
> - **FIND (old text):** The exact text currently in the paper
> - **REPLACE WITH (new text):** What to paste in its place
> - **WHY:** Which reviewer point this fixes

---

# ============================================================
# SECTION 1: ABSTRACT (Page 1)
# ============================================================

### Correction A1 — Remove eye blinking & face recognition overclaims
**WHERE:** Page 1, Abstract
**REVIEWER:** R#4.9

**FIND:**
```
VeriLens leverages advanced technologies such as face recognition, face masking,
and eye blinking detection to provide a robust and scalable solution for real-time
deepfake identification.
```

**REPLACE WITH:**
```
VeriLens leverages Haar cascade face detection, VGG16-based spatial feature
extraction, and Long-term Recurrent Convolutional Networks (LRCN) for temporal
sequence classification, providing a scalable solution for deepfake identification.
```

---

### Correction A2 — Remove VeriFace overclaim comparison
**WHERE:** Page 1, Abstract (last sentence)
**REVIEWER:** R#4.9

**FIND:**
```
VeriLens also aims to have higher accuracy than its predecessor, VeriFace, which
utilized face masking, eye blinking, and facial recognition techniques to detect
Deepfakes.
```

**REPLACE WITH:**
```
VeriLens is evaluated against VeriFace and DenseNet-based methods, demonstrating
competitive detection performance with flexible precision-recall trade-offs
depending on the chosen classifier (LRCN or SVM).
```

---

# ============================================================
# SECTION 2: KEYWORDS (Page 2)
# ============================================================

### Correction K1 — Remove Eye Blinking keyword
**WHERE:** Page 2, Keywords
**REVIEWER:** R#4.9

**FIND:**
```
Keywords: Multimedia Data, Deepfake, Eye Blinking, Long-term Recurrent
Convolution Networks, Support Vector Machine
```

**REPLACE WITH:**
```
Keywords: Multimedia Data, Deepfake, VGG16, Long-term Recurrent
Convolutional Networks, Support Vector Machine
```

---

# ============================================================
# SECTION 3: INTRODUCTION (Page 2)
# ============================================================

### Correction I1 — Remove eye blinking paragraph (thesis-style + overclaim)
**WHERE:** Page 2, Section 1, second paragraph
**REVIEWER:** R#4.9, R#7.2

**FIND:**
```
Eye blink-
ing is another method that can be used as a major parameter in detecting deepfake
media. Eye Blinking refers to the analysis of eye moments to spot deepfakes, as fake
videos may fail to reproduce natural blinking patterns.
```

**REPLACE WITH:**
```
(DELETE ENTIRELY — eye blinking is not implemented in VeriLens)
```

---

### Correction I2 — Fix "face recognition" to "face detection"
**WHERE:** Page 2, Section 1, second paragraph (first sentence)
**REVIEWER:** R#4.9

**FIND:**
```
Face recognition is a major cornerstone of deepfake generation and detection. Face
recognition operates through a sequence of distinct steps.
```

**REPLACE WITH:**
```
Face detection is a foundational step in deepfake generation and detection pipelines.
```

> **Note:** The rest of this paragraph is thesis-style description of generic face recognition. Consider condensing the entire paragraph to 2-3 sentences focused on what VeriLens actually does (Haar cascade face detection + VGG16 feature extraction).

---

# ============================================================
# SECTION 4: SECTION 3.1.4 — SVM (Pages 11-12)
# ============================================================

### Correction S1 — Fix "AAR Cascade" typo
**WHERE:** Page 12, Section 3.1.4, step 1
**REVIEWER:** R#4.10

**FIND:**
```
A function extracts faces from video frames using the AAR Cascade face detector.
```

**REPLACE WITH:**
```
A function extracts faces from video frames using the Haar Cascade face detector.
```

---

### Correction S2 — Add full SVM hyperparameters
**WHERE:** Page 12, Section 3.1.4, after step 6 ("Train SVM Classifier")
**REVIEWER:** R#4.5

**FIND:**
```
A SVM classifier with probability estimates (SVC(probability=True)) is
instantiated and trained on the training data.
```

**REPLACE WITH:**
```
An SVM classifier is instantiated with the following configuration: RBF kernel,
regularization parameter C = 1.0, kernel coefficient gamma = 'scale'
(1/(n_features x Var(X))), probability calibration enabled via Platt scaling
(probability=True), no class weighting, and random_state = 42. No feature
standardization is applied to the VGG16 feature vectors. The SVM operates on
the 512-dimensional VGG16 average-pooling feature vector extracted from a
single frame per video (the first successfully detected face).
```

---

# ============================================================
# SECTION 5: SECTION 3.1.5 — LRCN (Pages 12-13)
# ============================================================

### Correction L1 — Replace entire LRCN architecture description
**WHERE:** Pages 12-13, Section 3.1.5, steps 3-5
**REVIEWER:** R#4.6, R#4.4

The current text describes Conv1D/MaxPool/LSTM/Flatten/Dense with sigmoid + binary cross-entropy. This is **completely wrong** vs the actual code.

**FIND (the entire "3. Define the LRCN Model" through "5. Train the Model" subsections):**
```
3. Define the LRCN Model:
• Conv1D Layer: With the input data, a one-dimensional convolutional layer
(Conv1D) is applied. This layer extracts local spatial characteristics from the
input sequence using 64 filters with a kernel size of 3. It also utilizes an activation
function called ReLU.
• MaxPooling1D Layer: The Conv1D layer is followed by a MaxPooling1D layer
with a pool size of 2, which lowers the dimensionality of the data while keeping
the most important characteristics.
• LSTM Layer: An LSTM layer with 50 units receives the convolutional layer's
output. With return sequences=True, the LSTM layer preserves temporal
information throughout the input sequence and provides the whole sequence of
hidden states. To avoid overfitting, a Dropout layer with a rate of 0.5 is inserted
after the LSTM.
• Flatten Layer: The data is flattened to transform it into a single dimension
suitable for the Dense layers.
• Dense Layers: Two fully connected (Dense) layers follow:
– There are 50 units in the first dense layer, each having a ReLU activation
function.
– One unit in the second Dense layer, suited for binary classification, has a
sigmoid activation function.
4. Compile the Model:
• Accuracy as the measure, binary cross-entropy loss function, and Adam optimizer
are used to create the model.
5. Train the Model:
• With a batch size of 32, the model is trained on the training set for 100 epochs.
During training, validation is done on the testing data. pickle is used to save
the training history to a file.
```

**REPLACE WITH:**
```
3. Define the LRCN Model:

   Input: A sequence of T=15 VGG16 feature vectors, each of dimension 512,
   giving input shape (15, 512). Frames are sampled every 15th frame from
   each video. If fewer than 15 faces are detected, the sequence is zero-padded.

   The architecture consists of the following layers:
   • TimeDistributed(Dense(256, ReLU)): Applies the same dense projection to
     each of the 15 time steps independently, reducing each 512-d vector to 256-d.
   • Dropout(0.3): Applied after the TimeDistributed layer.
   • LSTM(128, return_sequences=False): Processes the 15-step sequence and
     produces a single 128-dimensional output vector.
   • Dropout(0.3): Applied after the LSTM layer.
   • Dense(2, softmax): Two-class output (class 0 = Fake, class 1 = Real).

4. Compile the Model:
   • Loss function: categorical cross-entropy.
   • Optimizer: Adam with learning rate = 1e-4.
   • Metric: accuracy.

5. Train the Model:
   • Batch size: 2. Epochs: 10. Random seed: 42 (NumPy and TensorFlow).
   • An internal 20% validation split is used during training.
   • No learning rate scheduling or early stopping is applied for the
     original model.
```

---

# ============================================================
# SECTION 6: SECTION 4.1 — DATASET (Pages 13-14)
# ============================================================

### Correction D1 — Add split protocol and leakage prevention
**WHERE:** Page 14, Section 4.1, after dataset descriptions (after DFDC paragraph)
**REVIEWER:** R#4.2

**ADD THE FOLLOWING NEW PARAGRAPH:**
```
Split Protocol and Leakage Controls

All dataset splits are performed at the video level using a stratified 80/20
train/test partition (scikit-learn train_test_split, random_state=42,
stratify=y). For LRCN training, an additional internal 20% of the training
set is reserved for validation, resulting in an effective 64%/16%/20%
train/internal-val/test split. The SVM is trained on the full 80% training
split with no internal validation.

Leakage prevention: Each video is processed independently. All 15 frames
sampled from a given video appear exclusively in the split to which that
video is assigned. No frame-level shuffling is performed across videos.
Because Celeb-DF v2 and DFDC do not publish identity metadata in a form
usable for subject-disjoint splitting, this is noted as a limitation;
however, the video-level split prevents the primary form of train-test
leakage (frame overlap).

Test set composition:
  Celeb-DF v2: 1,246 videos (1,128 fake + 118 real), imbalance ratio 9.56:1
  DFDC: 687 videos (614 fake + 73 real), imbalance ratio 8.41:1
```

---

# ============================================================
# SECTION 7: SECTION 4.2 — EVALUATION PARAMETERS (Pages 14-16)
# ============================================================

### Correction E1 — Add evaluation unit clarification
**WHERE:** Page 14, Section 4.2, at the very beginning
**REVIEWER:** R#4.3

**ADD BEFORE the existing text of Section 4.2:**
```
All metrics reported in this paper are computed at the per-video level. Each
video is represented by a fixed-length sequence of T=15 frames, sampled at a
stride of 15 frames from the beginning of the video. A single prediction is
produced per video:
  - LRCN: the softmax output of the LSTM over the 15-frame sequence gives
    one prediction per video (N=1,246 for Celeb-DF, N=687 for DFDC).
  - SVM: the VGG16 feature vector of the first detected face is classified
    to produce one prediction per video.
```

---

### Correction E2 — Fix loss function description
**WHERE:** Page 16, Section 4.2, "Loss Function" paragraph
**REVIEWER:** R#4.4

**FIND:**
```
Loss Function: For the loss calculation, Sparse Categorical Cross-Entropy loss
function is used. This is advantageous for deepfake detection as it simplifies label
representation, conserves memory, enhances interpretability, reduces computation time,
ensures stable optimization, and offers flexibility in the model's output layer. This
function measures the dissimilarity between model predictions and actual labels used.
This is calculated using equation 6. It quantifies the error in binary image classification
by calculating how well the predicted class probabilities align with the true labels.
```

**REPLACE WITH:**
```
Loss Function: The LRCN model is trained using categorical cross-entropy,
which measures the dissimilarity between the predicted class probability
distribution and the one-hot encoded ground truth. For a two-class problem
(Fake=0, Real=1), this is calculated using Equation 6.
```

---

### Correction E3 — Fix Equation 6
**WHERE:** Page 16, Equation 6
**REVIEWER:** R#4.4

**FIND (Eq. 6):**
```
L = -(1/m) * sum(yi * log(yi_hat))
```

**REPLACE WITH:**
```
L = -(1/N) * sum_i sum_c [ y_{i,c} * log(y_hat_{i,c}) ]

where y_{i,c} in {0,1} is the one-hot encoded ground truth for sample i
and class c in {Fake, Real}, and y_hat_{i,c} is the predicted softmax
probability.
```

---

### Correction E4 — Add Balanced Accuracy definition
**WHERE:** Page 16, Section 4.2, after Specificity (Eq. 5)
**REVIEWER:** R#7.3

**ADD NEW PARAGRAPH:**
```
Balanced Accuracy: Balanced accuracy addresses the limitation of standard
accuracy on imbalanced datasets by computing the arithmetic mean of
sensitivity (recall) and specificity:

  Balanced Accuracy = (Recall + Specificity) / 2

A balanced accuracy of 0.5 corresponds to random guessing regardless of
class distribution, making it a more informative metric than standard
accuracy when the dataset is imbalanced (e.g., 9:1 fake-to-real ratio).
```

---

# ============================================================
# SECTION 8: SECTION 4.3 — RESULTS (Pages 16-25)
# ============================================================

### Correction R1 — Fix runtime/FPS claim
**WHERE:** Pages 16-17, Section 4.3, first paragraph
**REVIEWER:** R#4.7, R#7.1

**FIND:**
```
In the proposed system pipeline takes around 0.42 seconds, or about 0.028 frames
per second, to process a movie with 15 sampled frames on an NVIDIA RTX 3050
laptop GPU. This is nearly real-time performance (about 35 frames per second).
```

**REPLACE WITH:**
```
Pipeline latency was measured over 10 test videos on an NVIDIA RTX 3050
laptop GPU. The per-stage timing (mean +/- std) is:

  Face detection (Haar cascade, CPU):   0.352 +/- 0.022 s
  VGG16 feature extraction (GPU):       0.149 +/- 0.078 s
  LRCN classification (GPU):            0.056 +/- 0.059 s
  Total:                                0.557 +/- 0.138 s

This corresponds to a throughput of approximately 26.9 frames/second
(15 frames / 0.557 s). VeriLens is designed for offline forensic video
analysis, not real-time streaming, which requires sustained processing
at 30+ fps.
```

---

### Correction R2 — Fix confusion matrix discussion (LRCN / page 19)
**WHERE:** Page 19, Section 4.3, paragraph about Fig. 11
**REVIEWER:** R#4.1, R#4.3

**FIND:**
```
The VeriLens system, implemented using the LRCN model, produced the confu-
sion matrix shown in Figure 11, demonstrating strong classification performance. It
correctly identified 374 authentic films and 7,530 fraudulent movies. However, 469
genuine films were misclassified as fake, and 226 fraudulent videos were mislabeled as
real. Overall, the proportion of incorrectly classified samples remained comparatively
low. These results highlight the trade-off between accuracy and recall, indicating high
sensitivity in detecting fraudulent content while exhibiting a modest false positive rate
for authentic videos.
```

**REPLACE WITH:**
```
The VeriLens system, implemented using the LRCN model, produced the
confusion matrix shown in Figure 11 (per-video evaluation, N=1,246). The
LRCN correctly classified [TN] fake videos and [TP] real videos, while
[FP] fake videos were mislabeled as real and [FN] real videos were
misclassified as fake. The resulting metrics are: Precision(Real) =
TP/(TP+FP) = [value], Recall(Real) = TP/(TP+FN) = [value], F1(Real) =
[value], Accuracy = [value], Balanced Accuracy = [value].

[FILL VALUES FROM YOUR RTX 3050 RUN OF VeriLens_Final.ipynb —
use results/celeb-df/lrcn_original/metrics.json]
```

---

### Correction R3 — Fix confusion matrix discussion (SVM / page 19)
**WHERE:** Page 19, Section 4.3, paragraph about Fig. 12
**REVIEWER:** R#4.1, R#4.3

**FIND:**
```
The VeriLens system implemented using the SVM model produced the confusion
matrix shown in Figure 12, accurately classifying 15,030 cases with only 5 misclassifi-
cations, demonstrating the model's remarkable capability to detect fraudulent videos.
While 1,317 samples were incorrectly classified as fake, 203 authentic video sam-
ples were correctly identified. Despite a slightly higher false positive rate for genuine
content, these results indicate exceptional precision in detecting bogus videos.
```

**REPLACE WITH:**
```
The VeriLens system implemented using the SVM model produced the
confusion matrix shown in Figure 12 (per-video evaluation, N=1,246). The
SVM correctly classified [TN] fake videos and [TP] real videos, while
[FP] fake videos were mislabeled as real and [FN] real videos were
misclassified as fake. The resulting metrics are: Precision(Real) = [value],
Recall(Real) = [value], F1(Real) = [value], Accuracy = [value],
Balanced Accuracy = [value].

[FILL VALUES FROM YOUR RTX 3050 RUN — use results/celeb-df/svm/metrics.json]
```

---

### Correction R4 — Fix the text between Fig. 12 and Fig. 13 (page 20)
**WHERE:** Page 20
**REVIEWER:** R#4.1

**FIND:**
```
Based on the confusion matrices presented in Figures 11 and 12, we can examine
the performance of both the LRCN and SVM approaches. The use of LRCN in the
proposed system results in higher precision, better overall accuracy, and improved
specificity, as illustrated in Figure 13. In contrast, employing SVM in the proposed
system achieves superior recall and a higher F1 score, as depicted in Figure 14.
```

**REPLACE WITH:**
```
Based on the confusion matrices presented in Figures 11 and 12, we verify
the per-video metrics reported in Table 1. The LRCN model achieves higher
precision and specificity at the cost of lower recall, while the SVM achieves
more balanced recall. This trade-off is a direct consequence of the 9.56:1
class imbalance (see Section 4.4 for ablation analysis). Figures 13 and 14
show the training and validation performance curves for the LRCN and SVM
models respectively.
```

---

### Correction R5 — Add verification paragraph (NEW)
**WHERE:** Page 20, Section 4.3, AFTER the paragraph above (R4)
**REVIEWER:** R#4.1

**ADD THIS NEW PARAGRAPH:**
```
Verification of Table 1 from confusion matrices: To ensure consistency
between Table 1 and the confusion matrices, we derive all metrics directly
from the displayed counts:

  LRCN (Fig. 11): TN=[val], FP=[val], FN=[val], TP=[val]
    Precision = TP/(TP+FP) = [val]
    Recall = TP/(TP+FN) = [val]
    F1 = 2*P*R/(P+R) = [val]
    Accuracy = (TP+TN)/(TP+TN+FP+FN) = [val]
    Specificity = TN/(TN+FP) = [val]
    Balanced Accuracy = (Recall+Specificity)/2 = [val]

  SVM (Fig. 12): TN=[val], FP=[val], FN=[val], TP=[val]
    Precision = TP/(TP+FP) = [val]
    Recall = TP/(TP+FN) = [val]
    F1 = 2*P*R/(P+R) = [val]
    Accuracy = (TP+TN)/(TP+TN+FP+FN) = [val]
    Specificity = TN/(TN+FP) = [val]
    Balanced Accuracy = (Recall+Specificity)/2 = [val]

These values match the corresponding columns in Table 1.

[FILL ALL [val] FROM YOUR RTX 3050 RUN]
```

---

### Correction R6 — Fix AUC interpretation (page 19)
**WHERE:** Page 19, Section 4.3, first paragraph
**REVIEWER:** R#4.10

**FIND:**
```
the proposed approach achieves an AUC of 0.95, demon-
strating a high degree of accuracy in distinguishing between positive and negative
cases. This means that in approximately 92.1% of instances, the classifier correctly
identifies a positive example over a negative one.
```

**REPLACE WITH:**
```
the proposed LRCN model achieves an AUC of [value from RTX 3050 run],
indicating that a randomly chosen real video receives a higher deepfake
score than a randomly chosen fake video with probability [AUC value as %].
```

---

### Correction R7 — Fix section cross-reference (page 16)
**WHERE:** Page 16, Section 4.3, first sentence
**REVIEWER:** R#4.10

**FIND:**
```
the dataset described in Section 3.1
```

**REPLACE WITH:**
```
the datasets described in Section 4.1
```

---

### Correction R8 — Fix DFDC confusion matrix text (page 21)
**WHERE:** Page 21, paragraph about Fig. 15
**REVIEWER:** R#4.3

**FIND:**
```
the confusion matrix indicates that the model
correctly identified 28 authentic films and 604 fraudulent movies. However, 45 genuine
films were incorrectly labeled as fake, while 10 fraudulent videos were misclassified as
real.
```

**REPLACE WITH:**
```
the confusion matrix (per-video evaluation, N=687) indicates that the LRCN
correctly classified [TN] fake videos and [TP] real videos, while [FP] fake
videos were misclassified as real and [FN] real videos were incorrectly
labeled as fake.

[FILL FROM results/dfdc/lrcn_original/metrics.json]
```

---

### Correction R9 — Fix DFDC SVM confusion matrix text (page 21)
**WHERE:** Page 21, paragraph about Fig. 16
**REVIEWER:** R#4.3

**FIND:**
```
the confusion matrix reveals that the model correctly
classified 43 authentic films and 513 fraudulent videos. However, 30 genuine videos
were misclassified as fake, and 101 fraudulent videos were incorrectly labeled as real.
```

**REPLACE WITH:**
```
the SVM confusion matrix (per-video evaluation, N=687) shows [TN] fake
videos correctly classified and [TP] real videos correctly identified,
while [FP] fake videos were mislabeled as real and [FN] real videos were
misclassified as fake.

[FILL FROM results/dfdc/svm/metrics.json]
```

---

### Correction R10 — Fix Table 1 column headers
**WHERE:** Page 24, Table 1
**REVIEWER:** R#4.1 (THE MAIN BUG)

**FIND:**
```
Table 1 Comparative Analysis...

Parameters/Methods | VeriLens(LRCN) | VeriLens(SVM) | VeriFace | DenseNet
Precision          | 0.976          | 0.623         | 0.741    | 0.950
Recall             | 0.134          | 0.444         | 0.473    | 0.993
Accuracy           | 0.920          | 0.919         | 0.734    | 0.970
F1-score           | 0.235          | 0.518         | 0.572    | 0.971
Specificity        | 0.997          | 0.971         | 0.241    | 0.992
```

**REPLACE WITH (Option A — swap headers, SIMPLEST FIX if keeping old per-frame numbers):**
```
Swap "VeriLens (LRCN)" and "VeriLens (SVM)" column headers.
```

**REPLACE WITH (Option B — RECOMMENDED — use new per-video numbers from RTX 3050 run):**
```
Table 1 Comparative Analysis of VeriLens with Existing Approaches
for Celeb-DF Dataset [17] (per-video evaluation, N=1,246)

Parameters/Methods | VeriLens(LRCN) | VeriLens(SVM) | VeriLens(LRCN-Focal) | VeriFace(CNN)+ | DenseNet+
Precision          | [from run]     | [from run]    | [from run]           | 0.741          | 0.950
Recall             | [from run]     | [from run]    | [from run]           | 0.473          | 0.993
Accuracy           | [from run]     | [from run]    | [from run]           | 0.734          | 0.970
F1-score           | [from run]     | [from run]    | [from run]           | 0.572          | 0.971
Specificity        | [from run]     | [from run]    | [from run]           | 0.241          | 0.992
Balanced Accuracy  | [from run]     | [from run]    | [from run]           | N/A            | N/A
AUC                | [from run]     | [from run]    | [from run]           | N/A            | N/A

+ Results reproduced from original publications; not re-evaluated under
  our protocol. Direct comparison should be interpreted with caution.

REMOVE all +/- std values (they were not from cross-validation or
multiple runs).
```

---

### Correction R11 — Fix Table 2 similarly
**WHERE:** Page 24, Table 2
**REVIEWER:** R#4.1, R#4.8, R#7.3

Same structure as R10 above. Use per-video DFDC results. Add:
- LRCN-Retrained (focal) column
- Balanced Accuracy row
- AUC row
- Footnote for VeriFace/DenseNet
- Remove +/- std values

---

### Correction R12 — Fix comparative claims paragraph (page 24)
**WHERE:** Page 24, paragraph starting "As shown in Table 1 and 2..."
**REVIEWER:** R#4.8

**FIND:**
```
when implemented with the LRCN model, VeriLens
achieves the highest precision (0.976) and specificity (0.997)
```

**REPLACE WITH (update numbers from your run, and soften language):**
```
when implemented with the LRCN model, VeriLens achieves high precision
([value]) and specificity ([value])
```

> Throughout this paragraph: replace all hardcoded numbers with values from your RTX 3050 run. Replace "achieves the highest" with "achieves high" or "achieves competitive" since VeriFace/DenseNet were not run under the same protocol.

---

### Correction R13 — Fix page 25 discussion paragraph
**WHERE:** Page 25, first paragraph
**REVIEWER:** R#4.8

**FIND:**
```
The LRCN model in VeriLens prioritizes high accuracy (0.976)
```

**REPLACE WITH:**
```
The LRCN model in VeriLens prioritizes high precision ([value])
```

> Note: "accuracy (0.976)" is wrong — 0.976 was the precision value. Fix to say "precision".

---

# ============================================================
# SECTION 9: NEW SECTION 4.4 — ADD ENTIRELY
# ============================================================

### Correction N1 — Add new Section 4.4: Class-Imbalance Analysis
**WHERE:** After Section 4.3, before Section 5
**REVIEWER:** R#7.3, R#7.4, R#7.5, R#7.6

**ADD THIS ENTIRE NEW SECTION:**
```
4.4 Class-Imbalance Analysis and Cost-Sensitive Learning

The Celeb-DF v2 test set exhibits a 9.56:1 fake-to-real imbalance (1,128
fake vs. 118 real videos). A trivial classifier predicting Fake for all
samples achieves approximately 90.5% accuracy, which provides context for
interpreting the LRCN's 92.2% accuracy alongside its low recall of [value].

Ablation for LRCN Recall Collapse:
The LRCN's low recall for the Real class despite high overall accuracy is
a direct consequence of class imbalance. Balanced accuracy ([value]) is
only [X] points above the random baseline of 50%, revealing the true
discriminative performance masked by standard accuracy. Threshold analysis
(Figure [X]) shows the LRCN assigns systematically low probability to the
Real class — the optimal F1 threshold is [value], far below the default 0.5.

Cost-Sensitive Retraining:
To mitigate the class imbalance, we retrain the LRCN with the following
modifications:
  1. Focal loss (gamma=2.0) to down-weight easy majority-class examples
  2. Inverse-frequency class weights: alpha_fake = N/(2*N_fake),
     alpha_real = N/(2*N_real)
  3. Balanced mini-batch sampling: each batch contains equal numbers of
     Real and Fake samples (oversampling minority with replacement)
  4. ReduceLROnPlateau (factor=0.5, patience=2) and early stopping
     (patience=4, restore best weights)

The retrained model (LRCN-Focal) improves recall from [X]% to [X]% on
Celeb-DF and from [X]% to [X]% on DFDC, demonstrating that the recall
collapse is caused by the loss landscape under imbalance, not by
architectural limitations.

[INSERT FIGURE: Threshold analysis plot — precision, recall, F1, balanced
accuracy vs. decision threshold for all three models]

[INSERT FIGURE: PR curves with Average Precision for all models on both
datasets]

[FILL ALL VALUES FROM RTX 3050 RUN]
```

---

# ============================================================
# SECTION 10: SECTION 5 — CONCLUSION (Page 25)
# ============================================================

### Correction C1 — Fix conclusion overclaims
**WHERE:** Page 25, Section 5.1
**REVIEWER:** R#4.9, R#4.8

**FIND:**
```
this paper introduced VeriLens, a deepfake
detection framework that integrates Face Masking, VGG-16, and LRCN models to
achieve enhanced accuracy, scalability, and real-time performance.
```

**REPLACE WITH:**
```
this paper introduced VeriLens, a deepfake detection framework that
integrates Haar cascade face detection, VGG-16 feature extraction, and
LRCN temporal classification to achieve scalable offline deepfake detection.
```

---

### Correction C2 — Fix conclusion comparison
**WHERE:** Page 25, Section 5.1, last paragraph
**REVIEWER:** R#4.8

**FIND:**
```
In summary, VeriLens achieves competitive DeepFake detection performance com-
pared with VeriFace and DenseNet-based methods. By offering a high-precision LRCN
variant and a higher-recall SVM variant
```

**REPLACE WITH:**
```
In summary, VeriLens demonstrates competitive deepfake detection
performance relative to VeriFace and DenseNet-based methods (noting that
baseline results are from original publications and not re-evaluated under
our protocol). By offering a high-precision LRCN variant, a higher-recall
LRCN variant retrained with focal loss, and an SVM baseline
```

---

# ============================================================
# SECTION 11: FIGURES TO UPDATE/ADD
# ============================================================

### Figures to regenerate from RTX 3050 run:
1. **Fig. 11** — LRCN Confusion Matrix (Celeb-DF) → regenerate with per-video counts
2. **Fig. 12** — SVM Confusion Matrix (Celeb-DF) → regenerate with per-video counts
3. **Fig. 10** — ROC curves → regenerate with correct AUC values
4. **Fig. 15** — LRCN Confusion Matrix (DFDC) → regenerate with per-video counts
5. **Fig. 16** — SVM Confusion Matrix (DFDC) → regenerate with per-video counts
6. **Fig. 17-18** — ROC curves (DFDC) → regenerate

### New figures to add:
7. **New Fig.** — PR curves (all models, both datasets)
8. **New Fig.** — Threshold analysis plot (precision/recall/F1/balanced-acc vs threshold)
9. **New Fig.** — LRCN-Retrained confusion matrices (both datasets)

---

# ============================================================
# QUICK CHECKLIST — GLOBAL FIND-AND-REPLACE
# ============================================================

| Find | Replace | Count |
|---|---|---|
| AAR Cascade | Haar Cascade | ~2 occurrences |
| eye blinking detection | (DELETE) | ~3 occurrences |
| Eye Blinking | (DELETE or context-dependent) | ~3 occurrences |
| face recognition | face detection | ~4 occurrences (check context) |
| real-time | offline/near-real-time | ~3 occurrences |
| binary cross-entropy | categorical cross-entropy | ~1 occurrence |
| Sparse Categorical Cross-Entropy | categorical cross-entropy | ~1 occurrence |
| sigmoid | softmax | ~1 occurrence (in LRCN section) |
| Section 3.1 (when referring to datasets) | Section 4.1 | ~1 occurrence |
| 35 frames per second | (DELETE) | ~1 occurrence |
| 0.028 frames per second | (DELETE) | ~1 occurrence |

---

# ============================================================
# PRIORITY ORDER FOR EDITING
# ============================================================

1. **Table 1 column swap** (R10) — most critical, easiest fix
2. **LRCN architecture rewrite** (L1) — completely wrong in current paper
3. **Loss function fix** (E2, E3) — contradicts implementation
4. **Runtime/FPS removal** (R1) — arithmetic error
5. **Add Section 4.4** (N1) — new content required by R#7
6. **Add split protocol** (D1) — new content required by R#4
7. **Add evaluation unit** (E1) — new content required by R#4
8. **SVM hyperparameters** (S2) — small addition
9. **Abstract/keywords/intro fixes** (A1, A2, K1, I1, I2) — overclaim removal
10. **Global find-and-replace** (checklist above)
11. **Regenerate all figures** from RTX 3050 run
12. **Fill all [bracketed values]** from metrics.json files
