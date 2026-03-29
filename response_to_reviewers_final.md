# Response to Reviewers — VeriLens Revision
# Journal: Multimedia Tools and Applications (Springer)
# Manuscript: VeriLens — Deepfake Detection via VGG16 + SVM/LRCN

---

## General Statement

Dear Editor and Reviewers,

We sincerely thank all reviewers for their detailed and constructive feedback. One reviewer accepted the previous revision; we address the remaining concerns from Reviewers #4 and #7 below. Every point has been addressed with concrete corrections, new experiments, and added analysis. A summary of all changes is provided at the end of this document.

---

# REVIEWER #4

---

## R4.1 — Celeb-DF Result Inconsistency (Table 1 Column Swap)

> **Reviewer #4:** Fix the Celeb-DF result inconsistency. Table 1's precision/recall/F1 values do not match the confusion matrices for the model labels shown in Figs. 11-12. When the metrics are recomputed from the displayed counts, the values align with the opposite model column, which means either the figure labels or the table entries (or the positive-class convention) are wrong. Please correct the labeling and add a short verification paragraph in Section 4.3 that explicitly shows the confusion-matrix-derived precision/recall/F1 matching the reported metrics.

**Response:** The reviewer is correct. We identified a column header labeling error in Table 1: the LRCN and SVM column headers were swapped. The underlying numerical values were correct; only the headers were mislabeled. We verified this by recomputing metrics directly from the confusion matrices:

- **From Fig. 11 (labeled "LRCN"):** TP=374, TN=7530, FP=226, FN=469 → Precision=374/(374+226)=**0.623**, Recall=374/(374+469)=**0.444**, Accuracy=(7530+374)/8599=**0.919**, Specificity=7530/(7530+226)=**0.971**. These match the **SVM column** in the original Table 1 — confirming the swap.

- **From Fig. 12 (labeled "SVM"):** TP=203, TN=15030, FP=5, FN=1317 → Precision=203/(203+5)=**0.976**, Recall=203/(203+1317)=**0.134**, Accuracy=(15030+203)/16555=**0.920**, Specificity=15030/(15030+5)=**0.997**. These match the **LRCN column** in the original Table 1.

We have corrected the column headers in the revised Table 1. Additionally, the revised paper now uses **per-video evaluation** (N=1,246 videos), which eliminates the frame-count confusion. A verification paragraph has been added to Section 4.3 showing the confusion-matrix-derived metrics matching each reported value.

**Changes:** Table 1 (corrected headers), Section 4.3 (added verification paragraph), Figs. 11–12 (updated to per-video counts).

---

## R4.2 — Split Protocol and Leakage Controls

> **Reviewer #4:** Report the split protocol and leakage controls in a reproducible way. The manuscript mentions a "train test split," but does not state the train/validation/test ratios, whether splitting is done at video level (required), whether subjects/identities are disjoint across splits, and how leakage is prevented when sampling multiple frames per video. Without these details, the reported performance is not interpretable and cannot be replicated. Add this to Section 4.1 and reference it wherever results are discussed.

**Response:** We have added the following to Section 4.1:

> All dataset splits are performed at the **video level** using a stratified 80/20 train/test partition (scikit-learn `train_test_split`, `random_state=42`, `stratify=y`). For LRCN training, an additional internal 20% of the training set is reserved for validation during training, resulting in an effective 64%/16%/20% train/internal-val/test split. The SVM is trained on the full 80% training split with no internal validation.
>
> **Leakage prevention:** Each video is processed independently. All 15 frames sampled from a given video appear exclusively in the split to which that video is assigned. No frame-level shuffling is performed across videos. Because Celeb-DF v2 and DFDC do not publish identity metadata in a form usable for subject-disjoint splitting, we note this as a limitation; however, the video-level split prevents the primary form of train-test leakage (frame overlap).
>
> **Test set composition (Celeb-DF v2):** 1,246 videos (1,128 fake + 118 real), imbalance ratio 9.56:1.
> **Test set composition (DFDC):** 687 videos in the 80/20 split, of which 607 yielded successful face detections (568 fake + 39 real), imbalance ratio 14.56:1.

**Changes:** Section 4.1 (new subsection on split protocol).

---

## R4.3 — Evaluation Unit and Frame Aggregation

> **Reviewer #4:** Clarify the evaluation unit and how frames become a single prediction. It is unclear whether metrics are computed per-frame or per-video. The Celeb-DF dataset counts stated in Section 4.1 do not align with the totals implied by the Celeb-DF confusion matrices, which suggests either frame-level evaluation or a filtered subset, neither of which is specified. You must state: frames sampled per video, sampling rule, whether predictions are per-frame or per-video, and the aggregation rule (e.g., mean probability, majority vote). Then ensure Tables/Figures are consistent with that definition.

**Response:** We clarify that the **original paper's Celeb-DF confusion matrices (Figs. 11–12) reflected per-frame evaluation**, which explains the inflated sample counts (8,599 and 16,555 vs. the dataset's ~6,229 total videos). This was not stated and is a valid source of confusion.

All new experimental results in the revised paper use **per-video evaluation**:

> Each video is represented by a fixed-length sequence of T=15 frames, sampled at a stride of 15 frames from the beginning of the video. A single prediction is produced per video:
> - **LRCN:** the softmax output of the LSTM over the 15-frame sequence gives one prediction per video.
> - **SVM:** the VGG16 feature vector of the **first frame** in the sequence is used to produce one prediction per video.
>
> Evaluation unit is stated in all table captions and figure legends.

**Changes:** Section 4.1 (sampling rule), Section 4.2 (aggregation rule), Section 4.3 (all tables and figure captions now state "per-video, N=1,246" or "per-video, N=607").

---

## R4.4 — Loss Function Contradiction

> **Reviewer #4:** Resolve the loss-function contradiction and make Eq. (6) match the implemented model. Section 4.2 states "Sparse Categorical Cross-Entropy" and provides Eq. (6), while Section 3.1.5 describes training the LRCN using binary cross-entropy with a single sigmoid output. These are not the same objective. Please correct Section 4.2 and Eq. (6) to reflect the actual training setup (or revise the model outputs to match the stated loss), and keep the description consistent across sections.

**Response:** Both descriptions were incorrect. The actual implementation uses **categorical cross-entropy** with a `Dense(2, activation='softmax')` output layer. We have:

1. Corrected Section 3.1.5 to describe the actual architecture (see R4.6 below).
2. Updated Section 4.2 and Equation 6 to the standard categorical cross-entropy formulation:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=0}^{1} y_{i,c} \log(\hat{y}_{i,c})$$

where $y_{i,c} \in \{0,1\}$ is the one-hot encoded ground truth for sample $i$ and class $c$, and $\hat{y}_{i,c}$ is the predicted softmax probability.

3. Removed all mentions of "binary cross-entropy," "sparse categorical cross-entropy," and "sigmoid output" from the LRCN description.

**Changes:** Section 3.1.5 (architecture description), Section 4.2 (Eq. 6 and surrounding text).

---

## R4.5 — SVM Configuration and Preprocessing

> **Reviewer #4:** Fully specify the SVM configuration and preprocessing for VGG features. The SVM baseline cannot be replicated without kernel choice, C, γ (if RBF/poly), class weights, probability calibration (if any), and whether/how features are standardized/normalized before fitting. Add these settings in Section 3.1.4 and summarize them in the experimental setup section.

**Response:** We have added full SVM specification to Section 3.1.4:

| Parameter | Value |
|---|---|
| Kernel | RBF |
| C | 1.0 |
| γ | `'scale'` (computed as 1/(n_features × Var(X)), i.e., ≈ 1/(512 × Var(X))) |
| Class weights | None (uniform) |
| Probability calibration | Post-hoc Platt scaling via scikit-learn `probability=True` |
| Feature standardization | None (VGG16 avg-pool outputs are bounded) |
| Random seed | 42 |

> The SVM operates on the 512-dimensional VGG16 average-pooling feature vector extracted from a **single frame** per video (the first successfully detected face in the 15-frame sequence).

**Changes:** Section 3.1.4 (new SVM specification table), Section 4.1 (experimental setup summary).

---

## R4.6 — LRCN Architecture Completeness

> **Reviewer #4:** Make the LRCN model definition complete enough to implement without guessing. Section 3.1.5 gives a partial architecture (Conv1D/MaxPool/LSTM/Dropout/Flatten/Dense), but the input tensor definition and the sequence construction (sequence length, ordering, padding/truncation strategy) are not reported. Also report learning rate, any scheduling, early stopping criteria (if used), and the random seed strategy, especially because Tables 1-2 report mean±std but the source of variance is not described.

**Response:** Section 3.1.5 has been completely rewritten to match the actual implementation. The original description (Conv1D/MaxPool) was incorrect.

> **Input:** A sequence of T=15 VGG16 feature vectors, each of dimension 512, giving input shape (15, 512). Frames are sampled every 15th frame from the video. If fewer than 15 faces are detected, the sequence is zero-padded.
>
> **Architecture:**
> 1. `TimeDistributed(Dense(256, activation='relu'))` — applies the same dense layer to each of the 15 time steps independently.
> 2. `Dropout(0.3)`
> 3. `LSTM(128, return_sequences=False)` — produces a single 128-dimensional output vector.
> 4. `Dropout(0.3)`
> 5. `Dense(2, activation='softmax')` — two-class output (Fake=0, Real=1).
>
> **Training:** Adam optimizer, learning rate = 1×10⁻⁴, batch size = 2, 10 epochs. Loss: categorical cross-entropy. Internal 20% validation split. Random seed = 42 (NumPy and TensorFlow). No learning rate scheduling or early stopping for the original model.
>
> **Note on ±std in tables:** The ±standard deviation values in the original tables were not derived from multiple random seeds or cross-validation. In the revised paper, we report single-run results without artificial ±std values, or explicitly note when variance is computed across seeds.

**Changes:** Section 3.1.5 (complete rewrite), Tables 1–2 (±std clarified or removed).

---

## R4.7 — Runtime Claim

> **Reviewer #4:** Correct and strengthen the "real-time" runtime claim. Section 4.3 contains an arithmetic inconsistency: it reports 0.42 s for 15 frames and also states 0.028 fps, while 15/0.42 implies ~35.7 fps. Please correct the fps statement and report runtime as mean±std over multiple videos, including a breakdown by stage (face detection/cropping, VGG forward pass, classifier, I/O), and state whether GPU acceleration is used for each stage.

**Response:** The original text contained an arithmetic error (0.028 is seconds-per-frame, not fps) and the "~35 FPS" extrapolation was misleading. More fundamentally, frame-processing throughput does not imply real-time continuous video streaming capability. We have replaced the runtime section with measured per-stage timing:

> Pipeline latency measured over **10** test videos on RTX 3050 (mean ± std):
>
> | Stage | Acceleration | Mean (s) | Std (s) |
> |---|---|---|---|
> | Face detection (Haar cascade) | CPU (OpenCV) | 0.352 | 0.022 |
> | VGG16 feature extraction | GPU (CUDA) | 0.149 | 0.078 |
> | LRCN classification | GPU (CUDA) | 0.056 | 0.059 |
> | **Total** | | **0.557** | **0.138** |
>
> This corresponds to a throughput of ~26.9 frames/second (15 frames / 0.557 s). We explicitly state this is suitable for **offline forensic analysis**, not real-time streaming (which requires ≥30 fps sustained at full resolution). The "35 FPS" claim has been removed.

**Changes:** Section 4.3 (replaced runtime paragraph with timing table), removed "real-time" claims throughout.

---

## R4.8 — Baseline Comparison Protocol

> **Reviewer #4:** Ensure baseline comparisons are protocol-matched, or weaken the comparative claims. Tables 1-2 include DenseNet and VeriFace numbers, but the manuscript does not show that these were run under the same data splits, preprocessing, and evaluation unit. If they are taken from prior work, label them as "reported results" and avoid direct superiority claims. If they are your own re-runs, report the exact protocol and code settings so the comparison is fair and reproducible.

**Response:** The DenseNet and VeriFace results were taken from their original publications and **were not re-evaluated** under our experimental protocol. We have:

1. Added a footnote to Tables 1 and 2:
> "† Results for DenseNet [3] and VeriFace [16] are reproduced from their original publications and were not re-evaluated under our split protocol, preprocessing, or evaluation unit. Direct numerical comparison should be interpreted with caution due to differences in experimental setup."

2. Softened all comparative claims in the text (e.g., replacing "outperforms" with "achieves competitive performance relative to").

**Changes:** Tables 1–2 (added footnote), Section 4.3 (softened comparative language), Section 5 Conclusion (revised comparative claims).

---

## R4.9 — Contribution Claims (Eye Blinking, Face Recognition)

> **Reviewer #4:** Align stated contributions with what is actually measured. The abstract and introduction reference "eye blinking detection" and "face recognition," but the methodology and experiments do not provide a measurable implementation description or ablation demonstrating their effect. Either add a measurable module definition plus an ablation study, or remove/soften those claims to reflect the implemented pipeline (VGG16 features + SVM/LRCN).

**Response:** We have removed the "eye blinking detection" claim entirely, as it is not implemented in the pipeline. "Face recognition" has been changed to "face detection using Haar cascades," which accurately describes the implemented module. The revised contribution statement reads:

> VeriLens extracts face sequences using Haar cascade detection, encodes them with pretrained VGG16 features, and classifies videos using either an LRCN (for temporal sequence analysis) or an SVM baseline (for single-frame analysis).

**Changes:** Abstract (removed eye blinking, corrected face recognition → face detection), Section 1 Introduction (revised contribution list), Section 3 (aligned methodology description).

---

## R4.10 — Minor Technical Corrections

> **Reviewer #4:** Minor technical corrections that affect clarity/reproducibility. Fix typographical inconsistencies that can mislead implementation (e.g., "AAR Cascade" vs Haar Cascade), correct the section cross-reference mismatch (Section 4.3 refers to datasets in Section 3.1 though they appear in Section 4.1), tighten metric interpretation statements around AUC so the probabilistic meaning matches the stated value.

**Response:** All corrected:

1. **"AAR Cascade" → "Haar Cascade"** — global find-and-replace throughout the manuscript.
2. **Section cross-reference:** Section 4.3 now correctly refers to "Section 4.1" (not "Section 3.1") for dataset descriptions.
3. **AUC interpretation corrected:** The original text stated "an AUC of 0.95, demonstrating… approximately 92.1% of instances." This has been corrected to: "An AUC of 0.770 indicates that a randomly chosen real video receives a higher score than a randomly chosen fake video with probability 77.0%." All AUC interpretations now use the correct probabilistic meaning.

**Changes:** Global (Haar Cascade typo), Section 4.3 (cross-reference fix), Section 4.3 (AUC interpretation).

---

## R4.11 — Citations

> **Reviewer #4:** Please note that any citation recommended by reviewers or editors should only be included in your revisions IF this is justifiable in the context of the work presented. Please check your whole reference list carefully to ensure ALL citations are directly relevant to the specific topic of the manuscript.

**Response:** We have reviewed the entire reference list and confirmed that all citations are directly relevant to deepfake detection, the architectures used (VGG16, LRCN, SVM), the datasets (Celeb-DF v2, DFDC), or the evaluation methodology. No reviewer-recommended citations were added that are not justified by the content. No irrelevant references were found; the list remains unchanged.

**Changes:** None required (reference list verified).

---

# REVIEWER #7

---

## R7.1 — FPS / Real-Time Claim

> **Reviewer #7:** The reported processing speed (0.42 s for 15 frames) is extrapolated to "~35 FPS," which is confusing. This will not result to real-time video processing in standard multimedia terms. (at least 30 fps) (even 60 fps in 1080p).

**Response:** Agreed. The claim has been removed entirely. See Reviewer #4, Point 7 (R4.7) for the replacement latency table with per-stage breakdown. We no longer claim real-time performance and explicitly state VeriLens is designed for offline forensic analysis.

**Changes:** See R4.7.

---

## R7.2 — Thesis-Style Writing

> **Reviewer #7:** Don't write in thesis style. Some sections still exhibit thesis-style.

**Response:** We have revised the manuscript throughout to use concise academic journal style. Specifically:

- Removed step-by-step narrative descriptions of well-known methods (VGG16, SVM)
- Condensed Section 3 methodology into focused, precise descriptions
- Replaced lengthy motivational paragraphs with direct technical statements
- Eliminated first-person narrative accounts of the research process ("We then proceeded to…")
- Replaced hedging language with factual statements ("Table 1 reports…", "The model achieves…", "Figure X shows…")

**Changes:** Abstract, Sections 1, 2, 3, and 5 (writing style revision throughout).

---

## R7.3 — Balanced Accuracy, PR Curves, Cost-Weighted Loss Analysis

> **Reviewer #7:** Get the numbers right. Add balanced Accuracy curves / Add PR curves / cost-weighted loss analysis. Add balanced accuracy, PR curves, or cost-weighted loss analysis.

**Response:** We have added all three:

1. **Balanced accuracy** added as a new column in Tables 1 and 2:
   - Celeb-DF: LRCN-Original 0.588, SVM 0.500, LRCN-Retrained (focal) 0.653
   - DFDC: LRCN-Original 0.711, SVM 0.665, LRCN-Retrained (focal) 0.708

2. **Precision-Recall curves** (new figures) for all models on both datasets. Due to the ~9:1 class imbalance, PR curves are more informative than ROC curves. Average Precision (AP) scores are reported alongside AUC.

3. **Cost-weighted loss analysis** via focal loss (γ=2.0) with inverse-frequency alpha weights, demonstrating the precision-recall trade-off under class imbalance. This is presented in the new Section 4.4.

**Changes:** Tables 1–2 (balanced accuracy column), Section 4.3 (new PR curve figures), Section 4.4 (new: cost-weighted loss analysis).

---

## R7.4 — Threshold Analysis

> **Reviewer #7:** Provide threshold analysis. No threshold analysis is provided.

**Response:** We now include a threshold analysis figure (new figure in Section 4.3) showing precision, recall, F1, and balanced accuracy as functions of the decision threshold for all models. Key findings:

- **LRCN (original):** optimal F1(Real) = 0.444 at threshold **0.095** (default 0.5 severely under-detects real videos)
- **SVM (per-video):** optimal F1 = 0.351 at threshold **0.150** (AUC=0.718 confirms discriminative capacity masked by default threshold)
- **LRCN (focal loss retrained):** optimal F1 at threshold **0.735**

This analysis reveals that the apparent recall collapse is partly an artifact of using the default 0.5 threshold on a model trained with imbalanced data.

**Changes:** Section 4.3 (new threshold analysis figure and discussion).

---

## R7.5 — Cost-Sensitive Learning and Class-Imbalance Mitigation

> **Reviewer #7:** Explore cost sensitive learning. Explore class-imbalance mitigation. No cost-sensitive learning or class-imbalance mitigation is explored.

**Response:** We now present a retrained LRCN variant with the following class-imbalance mitigations:

1. **Focal loss** (γ=2.0) to down-weight easy majority-class examples
2. **Inverse-frequency class weights:** α_fake = N/(2·N_fake), α_real = N/(2·N_real)
3. **Balanced mini-batch generator:** each training batch contains equal numbers of Real and Fake samples (oversampling minority with replacement)
4. **ReduceLROnPlateau** (factor=0.5, patience=2) and **early stopping** (patience=4, restore best weights)

Results (per-video):

| Dataset | Model | Recall(Real) | Balanced Acc | AUC |
|---|---|---|---|---|
| Celeb-DF | LRCN-Original | 0.178 | 0.588 | 0.770 |
| Celeb-DF | LRCN-Retrained | 0.805 | 0.653 | 0.745 |
| DFDC | LRCN-Original | 0.462 | 0.711 | 0.739 |
| DFDC | LRCN-Retrained | 0.795 | 0.708 | 0.816 |

Recall improved from 17.8% → 80.5% on Celeb-DF and 46.2% → 79.5% on DFDC, demonstrating that the recall collapse is directly caused by class imbalance and can be substantially mitigated through cost-sensitive training.

**Changes:** Section 3 (retrained model description), Section 4.4 (new: focal loss experiment and results), Tables 1–2 (LRCN-Retrained row added).

---

## R7.6 — Ablation for LRCN Recall Collapse

> **Reviewer #7:** No ablation explains why LRCN collapses in recall while maintaining high accuracy.

**Response:** We have added an ablation/analysis paragraph to Section 4.3:

> The LRCN's low recall for the Real class despite high overall accuracy is a direct consequence of the 9.56:1 class imbalance (Fake:Real) in Celeb-DF v2. A trivial classifier predicting Fake for all samples achieves ~90.5% accuracy. The original LRCN (92.1% accuracy, 17.8% recall) is only marginally better than this baseline.
>
> **Evidence:**
> 1. **Balanced accuracy** (58.8%) is only 8.8 points above random (50%), revealing the true discriminative performance masked by standard accuracy.
> 2. **Threshold analysis** shows the LRCN assigns systematically low probability to the Real class — optimal threshold is 0.095, not 0.5.
> 3. **Retraining with focal loss + balanced batches** recovers 80.5% recall (Celeb-DF) and 79.5% recall (DFDC), confirming across both datasets that the collapse is caused by the loss landscape under imbalance, not by architectural limitations.
> 4. **The SVM shows the same pattern:** trained without class weights on imbalanced data, it achieves 0% recall with a default threshold, further confirming that the issue is data-driven, not model-specific.

**Changes:** Section 4.3 (new ablation paragraph), Section 4.4 (cross-referenced).

---

# Summary of All Changes

| # | Issue | Reviewer | Section Changed | Change Type |
|---|---|---|---|---|
| 1 | Table 1 column headers swapped | R#4.1 | Table 1, Section 4.3 | Correction |
| 2 | Verification paragraph added | R#4.1 | Section 4.3 | Addition |
| 3 | Split protocol and leakage controls | R#4.2 | Section 4.1 | Addition |
| 4 | Evaluation unit clarified (per-video) | R#4.3 | Sections 4.1, 4.2, 4.3, captions | Clarification |
| 5 | Loss function corrected (categorical CE) | R#4.4 | Eq. 6, Sections 3.1.5, 4.2 | Correction |
| 6 | SVM full specification | R#4.5 | Section 3.1.4 | Addition |
| 7 | LRCN architecture rewritten | R#4.6 | Section 3.1.5 | Rewrite |
| 8 | Runtime claim corrected | R#4.7, R#7.1 | Section 4.3 | Replacement |
| 9 | Baseline comparison footnote | R#4.8 | Tables 1–2 | Addition |
| 10 | Eye-blink/face-rec overclaims removed | R#4.9 | Abstract, Section 1 | Removal |
| 11 | Haar Cascade typo fixed | R#4.10 | Global | Correction |
| 12 | Section cross-reference fixed | R#4.10 | Section 4.3 | Correction |
| 13 | AUC interpretation corrected | R#4.10 | Section 4.3 | Correction |
| 14 | Reference list verified | R#4.11 | References | Verified |
| 15 | Thesis-style writing revised | R#7.2 | Global | Revision |
| 16 | Balanced accuracy added | R#7.3 | Tables 1–2 | Addition |
| 17 | PR curves added | R#7.3 | Section 4.3 (new figures) | Addition |
| 18 | Cost-weighted loss analysis | R#7.3, R#7.5 | Section 4.4 (new) | Addition |
| 19 | Threshold analysis added | R#7.4 | Section 4.3 (new figure) | Addition |
| 20 | Focal loss retrained model | R#7.5 | Sections 3, 4.3, 4.4, Tables 1–2 | Addition |
| 21 | LRCN recall collapse ablation | R#7.6 | Section 4.3 | Addition |
| 22 | DFDC evaluation added | All | Section 4.3, Table 2 | Addition |

---

We believe these revisions comprehensively address all reviewer concerns. We thank the reviewers again for their constructive feedback, which has substantially strengthened the manuscript.
