# Mango leaf classification with pretrained features and KNN

## Why we built the experiment

We developed this project for the Semester 1 Micro Credential Course in Artificial Intelligence and Robotics at Amrita Vishwa Vidyapeetham, Coimbatore. We wanted to compare classification from raw pixels with classification from a pretrained image representation, then see whether a simple nearest-neighbour model could make useful decisions from those features.

This report describes the existing implementation and saved evidence. It does not report a new training run. The original revision and artifact checksums are recorded in the [reference manifest](../results/reference/manifest.json).

## Dataset and evaluation

The committed dataset has 4,000 images: 500 each for Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew and Sooty Mould. The sorted folder order matches [class_indices.json](../class_indices.json). The [dataset guide](../DataSet/README.md) records the counts and the source attribution carried over from the original documentation.

Each saved evaluation has 800 test images. Base and hybrid use a stratified 80/20 split with random state 42. Pure KNN uses an unstratified split, with different per-class test support. This limits a direct comparison between the raw-pixel baseline and the other two approaches.

## Implementation

### MobileNetV2 and softmax

The base script resizes RGB images to 224 × 224, divides pixel values by 255 and loads an ImageNet-pretrained MobileNetV2 without its original head. The backbone stays frozen. Global average pooling produces 1,280 features, followed by an eight-class softmax layer.

The head is trained with Adam and sparse categorical cross-entropy for 10 epochs, with training shuffling disabled. There is no separate validation split or data augmentation step. The scripts do not apply MobileNetV2's application-specific preprocessing function; the documented scaling is exactly what the code uses.

The saved H5 model ends at global average pooling and excludes the trained softmax head. Because the backbone is frozen, training that head does not fine-tune the representation used by the hybrid.

### MobileNetV2 features and KNN

The hybrid loads the pooled extractor, computes training and test features, then fits a StandardScaler and a KNN classifier. The scaler is fitted only on training features. KNN uses three neighbours, cosine distance and distance-weighted voting.

This is a feature-based pipeline: the pretrained backbone provides the representation, and KNN assigns the label. It is not knowledge distillation, joint CNN/KNN training or a fine-tuned backbone. The fitted pipeline is saved in `model/final_hybrid_model.pkl` and needs the matching feature extractor for image-based use.

### Raw-pixel KNN

The baseline resizes images to 100 × 100 and flattens the RGB values into 30,000 features. It standardises these values and fits three-neighbour KNN with default distance and voting settings.

Besides the representation, it differs from the hybrid in metric, voting weights and split procedure. The observed performance gap cannot therefore be attributed to feature extraction alone. The script also uses a fixed Windows dataset path and saves no fitted model; see the [script guide](../codes/README.md).

## Recorded findings

The following numbers were transcribed from the existing console-output PDFs, which show execution on 13 February 2026. No predictions were regenerated for this report.

| Model | Accuracy | MCC | Probability MSE | Probability MAE | Test images |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 + softmax | 99.12% | 0.9900 | 0.0023 | 0.0079 | 800 |
| MobileNetV2 features + KNN | 98.25% | 0.9801 | 0.0032 | 0.0058 | 800 |
| Raw-pixel KNN | 68.12% | 0.6380 | 0.0523 | 0.0847 | 800 |

Sources: [base PDF](../results/Results_BaseModel%28CNN%2BMobileV2%29.pdf), [hybrid PDF](../results/Results_HybridModel%28KNN%2BBaseModel%29.pdf), [pure KNN PDF](../results/Results_PureKNN.pdf). [Recorded metrics](../results/reference/recorded-metrics.json) also provides a machine-readable transcription.

The base classifier has the highest accuracy and MCC, and the lowest probability MSE. The hybrid's accuracy is 0.87 percentage points below the base's displayed accuracy, while its probability MAE is lower. Those measures capture different aspects of predictions; lower MAE does not make the hybrid more accurate or establish calibrated probabilities.

The matrices contain 793 correct base predictions, 786 correct hybrid predictions and 545 correct raw-pixel predictions out of 800. These agree with the displayed percentages after rounding. The base count corresponds to 99.125% before display rounding.

### Class-level errors

![Hybrid confusion matrix](../results/ConfusionMatrix%28HybridModel%29.png)

Anthracnose has 93 correct hybrid predictions from 100 images. Four are labelled Healthy, two Gall Midge and one Die Back. Bacterial Canker, Cutting Weevil and Die Back each have 100 correct; the remaining classes have 97–99 correct each.

The raw-pixel KNN is less consistent across classes. Gall Midge has 51 correct from 104 and Powdery Mildew 49 from 91. This is why the class support and confusion matrix matter alongside overall accuracy. The [results guide](../results/README.md) displays all matrices and links the one-vs-rest ROC figures. Their AUC labels are rounded; this report does not infer extra precision or statistical significance from them.

## Interpretation

Both approaches using MobileNetV2 are substantially stronger than raw-pixel KNN in the recorded experiment. The pretrained representation is useful enough for a simple nearest-neighbour classifier to reach 98.25% on the saved split. The softmax classifier still has the better accuracy in this run.

The result supports pretrained features as a useful starting point. It does not establish that KNN is universally better, that this architecture is novel, or that these figures will hold in another dataset or capture setting.

## Preserving and reproducing the work

The original scripts, dataset and nine result artifacts remain unchanged. A [reference ZIP](../results/reference/original-results.zip) preserves all original PNGs and PDFs, with hashes and the source revision in its manifest. This gives future runs a clear historical record to compare against.

Base and hybrid save fixed model filenames, so a rerun in the same checkout replaces generated models. They print metrics and show plots but do not export the existing PNG/PDF reports. The [reproduction guide](reproducing-results.md) uses a separate checkout, dated logs, manual plot capture and copied model artifacts.

Historical trained weights, exact package versions and split file lists were not committed. Seeds are set to 42, but filenames are read in unsorted order, dependencies are open-ended and numerical execution can vary by framework and hardware. The evidence can be preserved; identical future predictions cannot be guaranteed by the current scripts.

## Limits and next experiments

The split is at image level. The repository has no saved evaluation on new farms, plants or capture sessions and no documented duplicate/group-aware split. Similar images across a random split could make evaluation easier than a field test; this possibility has not been audited here.

All images are loaded into memory before splitting. The raw-pixel model uses 30,000 features per image, while the neural scripts keep full 224 × 224 RGB arrays before extracting features. This requires enough memory and is not a streaming deployment pipeline.

A stronger follow-up would save one shared test list, pin the environment, compare several seeds, investigate class errors and evaluate on an external dataset. Changes to preprocessing, backbone training or path handling would belong to a new code revision and a separate evaluation. They are future work, not changes made for this report.

The useful finding is the gap between raw pixels and pretrained features. The hybrid keeps the final classification step simple and performs well, while the base softmax remains stronger on the saved accuracy measure. Keeping the findings together with the original evidence makes the project easier to understand and build on.

Team details are in [CONTRIBUTORS.md](../CONTRIBUTORS.md); usage terms remain in [LICENCE](../LICENCE).
