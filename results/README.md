# Recorded results

These artifacts document the saved experiment on 13 February 2026. They have not been regenerated for this documentation update.

| Model | Accuracy | MCC | Probability MSE | Probability MAE | Console report |
|---|---:|---:|---:|---:|---|
| MobileNetV2 + softmax | 99.12% | 0.9900 | 0.0023 | 0.0079 | [Base](Results_BaseModel%28CNN%2BMobileV2%29.pdf) |
| MobileNetV2 features + KNN | 98.25% | 0.9801 | 0.0032 | 0.0058 | [Hybrid](Results_HybridModel%28KNN%2BBaseModel%29.pdf) |
| Raw-pixel KNN | 68.12% | 0.6380 | 0.0523 | 0.0847 | [Pure KNN](Results_PureKNN.pdf) |

Each report evaluates 800 images. MSE and MAE compare probabilities with one-hot labels, averaged over samples and classes; they are not errors in numeric class IDs or independent proof of calibration.

## Confusion matrices

Rows are actual classes; columns are predictions. Base and hybrid have 100 test images per class. Pure KNN's test support varies because its split is not stratified.

### Base model

![Base confusion matrix](ConfusionMatrix%28BaseModel%29.png)

The matrix has 793 correct predictions out of 800, or 99.125% before display rounding. The console report displays 99.12%.

### Hybrid model

![Hybrid confusion matrix](ConfusionMatrix%28HybridModel%29.png)

The hybrid correctly classifies 786 of 800 images. Anthracnose has 93 correct from 100; four are classified as Healthy, two as Gall Midge and one as Die Back.

### Raw-pixel KNN

![Pure KNN confusion matrix](ConfusionMatrix%28KNN%20model%29.png)

The raw-pixel model correctly classifies 545 of 800 images. Gall Midge has 51 correct out of 104; Powdery Mildew has 49 out of 91.

## ROC plots

The figures show one-vs-rest discrimination. Their printed AUC values are rounded, so avoid reporting extra precision from these images.

- [Base ROC](ROC%28BaseModel%29.png)
- [Hybrid ROC](ROC%28HybridModel%29.png)
- [Pure KNN ROC](ROC%28KNNModel%29.png)

## Original evidence

[original-results.zip](reference/original-results.zip) contains unchanged copies of all six PNGs and three PDFs. [manifest.json](reference/manifest.json) records SHA-256 hashes, the source revision, original code hashes and dataset Git tree. [recorded-metrics.json](reference/recorded-metrics.json) transcribes the summary numbers.

No historical trained weights were committed. This archive preserves evidence, not an exact reloadable historical model. Follow the [rerun guide](../docs/reproducing-results.md) and keep future outputs separate.
