# Running the experiments

The three scripts are kept in their original form. This guide describes what they actually do and where their outputs go.

| Script | Input and split | Classifier | Saved output |
|---|---|---|---|
| `BaseModel(CNN+MobileV2).py` | 224 × 224 RGB / 255; stratified 80/20 split, seed 42 | Frozen ImageNet MobileNetV2 + pooling + trained softmax | `model/mango_basis_extractor.h5` |
| `KNN+BaseModel(CNN+MobileV2).py` | Same resizing and stratified split procedure; requires the extractor | StandardScaler + KNN, k=3, cosine, distance weighting | `model/final_hybrid_model.pkl` |
| `KNN_model.py` | 100 × 100 RGB flattened; unstratified 80/20 split, seed 42 | StandardScaler + default KNN, k=3 | No fitted model saved |

## Base, then hybrid

Run these from the repository root in a [separate experimental checkout](../docs/reproducing-results.md):

```bash
python "codes/BaseModel(CNN+MobileV2).py"
python "codes/KNN+BaseModel(CNN+MobileV2).py"
```

Both scripts locate `DataSet/` and `model/` relative to their own files. Do not run only the hybrid in a fresh clone: it raises `FileNotFoundError` until the base script has saved its extractor.

The backbone is frozen. Training changes the softmax head, while the saved extractor ends at global average pooling and excludes that head. The hybrid therefore uses pretrained pooled features, not a fine-tuned backbone or a distilled model.

## Pure KNN's fixed path

`KNN_model.py` expects its dataset under:

```text
C:\Users\WORK STATION\OneDrive\Desktop\Projects\Micro Credential Course\Hybrid-Vision-Transfer-Classifier\DataSet
```

Its path does not depend on the current working directory. Running `python codes/KNN_model.py` from another folder does not fix it. An unchanged-code run requires the dataset to be available at that exact location. For a fresh portable setup, a code change would be needed; this documentation update intentionally leaves the script intact. Do not treat the historical KNN figures as a successful rerun on your machine.

## Plots, metrics and model files

Each script prints accuracy, MCC, probability MSE/MAE and a classification report, then opens confusion-matrix and ROC windows. Close the plot windows to let execution finish. Save a plot through its window's save control if you want a new PNG. Console logs can be captured with `Tee-Object` in PowerShell, as shown in the reproduction guide.

The original `results/` files were captured separately; the scripts contain no automatic PNG or PDF export. Both model-save filenames are fixed, so reruns overwrite the corresponding model in that checkout. H5 and PKL outputs are ignored by Git. Keep the extractor and hybrid pipeline together, with their class mapping and environment record.

## Labels and preprocessing

Class directories are sorted alphabetically, but image filenames are not sorted. [class_indices.json](../class_indices.json) records the supplied eight-class mapping; the scripts recompute their own labels and do not read or rewrite that JSON file. A replacement dataset needs its own matching mapping.

The MobileNetV2 paths use division by 255, exactly as written. No augmentation or application-specific `preprocess_input` is applied. Keep this distinction when reproducing the saved experiment; changing preprocessing would define a different experiment.
