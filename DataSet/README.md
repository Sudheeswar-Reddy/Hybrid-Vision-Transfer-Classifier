# Mango leaf dataset

The committed dataset contains **4,000 image files**, evenly divided across eight folders. This count was checked against the Git tree at source revision `dd9c9f2d3be0d5330adb1ef40cac55b0d44ff500`.

| Folder / label | Images | Index |
|---|---:|---:|
| Anthracnose | 500 | 0 |
| Bacterial Canker | 500 | 1 |
| Cutting Weevil | 500 | 2 |
| Die Back | 500 | 3 |
| Gall Midge | 500 | 4 |
| Healthy | 500 | 5 |
| Powdery Mildew | 500 | 6 |
| Sooty Mould | 500 | 7 |

Folder names include spaces. They match [class_indices.json](../class_indices.json). Each script sorts class directories to assign numeric labels and reads `.jpg`, `.jpeg` and `.png` files. The scripts do not use the JSON mapping as an input.

The previous project documentation attributes the images to the [Mango Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset). Check the source's attribution and licence before redistributing or using the data for another purpose; the repository's code licence should not be assumed to settle dataset rights.

Keep the dataset unchanged when comparing against the recorded results. Changing images, filenames or file enumeration can change which samples fall into the split, even with the same random seed. Do not put new class directories here unless you intend to create a different experiment.

This image-level split does not demonstrate generalisation to new farms, plants or capture sessions. The repository does not include a saved group-based split or an external field-test dataset.
