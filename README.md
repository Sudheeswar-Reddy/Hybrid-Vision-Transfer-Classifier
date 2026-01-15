# Hybrid-Vision-Transfer-Classifier

Achieved high-accuracy image classification by combining the spatial intelligence of **Deep Learning** with the robust decision-making of **Classical Machine Learning**. This project features a stable, reproducible architecture designed for categorical image datasets. Link of sample DataSet is provided and also it is available in this repository.

---

## 📁 Project Structure

```text
Hybrid-Vision-Transfer-Classifier/
│
├── codes/                        # Core Python scripts
│   ├── BaseModel(CNN+MobileV2).py    # CNN feature extractor training
│   ├── KNN_model.py                   # Pure KNN classifier
│   └── KNN+BaseModel(CNN+MobileV2).py # Hybrid KNN+CNN pipeline
│
├── model/                        # Saved model weights & serialized objects
│   ├── mango_basis_extractor.h5  # CNN Feature Extractor (Parent)
│   └── final_hybrid_model.pkl    # Hybrid KNN Pipeline (Student)
│
├── results/                      # Performance reports & analysis
│   ├── Results_BaseModel(CNN+MobileV2).pdf
│   └── Results_HybridModel(KNN+BaseModel).pdf
│
├── DataSet/                      # Categorical image dataset folders
│   ├── Anthracnose/
│   ├── Bacterial_Canker/
│   ├── Cutting_Weevil/
│   ├── Die_Back/
│   ├── Gall_Midge/
│   ├── Healthy/
│   ├── Powdery_Mildew/
│   └── Sooty_Mould/
│
├── class_indices                 # Class label mappings for prediction
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── CONTRIBUTORS.md               # Team member details
├── LICENSE                       # Dual licensing terms
└── README.md                     # Project documentation
```

---

## 🚀 The Hybrid Approach

This system provides **three different approaches** to mango disease classification, allowing for performance comparison:

### 1. Base Model (CNN + MobileNetV2)

The `BaseModel(CNN+MobileV2).py` script uses transfer learning with MobileNetV2 (pre-trained on ImageNet). This pure deep learning approach processes raw images through convolutional layers to extract spatial features and classify diseases using a standard neural network head.

### 2. Pure KNN Model

The `KNN_model.py` script implements a traditional K-Nearest Neighbors classifier. This baseline classical machine learning approach can be used to benchmark performance against deep learning methods.

### 3. Hybrid Model (KNN + CNN)

The `KNN+BaseModel(CNN+MobileV2).py` script combines the best of both worlds. It uses the CNN to extract a **1,280-dimensional feature vector** from images, then feeds these features into a **KNN Classifier** wrapped in a `StandardScaler` pipeline. This hybrid approach leverages deep learning's perception with classical machine learning's robust, distance-based classification logic.

---

## 📊 Performance & Stability

To ensure professional-grade reliability and academic integrity, this project is **100% deterministic**.

- **Global Seeding:** All scripts use `RandomSeed=42` across NumPy, TensorFlow, and Python's internal random module to ensure identical results on every run.
- **Metrics:** The model evaluates performance using **Accuracy**, **Mean Squared Error (MSE)**, and **Mean Absolute Error (MAE)**.
- **Locked Architecture:** The Hybrid model is designed to be "locked and stable," providing a consistent baseline for comparison.
- **Detailed Reports:** Performance analysis and evaluation metrics for both CNN and Hybrid models are available in the `results/` folder as comprehensive PDF reports.

---

## 📈 Results & Analysis

Comprehensive performance reports are available in the `results/` directory:

- **📄 Results_BaseModel(CNN+MobileV2).pdf** - Complete analysis of the pure deep learning approach
- **📄 Results_HybridModel(KNN+BaseModel).pdf** - Detailed evaluation of the hybrid KNN+CNN model

Both reports include:
- Confusion matrices
- Classification reports (Precision, Recall, F1-Score)
- Accuracy metrics across all 8 disease categories
- Training/validation performance curves
- Model comparison and insights

📁 **View Reports:** Navigate to the `/results` folder for detailed PDF documentation.

---

## 🛠️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Hybrid-Vision-Transfer-Classifier.git
cd Hybrid-Vision-Transfer-Classifier
```

### 2. Dataset Setup

This repository does not include the dataset to keep the repo size manageable and respect data privacy.

**To use this project:**

1. **Download the Dataset:**
   - Visit the [Mango Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)
   - Download and extract the dataset

2. **Setup the DataSet Folder:**
   ```bash
   # Create the DataSet directory if it doesn't exist
   mkdir -p DataSet
   ```

3. **Organize the Dataset:**
   - Extract the downloaded dataset into the `DataSet/` folder
   - Ensure your folder structure matches:
   ```
   DataSet/
   ├── Anthracnose/
   ├── Bacterial_Canker/
   ├── Cutting_Weevil/
   ├── Die_Back/
   ├── Gall_Midge/
   ├── Healthy/
   ├── Powdery_Mildew/
   └── Sooty_Mould/
   ```

### 3. Install Dependencies

Install all required libraries using the provided requirements file:

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install tensorflow scikit-learn numpy joblib Pillow pandas matplotlib seaborn
```

### 4. Pre-trained Models

Model weights (`.h5` and `.pkl` files) are **automatically generated** when you run the training scripts.

- Models will be saved in the `model/` folder after training
- If you want to use pre-trained models without training, contact the team or check the Releases section

### 5. Execution Order

The scripts use relative pathing, allowing the project to run immediately after cloning.

**Option 1: Train Base CNN Model**
```bash
cd codes
python "BaseModel(CNN+MobileV2).py"
```
This trains a pure deep learning model for mango disease classification.

**Option 2: Train Pure KNN Model**
```bash
python KNN_model.py
```
This trains a classical K-Nearest Neighbors classifier as a baseline.

**Option 3: Train Hybrid Model (Recommended)**
```bash
python "KNN+BaseModel(CNN+MobileV2).py"
```
This creates the hybrid model combining CNN feature extraction with KNN classification.

---

## ⚖️ Credits & Licensing

### Dataset Attribution

The dataset used in this project was sourced from **Kaggle**.

- **Dataset:** [Mango Leaf Disease Dataset](https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset)
- **Author:** Arya Shah
- **License:** CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial)
- **Usage:** This data is used strictly for educational/research purposes.

**Note:** The dataset is NOT included in this repository. Please download it from the link above and follow the setup instructions.

### Code License

This project is available under **Dual Licensing**:

#### 🆓 Non-Commercial License (CC BY-NC-SA 4.0)

**For Students, Researchers & Personal Projects:**
- ✅ Free to use, modify, and share
- ✅ Must give appropriate credit
- ✅ Must share derivatives under same license
- ❌ No commercial use allowed

[View Full License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

#### 💼 Commercial License

**For Businesses & Profit-Making Ventures:**
- Requires a separate commercial licensing agreement
- Includes revenue-sharing or one-time licensing fee
- Custom terms negotiable based on use case
- 📧 **Contact:** cb.ai.u4aar25053@cb.students.amrita.edu for commercial licensing inquiries

---

**© 2025 Amrita Vishwa Vidyapeetham, Coimbatore. All Rights Reserved.**

---

## 👨‍💻 Developer Note

This project was developed by students of **Amrita Vishwa Vidyapeetham, Coimbatore, Tamil Nadu, India** at the **School of Artificial Intelligence** as part of the **Micro Credential Course (Semester 1)** for the **Artificial Intelligence and Robotics (AAR)** branch.

The project demonstrates proficiency in Neural Network architecture, Hybrid Model design, Transfer Learning, and professional AI development practices.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

For detailed information about the team members and their contributions, see [CONTRIBUTORS.md](CONTRIBUTORS.md).

## ⭐ Show your support

Give a ⭐️ if this project helped you!
