import os
import random
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                             classification_report, confusion_matrix, matthews_corrcoef,
                             roc_curve, auc)

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
model_dir = os.path.join(project_root, "model")
dataset_path = os.path.join(project_root, "DataSet")
extractor_path = os.path.join(model_dir, "mango_basis_extractor.h5")

if not os.path.exists(extractor_path):
    raise FileNotFoundError(f"Missing {extractor_path}.")

feature_extractor = load_model(extractor_path, compile=False)

# 2. Load Data
X, y = [], []
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

for label_index, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = load_img(os.path.join(class_folder, file), target_size=(224, 224))
            X.append(img_to_array(img) / 255.0)
            y.append(label_index)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42, stratify=y)

# 4. Feature Extraction
X_train_feat = feature_extractor.predict(X_train, verbose=0)
X_test_feat = feature_extractor.predict(X_test, verbose=0)

# 5. Build Pipeline
hybrid_model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine'))
])

hybrid_model.fit(X_train_feat, y_train)

# 6. Evaluation
y_pred = hybrid_model.predict(X_test_feat)
y_probs = hybrid_model.predict_proba(X_test_feat)
y_test_oh = tf.one_hot(y_test, depth=len(class_names)).numpy()

print(f"\n==============================")
print(f" STABLE HYBRID MODEL REPORT ")
print(f"==============================")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"MCC:      {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"MSE:      {mean_squared_error(y_test_oh, y_probs):.4f}")
print(f"MAE:      {mean_absolute_error(y_test_oh, y_probs):.4f}")
print("-" * 30)
print("CLASSIFICATION REPORT:\n", classification_report(y_test, y_pred, target_names=class_names))

# VISUALIZATIONS
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Hybrid Model')
plt.show()

plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve: Hybrid Model')
plt.legend()
plt.show()

joblib.dump(hybrid_model, os.path.join(model_dir, "final_hybrid_model.pkl"))