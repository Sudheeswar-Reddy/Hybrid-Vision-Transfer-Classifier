import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                             classification_report, confusion_matrix, matthews_corrcoef,
                             roc_curve, auc)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
set_seeds(42)

project_root = r"C:\Users\WORK STATION\OneDrive\Desktop\Projects\Micro Credential Course\Hybrid-Vision-Transfer-Classifier"
dataset_path = os.path.join(project_root, "DataSet")
img_height, img_width = 100, 100

X, y = [], []
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

for label_index, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = load_img(os.path.join(class_folder, file), target_size=(img_height, img_width))
            X.append(img_to_array(img).flatten())
            y.append(label_index)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn_pure = KNeighborsClassifier(n_neighbors=3)
knn_pure.fit(X_train_s, y_train)

y_pred = knn_pure.predict(X_test_s)
y_probs = knn_pure.predict_proba(X_test_s)
y_test_oh = tf.one_hot(y_test, depth=len(class_names)).numpy()

print(f"\n--- PURE KNN METRICS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"MCC:      {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"MSE:      {mean_squared_error(y_test_oh, y_probs):.4f}")
print(f"MAE:      {mean_absolute_error(y_test_oh, y_probs):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# VISUALIZATION: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Pure KNN')
plt.show()

# VISUALIZATION: ROC Curve
plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve: Pure KNN')
plt.legend()
plt.show()