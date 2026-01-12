import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
import random

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
print(f"MSE: {mean_squared_error(y_test_oh, y_probs):.4f}")
print(f"MAE: {mean_absolute_error(y_test_oh, y_probs):.4f}")