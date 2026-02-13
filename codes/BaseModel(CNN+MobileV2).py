import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                             classification_report, confusion_matrix, matthews_corrcoef,
                             roc_curve, auc)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
dataset_path = os.path.join(project_root, "DataSet")
model_dir = os.path.join(project_root, "model")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

img_height, img_width = 224, 224

# 1. Load Data
X, y = [], []
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

for label_index, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = load_img(os.path.join(class_folder, file), target_size=(img_height, img_width))
            X.append(img_to_array(img) / 255.0)
            y.append(label_index)

X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42, stratify=y)

# 2. Build Basis
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False
x = layers.GlobalAveragePooling2D()(base_model.output)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, verbose=1, shuffle=False)

# 3. Calculate All Metrics
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_test_oh = tf.one_hot(y_test, depth=len(class_names)).numpy()

print(f"\n--- PARENT BASIS METRICS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"MCC:      {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"MSE:      {mean_squared_error(y_test_oh, y_probs):.4f}")
print(f"MAE:      {mean_absolute_error(y_test_oh, y_probs):.4f}")
print("\nClassification Report (includes F1-Score):\n", classification_report(y_test, y_pred, target_names=class_names))

# VISUALIZATION: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix: Base Model')
plt.show()

# VISUALIZATION: ROC Curve
plt.figure(figsize=(8, 6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_probs[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve: Base Model')
plt.legend()
plt.show()

feature_extractor = models.Model(inputs=model.input, outputs=x)
feature_extractor.save(os.path.join(model_dir, "mango_basis_extractor.h5"))