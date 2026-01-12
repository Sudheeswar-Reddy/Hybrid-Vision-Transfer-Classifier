import os

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report
import numpy as np
import random

#GLOBAL SEEDING
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)


#UPDATED PATHS ONLY
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
model.fit(X_train, y_train, epochs=10, verbose=1, shuffle=False) # shuffle=False for total consistency

# 3. Calculate Advanced Metrics
y_probs = model.predict(X_test)
y_pred = np.argmax(y_probs, axis=1)
y_test_oh = tf.one_hot(y_test, depth=len(class_names)).numpy()

print(f"\n--- PARENT BASIS METRICS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_oh, y_probs):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_oh, y_probs):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# Save
feature_extractor = models.Model(inputs=model.input, outputs=x)
feature_extractor.save(os.path.join(model_dir, "mango_basis_extractor.h5"))