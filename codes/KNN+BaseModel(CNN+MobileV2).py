import os
import random

os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, classification_report
import numpy as np
import joblib

#GLOBAL SEEDING
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(42)

#PATHS
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
model_dir = os.path.join(project_root, "model")
dataset_path = os.path.join(project_root, "DataSet")
extractor_path = os.path.join(model_dir, "mango_basis_extractor.h5")

# 1. Use the "AI Eyes" (CNN + MobileV2)
if not os.path.exists(extractor_path):
    raise FileNotFoundError(f"Missing {extractor_path}. You must have the Parent .h5 file in the folder.")

feature_extractor = load_model(extractor_path, compile=False)

# 2. Load Data
X, y = [], []
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

print("Loading dataset for Hybrid training...")
for label_index, class_name in enumerate(class_names):
    class_folder = os.path.join(dataset_path, class_name)
    for file in os.listdir(class_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = load_img(os.path.join(class_folder, file), target_size=(224, 224))
            X.append(img_to_array(img) / 255.0)
            y.append(label_index)

X = np.array(X)
y = np.array(y)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Extraction (The Bridge)
X_train_feat = feature_extractor.predict(X_train, verbose=0)
X_test_feat = feature_extractor.predict(X_test, verbose=0)

# 5. Build the Hybrid Model
hybrid_model = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance', metric='cosine'))
])

print("Training the Hybrid Classifier...")
hybrid_model.fit(X_train_feat, y_train)

# 6. Consistent Metrics
y_pred = hybrid_model.predict(X_test_feat)
y_probs = hybrid_model.predict_proba(X_test_feat)
y_test_oh = tf.one_hot(y_test, depth=len(class_names)).numpy()

print(f"\n" + "="*30)
print(f" STABLE HYBRID MODEL REPORT ")
print(f"="*30)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"MSE:      {mean_squared_error(y_test_oh, y_probs):.4f}")
print(f"MAE:      {mean_absolute_error(y_test_oh, y_probs):.4f}")
print("-" * 30)
print("CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred, target_names=class_names))

#SAVE THE HYBRID MODEL
save_path = os.path.join(model_dir, "final_hybrid_model.pkl")
joblib.dump(hybrid_model, save_path)

print(f"\n✅ SUCCESS: Only the Hybrid Classifier has been saved to: {save_path}")