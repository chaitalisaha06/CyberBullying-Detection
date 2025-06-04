import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np

# Step 1: Load the dataset
file_path = "D:\\Image Cyberbullying\\final_combined_image_data.csv"
print(f"Loading dataset from: {file_path}")
data = pd.read_csv(file_path)
print(f"Dataset shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(data.head())

# Step 2: Separate labels and numeric data
labels = data["offensive"]
numeric_data = data.drop(columns=["Image Name", "offensive"])

print("\nChecking numeric data types:")
print(numeric_data.dtypes)
print("\nAny NaNs in numeric data?", numeric_data.isnull().values.any())

# Step 3: Standardize the data
print("\nStandardizing data...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

print(f"Scaled data shape: {scaled_data.shape}")
print("Saving scaler...")
joblib.dump(scaler, "scaler.pkl")

# Step 4: Apply PCA to retain 95% variance
print("\nApplying PCA...")
pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(scaled_data)

print(f"PCA result shape: {pca_result.shape}")
print(f"Explained variance ratio (sum): {np.sum(pca.explained_variance_ratio_):.3f}")
print("Saving PCA...")
joblib.dump(pca, "pca.pkl")

# Save transformed data
n_components_retained = pca.n_components_
pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_components_retained)])
pca_df.to_csv("pca_transformed_data.csv", index=False)
print(f"PCA completed. Components retained: {n_components_retained}")
print("Saved transformed data as 'pca_transformed_data.csv'")

# Step 5: Combine features and labels
print("\nCombining PCA features with labels...")
pca_df["offensive"] = labels.map({'Yes': 1, 'No': 0})
print(pca_df["offensive"].value_counts())

# Step 6: Split into training and test sets
X = pca_df.drop(columns=["offensive"])
y = pca_df["offensive"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print("Training label distribution:\n", y_train.value_counts())
print("Testing label distribution:\n", y_test.value_counts())

# Step 7: Build and train Neural Network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nTraining model...")
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save("neural_network_trained_model1.h5")
print("Model saved as 'neural_network_trained_model1.h5'.")

# Step 8: Predictions and metrics
print("\nGenerating predictions...")

y_train_prob = model.predict(X_train).flatten()
y_train_pred = (y_train_prob >= 0.5).astype(int)

y_test_prob = model.predict(X_test).flatten()
y_test_pred = (y_test_prob >= 0.5).astype(int)

print("\nSample train probabilities:", y_train_prob[:10])
print("Sample train predictions:", y_train_pred[:10])
print("Sample train true labels:", y_train.values[:10])

print("\nSample test probabilities:", y_test_prob[:10])
print("Sample test predictions:", y_test_pred[:10])
print("Sample test true labels:", y_test.values[:10])

# Metrics
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_report = classification_report(y_test, y_test_pred)

print(f"\nTrain Accuracy: {accuracy_train:.4f}")
print(f"Test Accuracy: {accuracy_test:.4f}")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
