# ==============================================================================
# train_model.py
#
# Main script to train and evaluate the multi-class X-ray disease detector.
# ==============================================================================

# --- 1. Import Libraries ---
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# --- 2. Configuration and Paths ---
# IMPORTANT: Update this path to your dataset location
base_dir = 'd2/'
train_dir = os.path.join(base_dir, 'train')

val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Model Parameters
IMG_WIDTH, IMG_HEIGHT = 1024, 1024
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = "xray_detector.h5"

# --- 3. Data Preprocessing and Augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create Data Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class info
num_classes = len(train_generator.class_indices)
class_labels = list(train_generator.class_indices.keys())
print(f"\nFound {num_classes} classes: {class_labels}\n")

# --- 4. Model Building (Transfer Learning) ---
base_model = Xception(weights="imagenet", include_top=False, input_tensor=Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
for layer in base_model.layers:
    layer.trainable = False

head_model = base_model.output
head_model = GlobalAveragePooling2D()(head_model)
head_model = Dense(256, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(num_classes, activation="softmax")(head_model)

model = Model(inputs=base_model.input, outputs=head_model)

# Compile the Model
optimizer = Adam(learning_rate=1e-4)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
model.summary()

# --- 5. Model Training ---
print("\n--- Starting Model Training ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)
print("\n--- Training Complete ---")

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")


# --- 6. Performance Evaluation ---
print("\n--- Evaluating Model on Test Data ---")
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Classification Report
print("\nClassification Report:")
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Disease')
plt.ylabel('Actual Disease')
plt.show()
