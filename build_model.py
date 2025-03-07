from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Image augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, zoom_range=0.2, 
    horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_data = train_datagen.flow_from_directory(
    "wildlife_dataset/train", target_size=(224, 224), batch_size=32, class_mode='categorical'
)
val_data = val_test_datagen.flow_from_directory(
    "wildlife_dataset/val", target_size=(224, 224), batch_size=32, class_mode='categorical'
)
test_data = val_test_datagen.flow_from_directory(
    "wildlife_dataset/test", target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False
)

# Automatically detect the number of classes
NUM_CLASSES = len(train_data.class_indices)
class_labels = list(train_data.class_indices.keys())
print(f"Detected {NUM_CLASSES} classes:", train_data.class_indices)

# Load MobileNetV2 with pre-trained ImageNet weights
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained layers

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
output_layer = Dense(NUM_CLASSES, activation="softmax")(x)  # Dynamically set number of output neurons

# Create model
model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()

# Define callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_wildlife_classifier.h5", save_best_only=True, monitor="val_accuracy", mode="max")
]

# Train the model
EPOCHS = 20
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predictions
test_data.reset()
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_data.classes

# Classification report
report = classification_report(y_true, y_pred, target_names=test_data.class_indices.keys())
print("Classification Report:")
print(report)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Save final model
model.save("wildlife_classifier.h5")

# Plot training history
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training & Validation Accuracy')
plt.show()