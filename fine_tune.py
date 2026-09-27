import os
import numpy as np

from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split

from preprocess import preprocess_digit


# Folder containing the NEW training handwriting
folder = "custom_train"

images = []
labels = []


# Load and preprocess all custom training images
for filename in sorted(os.listdir(folder)):

    if not filename.lower().endswith(".png"):
        continue

    # Example:
    # 7.4.png  -> label 7
    # 7.10.png -> label 7
    label = int(filename.split(".")[0])

    path = os.path.join(folder, filename)

    image = Image.open(path)

    try:
        canvas = preprocess_digit(image)
    except ValueError:
        print(f"Skipped {filename}: no digit detected")
        continue

    # CNN expects height, width, channel
    images.append(canvas[..., np.newaxis])
    labels.append(label)


# Convert Python lists into NumPy arrays
x_custom = np.array(images, dtype=np.float32)
y_custom = np.array(labels, dtype=np.int64)

print("Custom images:", x_custom.shape)
print("Custom labels:", y_custom.shape)


# Split the 100 images:
# 80 for fine-tuning
# 20 for validation
x_train, x_val, y_train, y_val = train_test_split(
    x_custom,
    y_custom,
    test_size=0.20,
    random_state=42,
    stratify=y_custom
)

print("Training images:", x_train.shape)
print("Validation images:", x_val.shape)


# Load our already-trained augmented CNN
model = keras.models.load_model(
    "mnist_cnn_augmented.keras"
)


# Use a SMALL learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# Continue training on your handwriting
history = model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping]
)


# Check validation performance
val_loss, val_accuracy = model.evaluate(
    x_val,
    y_val
)

print(
    "\nFine-tuned validation accuracy:",
    val_accuracy
)


# Save as a NEW model
# Do not overwrite the augmented model
model.save("mnist_cnn_finetuned.keras")

print("\nSaved: mnist_cnn_finetuned.keras")