import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

# 1. Load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("Training images:", x_train.shape)
print("Training labels:", y_train.shape)
print("Test images:", x_test.shape)
print("Test labels:", y_test.shape)

print("\nFirst label:", y_train[0])

plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.show()

# Normalize pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = keras.Sequential([
    keras.Input(shape=(28, 28)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax")
])

# Configure training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# Train
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Test on unseen data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest accuracy:", test_accuracy)

# Save the trained model
model.save("mnist_model.keras")

import numpy as np

# Predict the first test image
prediction = model.predict(x_test[0:1])

predicted_digit = np.argmax(prediction)

print("\nActual digit:", y_test[0])
print("Predicted digit:", predicted_digit)
print("Probabilities:", prediction[0])

# Predict all test images
all_predictions = model.predict(x_test)

predicted_classes = np.argmax(all_predictions, axis=1)
from sklearn.metrics import classification_report

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predicted_classes
    )
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, predicted_classes)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

display.plot()
plt.title("MNIST Confusion Matrix")
plt.show()

# Find incorrect predictions
wrong_indices = np.where(predicted_classes != y_test)[0]

print("\nWrong predictions:", len(wrong_indices))
print("Correct predictions:", len(y_test) - len(wrong_indices))

# Look at the first mistake
index = wrong_indices[0]

print("\nActual:", y_test[index])
print("Predicted:", predicted_classes[index])

plt.imshow(x_test[index], cmap="gray")
plt.title(
    f"Actual: {y_test[index]} | Predicted: {predicted_classes[index]}"
)
plt.show()




plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()