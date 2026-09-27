import tensorflow as tf
from tensorflow import keras
import numpy as np

# Training data
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    0,
    1,
    1,
    1
], dtype=float)

# Build the neural network
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Configure how the model learns
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    x,
    y,
    epochs=500,
    verbose=0
)

# Test predictions
predictions = model.predict(x)

print(predictions)

print("\nModel weights:")
for layer in model.layers:
    print(layer.get_weights())