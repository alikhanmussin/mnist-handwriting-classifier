import tensorflow as tf
from tensorflow import keras

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixels from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN expects a channel dimension:
# (28, 28) -> (28, 28, 1)
x_train = x_train[..., None]
x_test = x_test[..., None]

# Build CNN
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    keras.layers.Flatten(),

    keras.layers.Dense(
        64,
        activation="relu"
    ),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(
        10,
        activation="softmax"
    )
])

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

history = model.fit(
    x_train,
    y_train,
    epochs=15,
    validation_split=0.2,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nCNN test accuracy:", test_accuracy)

# Save CNN
model.save("mnist_cnn.keras")