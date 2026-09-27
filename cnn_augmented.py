from tensorflow import keras

# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add grayscale channel
x_train = x_train[..., None]
x_test = x_test[..., None]

# Data augmentation
data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.08),
    keras.layers.RandomTranslation(
        height_factor=0.08,
        width_factor=0.08
    ),
    keras.layers.RandomZoom(0.08)
])

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    data_augmentation,

    keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D((2, 2)),

    keras.layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu"
    ),

    keras.layers.MaxPooling2D((2, 2)),

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
    epochs=20,
    validation_split=0.2,
    callbacks=[early_stopping]
)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nAugmented CNN test accuracy:", test_accuracy)

model.save("mnist_cnn_augmented.keras")