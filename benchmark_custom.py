import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocess import preprocess_digit


# Load both models
normal_model = keras.models.load_model("mnist_cnn.keras")
augmented_model = keras.models.load_model("mnist_cnn_augmented.keras")
finetuned_model = keras.models.load_model("mnist_cnn_finetuned.keras")


# Folder containing your handwritten digits
folder = "custom_test"

# Optional: save exactly what the CNN receives
processed_folder = "processed_custom"
os.makedirs(processed_folder, exist_ok=True)


normal_correct = 0
augmented_correct = 0
finetuned_correct = 0
total = 0

actual_labels = []
augmented_predictions = []
finetuned_predictions = []

per_digit_total = {digit: 0 for digit in range(10)}
per_digit_normal = {digit: 0 for digit in range(10)}
per_digit_augmented = {digit: 0 for digit in range(10)}
per_digit_finetuned = {digit: 0 for digit in range(10)}


for filename in sorted(os.listdir(folder)):

    if not filename.endswith(".png"):
        continue

    # Example:
    # 7.1.png -> actual digit = 7
    actual_digit = int(filename[0])

    path = os.path.join(folder, filename)

    # Open original image
    image = Image.open(path)

    try:
        # Use SAME preprocessing as FastAPI
        canvas = preprocess_digit(image)
    except ValueError:
        print(f"{filename}: no digit detected")
        continue

    # Save processed version for inspection
    processed_image = Image.fromarray(
        (canvas * 255).astype(np.uint8)
    )

    processed_image.save(
        os.path.join(processed_folder, filename)
    )

    # CNN expects:
    # (batch, height, width, channels)
    image_input = canvas.reshape(1, 28, 28, 1)

    # Normal CNN prediction
    normal_prediction = normal_model.predict(
        image_input,
        verbose=0
    )

    # Augmented CNN prediction
    augmented_prediction = augmented_model.predict(
        image_input,
        verbose=0
    )

    finetuned_prediction = finetuned_model.predict(
        image_input,
        verbose=0
    )

    finetuned_digit = int(np.argmax(finetuned_prediction))
    finetuned_confidence = float(np.max(finetuned_prediction))

    normal_digit = int(np.argmax(normal_prediction))
    augmented_digit = int(np.argmax(augmented_prediction))

    if finetuned_digit != actual_digit:
        print(
            f"FINE-TUNED WRONG: {filename} | "
            f"Actual: {actual_digit} | "
            f"Predicted: {finetuned_digit}"
        )

    actual_labels.append(actual_digit)
    augmented_predictions.append(augmented_digit)
    finetuned_predictions.append(finetuned_digit)

    normal_confidence = float(np.max(normal_prediction))
    augmented_confidence = float(np.max(augmented_prediction))

    per_digit_total[actual_digit] += 1

    if normal_digit == actual_digit:
        normal_correct += 1
        per_digit_normal[actual_digit] += 1

    if augmented_digit == actual_digit:
        augmented_correct += 1
        per_digit_augmented[actual_digit] += 1

    if finetuned_digit == actual_digit:
        finetuned_correct += 1
        per_digit_finetuned[actual_digit] += 1

    total += 1

    print(
        f"{filename} | "
        f"Actual: {actual_digit} | "
        f"Normal: {normal_digit} ({normal_confidence:.2%}) | "
        f"Augmented: {augmented_digit} ({augmented_confidence:.2%}) | "
        f"Fine-tuned: {finetuned_digit} ({finetuned_confidence:.2%})"
    )


print("\n--- RESULTS ---")

print(
    f"Normal CNN: "
    f"{normal_correct}/{total} "
    f"({normal_correct / total * 100:.1f}%)"
)

print(
    f"Augmented CNN: "
    f"{augmented_correct}/{total} "
    f"({augmented_correct / total * 100:.1f}%)"
)

print(
    f"Fine-tuned CNN: "
    f"{finetuned_correct}/{total} "
    f"({finetuned_correct / total * 100:.1f}%)"
)

print("\n--- PER DIGIT RESULTS ---")

print("Digit | Normal | Augmented | Fine-tuned")
print("--------------------------")

for digit in range(10):
    digit_total = per_digit_total[digit]

    normal_count = per_digit_normal[digit]
    augmented_count = per_digit_augmented[digit]

    normal_accuracy = normal_count / digit_total * 100
    augmented_accuracy = augmented_count / digit_total * 100

    finetuned_count = per_digit_finetuned[digit]
    finetuned_accuracy = finetuned_count / digit_total * 100

    print(
        f"{digit:>5} | "
        f"{normal_count}/{digit_total} ({normal_accuracy:>5.1f}%) | "
        f"{augmented_count}/{digit_total} ({augmented_accuracy:>5.1f}%) | "
        f"{finetuned_count}/{digit_total} ({finetuned_accuracy:>5.1f}%)"
    )

cm = confusion_matrix(
    actual_labels,
    finetuned_predictions,
    labels=list(range(10))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10))
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Custom Handwriting - Fine-tuned CNN")
plt.show()