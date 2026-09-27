from PIL import Image
import numpy as np


def preprocess_digit(image: Image.Image) -> np.ndarray:
    # Convert to grayscale
    image = image.convert("L")

    # Convert pixel values from 0-255 to 0-1
    image_array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    # MNIST uses white digits on a black background
    if np.mean(image_array) > 0.5:
        image_array = 1.0 - image_array

    # Estimate background brightness from the image borders
    border_pixels = np.concatenate([
        image_array[0, :],
        image_array[-1, :],
        image_array[:, 0],
        image_array[:, -1]
    ])

    background = np.median(border_pixels)

    # Remove the background while preserving thin digit strokes
    image_array = np.clip(
        image_array - background,
        0.0,
        1.0
    )

    # Stretch remaining digit values back toward 0-1
    if image_array.max() > 0:
        image_array = image_array / image_array.max()

    # Find the actual digit
    mask = image_array > 0.08
    coordinates = np.argwhere(mask)

    if coordinates.size == 0:
        raise ValueError("No digit detected")

    # Find bounding box
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0) + 1

    # Crop empty space
    digit = image_array[
        y_min:y_max,
        x_min:x_max
    ]

    # Preserve proportions while resizing
    height, width = digit.shape

    scale = 20 / max(height, width)

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    digit_image = Image.fromarray(
        (digit * 255).astype(np.uint8)
    )

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create MNIST-style 28x28 canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.float32
    )

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = np.array(
        digit_image,
        dtype=np.float32
    ) / 255.0

    return canvas