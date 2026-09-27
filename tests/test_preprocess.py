import numpy as np
import pytest
from PIL import Image

from preprocess import preprocess_digit


def test_preprocess_output_shape():
    # Create simple black image
    image_array = np.zeros((100, 100), dtype=np.uint8)

    # Draw a white vertical line
    image_array[20:80, 45:55] = 255

    image = Image.fromarray(image_array)

    result = preprocess_digit(image)

    assert result.shape == (28, 28)


def test_preprocess_pixel_range():
    image_array = np.zeros((100, 100), dtype=np.uint8)
    image_array[20:80, 45:55] = 255

    image = Image.fromarray(image_array)

    result = preprocess_digit(image)

    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_detects_empty_image():
    image_array = np.zeros((100, 100), dtype=np.uint8)

    image = Image.fromarray(image_array)

    with pytest.raises(ValueError):
        preprocess_digit(image)