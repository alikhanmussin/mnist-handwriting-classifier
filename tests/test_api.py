import io

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_predict_image_endpoint():
    # Create a simple digit-like image
    image_array = np.zeros((100, 100), dtype=np.uint8)

    # Draw a white vertical line
    image_array[20:80, 45:55] = 255

    image = Image.fromarray(image_array)

    # Save image into memory instead of creating a real file
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = client.post(
        "/predict-image",
        files={
            "file": (
                "test.png",
                image_bytes,
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "predicted_digit" in data
    assert "confidence" in data
    assert "uncertain" in data
    assert "top_3" in data

    assert 0 <= data["predicted_digit"] <= 9
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["top_3"]) == 3