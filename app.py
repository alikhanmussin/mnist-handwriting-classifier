from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from tensorflow import keras
from PIL import Image
from io import BytesIO
from preprocess import preprocess_digit
import numpy as np

app = FastAPI()

# Load trained model once when API starts
model = keras.models.load_model("mnist_cnn_finetuned.keras")


class PredictionRequest(BaseModel):
    pixels: list[list[float]]


@app.get("/")
def root():
    return {"message": "MNIST API is running"}


@app.post("/predict")
def predict(request: PredictionRequest):
    image = np.array(request.pixels, dtype=float)

    # Model expects shape: (batch, 28, 28)
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    is_uncertain = confidence < 0.60

    top_3_indices = np.argsort(prediction[0])[-3:][::-1]

    top_3 = [
        {
            "digit": int(index),
            "confidence": float(prediction[0][index])
        }
        for index in top_3_indices
    ]

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "uncertain": is_uncertain,
        "top_3": top_3
    }


@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(BytesIO(contents))

    try:
        canvas = preprocess_digit(image)
    except ValueError as error:
        return {"error": str(error)}

    # CNN expects: batch, height, width, channels
    image_array = canvas.reshape(1, 28, 28, 1)

    prediction = model.predict(image_array)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    is_uncertain = confidence < 0.60

    top_3_indices = np.argsort(prediction[0])[-3:][::-1]

    top_3 = [
        {
            "digit": int(index),
            "confidence": float(prediction[0][index])
        }
        for index in top_3_indices
    ]

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "uncertain": is_uncertain,
        "top_3": top_3
    }