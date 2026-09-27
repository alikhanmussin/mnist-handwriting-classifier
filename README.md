# MNIST Handwriting Classifier

An end-to-end handwritten digit classification project built with **TensorFlow, CNNs, image preprocessing, fine-tuning, FastAPI, pytest, and GitHub Actions**.

The project started with a basic MNIST classifier and was progressively improved using convolutional neural networks, data augmentation, custom handwriting data, preprocessing, and fine-tuning.

## Results

Performance was measured on a separate set of **50 handwritten test images** that were not used for training.

| Model          | Custom Test Accuracy |
| -------------- | -------------------: |
| Normal CNN     |                  82% |
| Augmented CNN  |                  90% |
| Fine-tuned CNN |              **94%** |

The final fine-tuned model correctly classified:

```text
47 / 50 handwritten test images
```

## Model Progression

```text
Basic neural network
        ↓
CNN
        ↓
CNN + data augmentation
        ↓
Fine-tuning on custom handwriting
        ↓
Final model
```

Data augmentation improved custom handwriting accuracy from:

```text
82% → 90%
```

Fine-tuning improved it further to:

```text
90% → 94%
```

## Custom Dataset

I created a custom handwritten digit dataset to evaluate the model on handwriting outside the original MNIST dataset.

```text
custom_train/
    100 images
    10 examples per digit

custom_test/
    50 images
    5 examples per digit
```

The 100 custom training images were split into:

```text
80 images → fine-tuning
20 images → validation
```

The separate 50-image test set remained untouched during training and was used for the final benchmark.

## Image Preprocessing

Uploaded images are transformed into a format similar to MNIST before being passed to the CNN.

The preprocessing pipeline:

```text
Uploaded image
      ↓
Convert to grayscale
      ↓
Normalize pixel values
      ↓
Detect digit
      ↓
Remove unnecessary background
      ↓
Crop digit
      ↓
Preserve aspect ratio
      ↓
Resize
      ↓
Center on 28×28 canvas
      ↓
CNN prediction
```

This allows the API to handle handwritten images that are larger than the original MNIST `28 × 28` format.

## Fine-Tuning

The augmented CNN is used as the starting model:

```text
mnist_cnn_augmented.keras
```

It is then fine-tuned using the custom handwriting dataset with a smaller learning rate.

The resulting model is saved as:

```text
mnist_cnn_finetuned.keras
```

The final model reached:

```text
94% accuracy
```

on the untouched 50-image custom test set.

## FastAPI Inference API

The trained model is exposed through a FastAPI API.

Start the server:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

The main endpoint is:

```text
POST /predict-image
```

Upload an image containing a handwritten digit.

Example response:

```json
{
  "predicted_digit": 7,
  "confidence": 0.9995,
  "uncertain": false,
  "top_3": [
    {
      "digit": 7,
      "confidence": 0.9995
    },
    {
      "digit": 2,
      "confidence": 0.0003
    },
    {
      "digit": 3,
      "confidence": 0.00003
    }
  ]
}
```

## Confidence and Top-3 Predictions

The API returns:

- predicted digit
- prediction confidence
- uncertainty flag
- top 3 most likely digits

This makes the prediction output more informative than returning only the most likely class.

## Evaluation

The custom benchmark compares three models:

```text
Normal CNN
Augmented CNN
Fine-tuned CNN
```

It reports:

- prediction for every custom image
- confidence
- overall accuracy
- per-digit accuracy
- misclassified images
- confusion matrix

Example final result:

```text
Normal CNN:      41/50 = 82%
Augmented CNN:   45/50 = 90%
Fine-tuned CNN:  47/50 = 94%
```

## Confusion Matrix

A confusion matrix is generated for the fine-tuned CNN to show which digits are confused with each other.

On the final custom benchmark, the fine-tuned model made only three errors out of 50 images.

## Automated Tests

The project uses `pytest`.

Run:

```bash
python -m pytest -v
```

Current tests verify:

- preprocessing returns a `28 × 28` image
- normalized pixels remain between `0` and `1`
- empty images are rejected
- `/predict-image` returns the expected API response structure

Current result:

```text
4 passed
```

## Continuous Integration

GitHub Actions automatically runs the test suite on:

```text
push
pull request
```

Workflow:

```text
GitHub push
     ↓
Create Ubuntu environment
     ↓
Install Python
     ↓
Install requirements
     ↓
Run pytest
     ↓
Pass / Fail
```

Workflow file:

```text
.github/workflows/tests.yml
```

## Installation

Clone the repository and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn app:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Run Tests

```bash
python -m pytest -v
```

## Run Custom Benchmark

```bash
python benchmark_custom.py
```

## Project Structure

```text
ml-first-model/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── custom_train/
├── custom_test/
│
├── tests/
│   ├── test_api.py
│   └── test_preprocess.py
│
├── app.py
├── preprocess.py
├── benchmark_custom.py
├── fine_tune.py
│
├── cnn_mnist.py
├── cnn_augmented.py
├── mnist.py
├── train.py
│
├── mnist_model.keras
├── mnist_cnn.keras
├── mnist_cnn_augmented.keras
├── mnist_cnn_finetuned.keras
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

- Python
- TensorFlow / Keras
- Convolutional Neural Networks
- NumPy
- Pillow
- scikit-learn
- Matplotlib
- FastAPI
- Uvicorn
- pytest
- GitHub Actions

## What I Learned

This project demonstrates the full machine-learning workflow:

```text
dataset
→ training
→ evaluation
→ error analysis
→ CNN
→ data augmentation
→ preprocessing
→ custom benchmark
→ fine-tuning
→ API inference
→ automated tests
→ continuous integration
```

It also demonstrates why test data must remain separate from training data and how augmentation and fine-tuning can improve generalization to handwriting outside the original training distribution.
