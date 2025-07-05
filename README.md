# BankNote Authentication - FastAPI Deployment

This project demonstrates how to train a machine learning model to classify banknotes as genuine or counterfeit, and deploy the model using **FastAPI**. The primary objective is to learn and implement end-to-end ML model deployment using a lightweight, high-performance API framework.

## Project Overview

The goal of this project is to build a classification model that authenticates banknotes based on statistical features extracted from images, and expose it as a REST API. The model is built using scikit-learn's Random Forest Classifier and deployed using FastAPI.

The deployment allows for real-time prediction of whether a banknote is fake or genuine by sending JSON requests to the API.

## Key Learning Objective

The main intent of this project is to learn how to:

- Train and evaluate a machine learning model.
- Serialize and save the model using `pickle`.
- Create a REST API using FastAPI to serve the model.
- Handle input validation using `pydantic`.
- Run and test the API locally using Uvicorn.

## Dataset

The dataset used is the [BankNote Authentication Dataset](https://www.kaggle.com/datasets/hellbuoy/banknote-authentication) from Kaggle.

### Features:

- `variance`: Variance of the wavelet-transformed image  
- `skewness`: Skewness of the wavelet-transformed image  
- `curtosis`: Curtosis of the wavelet-transformed image  
- `entropy`: Entropy of the image  

### Target:

- `class`: 0 for fake note, 1 for genuine note

## Model

A Random Forest Classifier is trained on the dataset and evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  

After evaluation, the model is saved to disk as `random_forest_banknote_model.pkl` using the `pickle` library.

## API Details

The model is served through a FastAPI application. The API exposes endpoints for testing and predicting.

### Endpoints

- **GET /**  
  Returns a welcome message. Can be used as a health check.

- **POST /predict/**  
  Accepts a JSON input with the four statistical features and returns the prediction (0 or 1), along with a textual label ("Fake Note" or "Genuine Note").

### Input Format

```json
{
  "variance": float,
  "skewness": float,
  "curtosis": float,
  "entropy": float
}
```

### Output Format

```json
{
  "prediction": "The note is legit"
}
```
## Project Structure

    .
    ├── BankNote_Authentication.csv          # Dataset file
    ├── train_model.py                       # Script to train and pickle the model
    ├── random_forest_banknote_model.pkl     # Saved model file
    ├── main.py                              # FastAPI application
    └── README.md                            # Project documentation
