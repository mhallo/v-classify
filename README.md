# V-classify

This project is a pretty simple machine learning pipeline that predicts the genre of a game based solely on its title.

Built with `scikit-learn`, `Polars`, and `TfidfVectorizer`, it includes a full pipeline for preprocessing, training, evaluation, and prediction.

---

## Project Structure

```
.
├── data/
│   └── games.csv               # Dataset of video games provided from Kaggle
├── models/
│   └── genre_classifier.pkl    # Saved trained model
├── src/
   ├── predictor.py                # Script to load model & predict genres
   └── train_classifier.py         # Model training script
```

## Overview

Cleans and preprocesses game titles

Trains a genre classifier using TF-IDF and Logistic Regression

Balances class distribution for more accurate results

Supports prediction via a CLI or script

## Installing

Build and activate a virtual environment, and install the requirements to run the trainer and the predictor

`pip3 install -r requirements.txt`

## Formatting

This project uses `ruff` for formatting and linting.

## Training 

Invoke the training script

`python3 src/train_classifier.py`

## Predicting

For now, invoke the predictor and pass in the string of the title you'd like to predict

`python3 src/predictor.py`
