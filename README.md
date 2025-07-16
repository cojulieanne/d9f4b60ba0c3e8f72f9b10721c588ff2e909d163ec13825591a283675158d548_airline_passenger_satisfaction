# Airline Passenger Satisfaction (Model Development Pipeline)

## Project Overview

This project addresses the problem of predicting airline passenger satisfaction using survey and service-related data. By modeling this classification task, we aim to help airlines better understand the factors that lead to higher satisfaction and act upon them to improve customer experience. The dataset includes labeled instances with demographic details, flight-related attributes, and in-flight service ratings. It is small (<100MB), well-structured, and ideal for training a single supervised ML model end-to-end. This makes it a good candidate for demonstrating a production-oriented MLOps workflow.

## How to Get the Data

Access the dataset and data card from Kaggle:  
[https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

After downloading the `train.csv` and `test_csv`, these two files were combined into one dataset `full_data.csv` and removed the unnecessary column `Unnamed: 0`. 
This allows for flexibility of data split based on custom settings.

## Project Structure

```text
.
├── analysis/
│   ├── eda.ipynb                   # Exploratory Data Analysis
├── data/
│   ├── full_data.csv               # Combined Train and Test Data
│   └── test.csv                    # Original Data from Kaggle
│   └── train.csv                   # Original Data from Kaggle
├── results/                        # Trained model artifacts and results (e.g., model.pkl)
├── src/
│   ├── data_preprocessing.py      # Load, clean, split dataset
│   ├── evaluation.py              # Save/Load model and evaluate on test
│   ├── model_training.py          # Model training with CV
│   └── pipeline.py                # End-to-end pipeline for modelling 
├── tests/
│   └── test_data_preprocessing.py # Sample unit tests (optional)
├── .pre-commit-config.yaml        # Pre-commit hook definitions
├── pyproject.toml                 # UV-compatible dependency definitions
└── README.md
```
