# Titanic Survival Prediction App
<div>
<img width="1677" height="872" alt="Screenshot 2026-01-04 144241" src="https://github.com/user-attachments/assets/3fca7e84-35f3-499d-956f-1cd6bbabdedb" />
<img width="1687" height="819" alt="Screenshot 2026-01-04 144307" src="https://github.com/user-attachments/assets/bcd4e49c-3d99-4472-a0cd-8a4e03d06af3" />
</div>

This project predicts whether a passenger survived the Titanic disaster using demographic and travel-related information.  
It includes a machine learning pipeline, a Streamlit web app, and a Docker setup for easy deployment.

---

## Overview

The goal of this project is to predict passenger survival based on:
- Personal attributes
- Family relationships
- Ticket and travel details

The model is trained on the classic Titanic dataset and deployed as an interactive web application.

---

## Features Used

### Engineered Features
- FamilySize
- IsAlone
- Title (extracted from passenger name)

### Original Features
- Age
- Fare
- Passenger class
- Sex
- Embarkation port
- Siblings / spouses aboard
- Parents / children aboard

---

## Model Details

- **Algorithm:** Decision Tree Classifier  
- **Preprocessing:**
  - Median imputation for numeric features
  - Most frequent imputation for categorical features
  - Power transformation for skewed numeric data
  - One-hot encoding for categorical variables
- **Pipeline:** End-to-end Scikit-learn pipeline

---

## Model Evaluation

The model is evaluated on a hold-out test set using:
- Accuracy
- ROC-AUC score
- Confusion matrix

---

## Streamlit Application

The app supports:
- Single passenger survival prediction
- Batch prediction using CSV upload
- Survival probability output
- User-friendly interface

---

## Running Locally
```bash
Install dependencies:

pip install -r requirements.txt

Run the app:
streamlit run app/app.py

Running with Docker

Build the image:
docker build -t titanic-survival-app .


Run the container:
docker run -p 5000:8501 titanic-survival-app


Open:
http://localhost:5000

