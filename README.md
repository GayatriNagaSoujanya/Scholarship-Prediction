# Scholarship Prediction using Machine Learning

This project uses machine learning to predict the likelihood of receiving a scholarship based on various applicant features. By analyzing a synthetic dataset that simulates real-world conditions, the model provides insights into factors influencing scholarship eligibility.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

---

## Overview

This project predicts scholarship eligibility based on features like academic performance, extracurricular involvement, financial need, and more. It automates the selection process, improving transparency and efficiency.

## Dataset

A synthetic dataset was created for this project, containing features such as:
- **GPA**: Grade Point Average
- **Test Scores**: Standardized test scores (e.g., SAT, ACT)
- **Extracurricular Activities**: Level of involvement in extracurriculars
- **Financial Need**: Indicator of financial background
- **Recommendation Strength**: Strength of letters of recommendation
- **Other relevant features** as simulated to represent real applicant data

Place the dataset in the `data/` folder (e.g., `data/scholarship_data.csv`).

## Requirements

Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```
## Project Structure

├── data/

│   └── scholarship_data.csv       # Synthetic dataset for training and testing

├── notebooks/

│   └── scholarship_prediction.ipynb  # Notebook with analysis, training, and evaluation

├── models/

│   └── final_model.pkl            # Saved trained model

└── README.md

## Preprocessing
Data preprocessing steps include:

Handling Missing Values: Using imputation or removing missing data

Feature Scaling: Scaling features to improve model performance

Encoding Categorical Variables: Encoding features like extracurricular involvement or recommendation strength

Feature Selection: Selecting the most impactful features for model training

```bash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data/scholarship_data.csv')

# Preprocessing steps
data.fillna(data.mean(), inplace=True)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
```

## Modeling
```bash
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
```

## Evaluation
Evaluate the model on test data using metrics such as accuracy, precision, recall, and F1-score:
```bash
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Results
Accuracy: Achieved 96% accuracy on the test set.

Important Features: GPA, Test Scores, Financial Need were among the most influential features.

## Usage
1.Clone the repository:
```bash
git clone https://github.com/GayatriNagaSoujanya/Scholarship-Prediction.git
```
2.Install dependencies:
```bash
pip install -r requirements.txt
```
3.Run the notebook scholarship_prediction.ipynb to train and evaluate the model.

4.Use the saved model (models/final_model.pkl) to predict scholarship eligibility for new applicants.
