# 📊 Customer Churn Analysis: Predicting & Preventing Customer Loss 🔄

## Overview
This project was developed as part of my internship at **SaiKetSystems**. It provides a comprehensive analysis of customer churn using the Telco Customer Churn dataset. The goal is to predict customer churn and identify key factors that contribute to customer loss. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, and the development of a neural network model to predict churn.

The repository for this project can be found here: [SaiKetSystems Customer Churn Analysis](https://github.com/mohamed682004/SaiKetSystems_Customer-Churn-Analysis).

---

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Data Preprocessing](#data-preprocessing)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
4. [Feature Engineering](#feature-engineering)
5. [Model Development](#model-development)
6. [Model Performance Analysis](#model-performance-analysis)
7. [Recommendations](#recommendations)
8. [Repository Structure](#repository-structure)
9. [Internship Context](#internship-context)

---

## Dataset Overview
The dataset used in this analysis is the Telco Customer Churn dataset, which includes information about customers' demographics, services they subscribe to, and whether they have churned.

### Key Columns:
- `customerID`: Unique identifier for each customer
- `gender`: Customer's gender
- `SeniorCitizen`: Whether the customer is a senior citizen
- `Partner`: Whether the customer has a partner
- `Dependents`: Whether the customer has dependents
- `tenure`: Number of months the customer has stayed with the company
- `PhoneService`: Whether the customer has a phone service
- `MultipleLines`: Whether the customer has multiple lines
- `InternetService`: Type of internet service
- `OnlineSecurity`: Whether the customer has online security
- `OnlineBackup`: Whether the customer has online backup
- `DeviceProtection`: Whether the customer has device protection
- `TechSupport`: Whether the customer has tech support
- `StreamingTV`: Whether the customer has streaming TV
- `StreamingMovies`: Whether the customer has streaming movies
- `Contract`: Type of contract
- `PaperlessBilling`: Whether the customer has paperless billing
- `PaymentMethod`: Payment method
- `MonthlyCharges`: Monthly charges
- `TotalCharges`: Total charges
- `Churn`: Whether the customer churned

---

## Data Preprocessing
The preprocessing steps include:
- Handling missing values
- Converting categorical variables to numerical using one-hot encoding
- Dropping irrelevant columns
- Standardizing numerical features

```python
# Convert TotalCharges to numerical
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
# Fill missing TotalCharges with MonthlyCharges
data['TotalCharges'] = data['TotalCharges'].fillna(data['MonthlyCharges'])
# Drop customerID
data.drop('customerID', axis=1, inplace=True)
# One-hot encode categorical variables
data = pd.get_dummies(data, columns=columns_to_encode)
```

---

## Exploratory Data Analysis (EDA)
The EDA includes:
- Distribution plots for numerical and categorical features
- Correlation analysis
- Visualization of key relationships (e.g., MonthlyCharges vs TotalCharges)

```python
# Plot distribution of numerical data
for column in numerical_data.columns:
    plt.figure(figsize=(10,6))
    sns.histplot(numerical_data[column], kde=True)
    plt.title(column)
    plt.show()
```

---

## Feature Engineering
Key steps include:
- Mapping binary categorical variables to 0 and 1
- Dropping highly correlated features
- Dropping features with low correlation to the target variable (Churn)

```python
# Map binary categorical variables
data['Partner'] = data['Partner'].map({'Yes': 1, 'No': 0})
data['Dependents'] = data['Dependents'].map({'Yes': 1, 'No': 0})
data['PhoneService'] = data['PhoneService'].map({'Yes': 1, 'No': 0})
data['PaperlessBilling'] = data['PaperlessBilling'].map({'Yes': 1, 'No': 0})
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
```

---

## Model Development
A neural network model is developed using Keras with the following architecture:
- Input layer with 64 neurons and ReLU activation
- Batch normalization and dropout layers to prevent overfitting
- Hidden layers with 32, 16, and 8 neurons
- Output layer with sigmoid activation for binary classification

```python
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    BatchNormalization(),
    
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## Model Performance Analysis
The model achieves an accuracy of **81%** on the test set. The training and validation accuracy plots show that the model converges quickly and does not overfit.

```python
# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

### Key Observations:
- The model converges quickly within the first few epochs.
- Test accuracy is slightly higher than training accuracy, indicating no overfitting.
- The use of dropout and batch normalization is effective in stabilizing the training process.

---

## Recommendations
1. **Early Stopping**: Implement early stopping to reduce unnecessary training time.
2. **Learning Rate Adjustment**: Adjust the learning rate to smooth out fluctuations in test accuracy.
3. **Additional Regularization**: Consider adding more regularization techniques if further stability is needed.

```python
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```

---

## Repository Structure
The repository is organized as follows:
```
SaiKetSystems_Customer-Churn-Analysis/
├── data/
│   └── Telco_Customer_Churn_Dataset.csv
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb
├── README.md
└── requirements.txt
```

---

## Internship Context
This project was completed as part of my internship at **SaiKetSystems**, where I worked on real-world data analysis and machine learning tasks. The goal was to develop a predictive model for customer churn and provide actionable insights to reduce customer attrition. This experience allowed me to apply my skills in data preprocessing, exploratory data analysis, and machine learning to solve a critical business problem.

---

## Conclusion
This notebook provides a detailed analysis of customer churn, from data preprocessing to model development and evaluation. The neural network model achieves good performance and can be further optimized with the recommended techniques. This project was a valuable learning experience during my internship at SaiKetSystems, and I am grateful for the opportunity to contribute to their data-driven initiatives.
