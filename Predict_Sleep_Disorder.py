# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:27:51 2023

@author: Benny
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv('sleephealth.csv')
df.columns = ['Person ID', 'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Sleep Disorder']

# Encode the gender column, to have numerical values
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Map the occupations
occupation_mapping = df['Occupation'].astype('category')
print('Occupations:')
print(dict(enumerate(occupation_mapping.cat.categories)))
# Encode occupation, create separate columns for each occupation
df['Occupation'] = df['Occupation'].astype('category').cat.codes

# Encode BMI using ordinal encoding
bmi_mapping = {
    'Normal': 1,
    'Normal Weight': 2,
    'Overweight': 3,
    'Obese': 4
}
df['BMI Category'] = df['BMI Category'].map(bmi_mapping)
    
# Split the bloodpressure
df['Systolic Pressure'] = df['Blood Pressure'].str.split('/').str[0].astype(int)
df['Diastolic Pressure'] = df['Blood Pressure'].str.split('/').str[1].astype(int)
df.drop('Blood Pressure', axis=1, inplace=True)  # Drop the original combined column

# Convert the categorical 'Sleep Disorder' column into separate dummy/indicator columns.
df = pd.get_dummies(df, columns=['Sleep Disorder'], prefix='Sleep_Disorder')

# Create a new column 'Sleep_Disorder_None' which is True if neither 'Sleep_Disorder_Insomnia' or
# 'Sleep_Disorder_Sleep Apnea' are True for a particular row.
df['Sleep_Disorder_None'] = ~df[['Sleep_Disorder_Insomnia', 'Sleep_Disorder_Sleep Apnea']].any(axis=1)

# Now, derive a single 'Sleep Disorder' column from the dummies.
conditions = [
    df['Sleep_Disorder_Insomnia'] == 1,
    df['Sleep_Disorder_Sleep Apnea'] == 1,
    df['Sleep_Disorder_None'] == 1
]
choices = ['Insomnia', 'Sleep Apnea', 'None']
df['Sleep Disorder'] = np.select(conditions, choices, default='None')
# Using the `np.select` method, we specify our conditions and corresponding choices.

# Drop the dummy variables as we have the combined column now.
df.drop(columns=['Sleep_Disorder_None', 'Sleep_Disorder_Insomnia', 'Sleep_Disorder_Sleep Apnea'], inplace=True)

# Setup features and target for the machine learning model
target = df['Sleep Disorder']
features = df.drop(columns=['Person ID', 'Sleep Disorder'])

# My own personal sleeping data to predict:
my_data = {
    'Gender': [1],
    'Age': [25],
    'Occupation': [9],  # Using the mapped occupation code
    'Sleep Duration': [7],
    'Quality of Sleep': [8],
    'Physical Activity Level': [2],
    'Stress Level': [3],
    'BMI Category': [2],  # Use the BMI mapping
    'Systolic Pressure': [119], #I actually dont know my blood pressure, but i googled the average of a 25 year old man so 
    'Diastolic Pressure': [70],
    'Heart Rate': [68],
    'Daily Steps': [3000]
}

my_data_df = pd.DataFrame(my_data)
my_data_df = my_data_df[features.columns] # To rearrange the columns to match the same order as in features.

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize logistic regression model
lr_model = LogisticRegression(max_iter=10000)  # increased max_iter for convergence

# Initialize random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the logistic regression model to the training data
lr_model.fit(X_train, y_train.values.ravel())

# Fit the random forest model to the training data
rf_model.fit(X_train, y_train)

# Predict logistic regression outcomes on the test set
y_pred = lr_model.predict(X_test)

# Predict random forest outcomes on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the performance of logistic regression
print('Performance of Logistic Regression:')
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
confusion_lr = confusion_matrix(y_test, y_pred)
print('Confusion Matrix for Logistic Regression:')
print(confusion_lr)

# Evaluate the performance of random forest
print('Performance of Random Forest:')
print(accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
confusion_rf = confusion_matrix(y_test, y_pred_rf)
print('Confusion Matrix for Random Forest:')
print(confusion_rf)

results = pd.DataFrame({
    'Actual': y_test.values.ravel(),
    'Predicted': y_pred
})

results_rf = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf
})
print('Logistic regression:')
print(results)

print('Random Forest:')
print(results_rf)

my_prediction_lr = lr_model.predict(my_data_df)
my_prediction_rf = rf_model.predict(my_data_df)
print('Your data predictions:')
print(f"Logistic Regression Prediction: {my_prediction_lr[0]}")
print(f"Random Forest Prediction: {my_prediction_rf[0]}")







