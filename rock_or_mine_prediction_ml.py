# -*- coding: utf-8 -*-
"""Mine or Rock Detector"""

import numpy as np
import pandas as pd
import joblib
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

# Define file paths
DATA_PATH = r"C:\Users\gkeer\OneDrive\Desktop\ML Projects\Rock or Mine ML\sonar data.csv"
MODEL_PATH = r"C:\Users\gkeer\OneDrive\Desktop\ML Projects\Rock or Mine ML\rock_or_mine_model.pkl"

"""Data Preprocessing"""

# Load dataset
data_sonar = pd.read_csv(DATA_PATH, header=None)

# Splitting features and labels
X = data_sonar.drop(columns=60, axis=1)
Y = data_sonar[60]

# Split data into train & test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Train & save model if not already saved
if not os.path.exists(MODEL_PATH):
    training_model = LogisticRegression()
    training_model.fit(X_train, Y_train)
    joblib.dump(training_model, MODEL_PATH)
else:
    training_model = joblib.load(MODEL_PATH)

"""Prediction Function"""
def predict_mine_or_rock(input_string):
    try:
        # Convert input string to list of floats
        input_values = list(map(float, input_string.strip().split(',')))

        # Ensure exactly 60 values
        if len(input_values) != 60:
            return "⚠️ Error: Please enter exactly 60 numerical values separated by commas."

        # Convert to numpy array & reshape
        input_data = np.array(input_values).reshape(1, -1)

        # Predict using loaded model
        prediction = training_model.predict(input_data)

        # Return result with emoji
        return "🪨 The object is a Rock" if prediction[0] == 'R' else "⛏️ The object is a Mine"

    except ValueError:
        return "⚠️ Error: Please enter valid numerical values separated by commas."

"""Gradio Interface"""
iface = gr.Interface(
    fn=predict_mine_or_rock,
    inputs=gr.Textbox(lines=3, placeholder="Enter 60 values separated by commas..."),
    outputs="text",
    title="Mine or Rock Detector",
    description="Enter 60 sensor values to predict whether the object is a Mine or Rock.",
)

iface.launch(share=True)
