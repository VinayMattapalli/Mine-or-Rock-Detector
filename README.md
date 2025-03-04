Mine or Rock Detector 🪨⛏️



A machine learning-based sonar signal classification system that predicts whether an object is a Mine or a Rock based on sonar signal readings. Built using Logistic Regression, this model analyzes sonar data to distinguish between these two categories with high accuracy.


🔍 Project Overview


This project utilizes Logistic Regression to classify sonar signals as either:

⛏️ Mine (M)
🪨 Rock (R)


🔹 Features



✅ Trained on the Sonar dataset


✅ Real-time input via Gradio UI


✅ Automatically detects Mine vs. Rock


✅ Machine Learning model with optimized hyperparameters


✅ Supports interactive testing with user-provided data

📁 Dataset


The dataset consists of 208 sonar readings, each with 60 features representing sonar signal intensities. It is labeled as either:


"M" → Mine
"R" → Rock


The dataset is preprocessed to remove redundant features and improve model accuracy.


🛠️ Installation


1️⃣ Clone the Repository
sh
Copy
Edit
git clone https://github.com/VinayMattapalli/Mine-or-Rock-Detector.git


cd mine-or-rock-detector


2️⃣ Install Dependencies


Ensure you have Python 3.8+ installed, then run:


sh
Copy
Edit
pip install -r requirements.txt
If using a virtual environment:


sh
Copy
Edit
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate


pip install -r requirements.txt


3️⃣ Run the Application
Launch the Gradio Web UI:

sh
Copy
Edit


python rock_or_mine_prediction_ml.py


After running, open http://127.0.0.1:7860 in your browser.

📊 Model Training


The Logistic Regression model was trained using:

Scikit-Learn’s Logistic Regression


Train-Test Split (80% Train, 20% Test)


Feature selection (removal of redundant features)


Hyperparameter tuning (C=10 for optimized performance)


🔹 Model Performance


Metric	Baseline Model	Optimized Model


Accuracy  	         80.95%	         88.10%


Precision (M)	       79.17%	         84.00%


Recall (M)	         86.36%	         95.45%


F1-score (M)	       82.61%	         89.36%


📌 The optimized model provides a significant boost in recall and F1-score! 🚀




🖥️ Usage


🎯 Using the Model via Gradio UI


Run the script:


sh

Copy

Edit

python rock_or_mine_prediction_ml.py

Open http://127.0.0.1:7860 in your browser.

Enter 60 comma-separated sonar signal values.

Click "Submit" to get a prediction:

⛏️ Mine

🪨 Rock

💻 Running the Model in Python

python

Copy

Edit

from joblib import load

import numpy as np

# Load trained model

model = load("mine_rock_model.pkl")


# Example input

input_data = np.array([0.0283, 0.0599, ..., 0.0079]).reshape(1, -1)


# Prediction

prediction = model.predict(input_data)

print("Prediction:", "Mine" if prediction[0] == "M" else "Rock")

🔗 Technologies Used

Python 3.8+

Scikit-Learn

Pandas & NumPy

Joblib (for model persistence)

Gradio (for web UI)

📝 License

This project is licensed under the MIT License. Feel free to modify and use it.



👨‍💻 Developed by: Vinay Mattapalli

🔗 GitHub: https://github.com/VinayMattapalli

🙌 Contributions & feedback are welcome! If you find issues or want to improve the model, feel free to create a pull request.

🚀 Star ⭐ the Repository if You Like It!
If this project helps you, consider giving it a ⭐ on GitHub!

Happy Coding! 🎯🔥
