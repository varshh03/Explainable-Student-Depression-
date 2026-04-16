import shap
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
data = pd.read_csv("C:/Users/varsh/Desktop/problems/depression/Student Depression Dataset.csv")

data = data.drop(["id"], axis=1)

# Encode using saved encoders
encoders = pickle.load(open("encoders.pkl", "rb"))

for col, le in encoders.items():
    data[col] = le.transform(data[col])

X = data.drop("Depression", axis=1)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global explanation plot
shap.summary_plot(shap_values[1], X)