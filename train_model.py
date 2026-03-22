import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv(r"C:\Users\varsh\Desktop\problems\depression\Student Depression Dataset.csv")
data = data.drop(["id"], axis=1)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
encoders = {}
categorical_cols = data.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save encoders
pickle.dump(encoders, open("encoders.pkl", "wb"))

# -----------------------------
# Split Data
# -----------------------------
X = data.drop("Depression", axis=1)
y = data["Depression"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Accuracy
# -----------------------------
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)

# -----------------------------
# Save Model
# -----------------------------
pickle.dump(model, open("model.pkl", "wb"))
# Save feature column order
feature_columns = X.columns
pickle.dump(feature_columns, open("features.pkl", "wb"))

print("Model + Encoders Saved Successfully!")


print(model.get_params())

