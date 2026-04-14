import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/churn.csv")

# Drop unnecessary column
df = df.drop("customerID", axis=1)

# Convert target
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# Encode categorical columns
for col in df.select_dtypes(include="object").columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, preds))