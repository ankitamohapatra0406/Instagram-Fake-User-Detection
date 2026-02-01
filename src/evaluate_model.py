from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "instagram_users.csv"
MODELS_DIR = BASE_DIR / "models"

df = pd.read_csv(DATA_PATH)

X = df.drop("is_fake", axis=1)
y = df["is_fake"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = joblib.load(MODELS_DIR / "scaler.pkl")
model = joblib.load(MODELS_DIR / "random_forest_model.pkl")

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
