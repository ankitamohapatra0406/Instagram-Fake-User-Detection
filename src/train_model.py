from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "instagram_users.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)
df = pd.read_csv(DATA_PATH)

X = df.drop("is_fake", axis=1)
y = df["is_fake"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# model
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X_train, y_train)

joblib.dump(rf, MODELS_DIR / "random_forest_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
print("Model and scaler saved successfully in models/")
