"""
Train a baseline Random Forest classifier on the processed Wine Quality data.

Run from the project root: python -m src.train
"""

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


def main():
    train_path = "data/processed/train.csv"
    model_dir = "models"
    model_path = os.path.join(model_dir, "baseline_rf.pkl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. "
            "Run `python -m src.prepare_data` first."
        )

    print("Loading training data...")
    df = pd.read_csv(train_path)
    
    X = df.drop(columns=["is_good"])
    y = df["is_good"]
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print("Training Random Forest classifier (with balanced class weights)...")
    clf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight="balanced"  # This tells the model to care about the minority class
)
    clf.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, model_path)
    
    print(f"Model trained and saved to {model_path}")
    
    # Quick sanity check on the training data
    train_acc = clf.score(X, y)
    print(f"Training accuracy: {train_acc:.2%}")


if __name__ == "__main__":
    main()
