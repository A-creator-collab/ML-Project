"""
Evaluate the trained baseline model on the unseen test set.

Run from the project root: python -m src.evaluate
"""

import os
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def main():
    test_path = "data/processed/test.csv"
    model_path = "models/baseline_rf.pkl"

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}. Run prepare_data first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train first.")

    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["is_good"])
    y_test = test_df["is_good"]

    print("Loading trained model...")
    clf = joblib.load(model_path)

    print("Making predictions...")
    y_pred = clf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print("EVALUATION RESULTS (Test Set)")
    print("=" * 40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Good", "Good"]))


if __name__ == "__main__":
    main()
