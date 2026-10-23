# Experiment Log: Wine Quality Classification

## Objective
Train a machine learning model to classify wines as "Good" (quality >= 7) or "Not Good" (quality < 7) based on their physicochemical properties.

## Dataset
- **Source:** UCI Machine Learning Repository
- **Initial Shape:** (1599, 12)
- **Cleaned Shape:** (1359, 12) — *Removed duplicate rows to prevent data leakage.*
- **Target Distribution:** Imbalanced. Only ~13.5% of the wines are classified as "Good".

## Preprocessing
- **Binarization:** Converted the multiclass target (3-8) into a binary target (0 or 1).
- **Scaling:** Applied `StandardScaler` to all 11 features (mean = 0, variance = 1).
- **Split:** 80/20 train/test split with `stratify=y` to maintain the class imbalance ratio in both sets.

## Experiment 1: Baseline Random Forest
- **Model:** `RandomForestClassifier(n_estimators=100, random_state=42)`
- **Training Accuracy:** 100.00% *(Expected, as decision trees can memorize the training data).*
- **Test Set Performance:**
  - **Accuracy:** ~90%
  - **Recall:** ~37%
  - **F1 Score:** ~46%
- **Observation:** The model was heavily biased toward the majority class. It missed over 60% of the actual "Good" wines because there were so few of them in the training data. Accuracy was misleadingly high due to the class imbalance.

## Experiment 2: Random Forest with Balanced Class Weights
- **Model:** `RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')`
- **Purpose:** To penalize the model more heavily for misclassifying the minority class ("Good" wine).
- **Test Set Performance:**
  - **Accuracy:** ~85% *(Slight drop is acceptable).*
  - **Recall:** ~65% *(Significant improvement).*
  - **F1 Score:** ~55%
- **Observation:** By adjusting the class weights, we successfully traded a small amount of overall accuracy for a much better ability to detect good wines. This is the preferred model.

## Next Steps
- Try hyperparameter tuning (e.g., adjusting `max_depth` or `min_samples_split`).
- Try other models (XGBoost, Logistic Regression).
- Look into SMOTE (Synthetic Minority Over-sampling Technique) for data-level balancing.
