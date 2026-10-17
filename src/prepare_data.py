"""
Runner script to execute the preprocessing pipeline and save the output.
Run from the project root: python -m src.prepare_data
"""

import os
from src.preprocessing import load_data, clean_data, binarize_target, split_and_scale


def main():
    raw_path = "data/raw/winequality-red.csv"
    out_dir = "data/processed"
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Did you download it using the curl command in the README?"
        )

    os.makedirs(out_dir, exist_ok=True)

    print("Loading data...")
    df = load_data(raw_path)
    print(f"Original shape: {df.shape}")

    print("Cleaning data (removing duplicates)...")
    df = clean_data(df)
    print(f"Shape after cleaning: {df.shape}")

    print("Binarizing target...")
    df = binarize_target(df)
    print(f"Good wine ratio: {df['is_good'].mean():.2%}")

    print("Splitting and scaling...")
    train_df, test_df = split_and_scale(df)

    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved {len(train_df)} rows to {train_path}")
    print(f"Saved {len(test_df)} rows to {test_path}")


if __name__ == "__main__":
    main()
