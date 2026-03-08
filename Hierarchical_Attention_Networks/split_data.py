import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')

    input_file = os.path.join(data_dir, 'ecommerceDataset.csv')
    train_file = os.path.join(data_dir, 'train.csv')
    valid_file = os.path.join(data_dir, 'valid.csv')
    test_file = os.path.join(data_dir, 'test.csv')

    # Read CSV
    df = pd.read_csv(input_file, usecols=[0, 1], names=['label', 'text'], header=0)

    # Drop missing values
    df = df.dropna(subset=['label', 'text'])

    # Remove noisy labels
    label_counts = df['label'].value_counts()
    valid_labels = label_counts[label_counts >= 10].index
    df = df[df['label'].isin(valid_labels)]

    print(f"Total samples: {len(df)}")

    # ---- STEP 1: Split train (70%) and temp (30%) ----
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df['label']
    )

    # ---- STEP 2: Split temp into valid (15%) and test (15%) ----
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,   # 50% of 30% -> 15%
        random_state=42,
        stratify=temp_df['label']
    )

    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(valid_df)}")
    print(f"Test size: {len(test_df)}")

    # Save datasets
    train_df.to_csv(train_file, index=False)
    valid_df.to_csv(valid_file, index=False)
    test_df.to_csv(test_file, index=False)

if __name__ == "__main__":
    main()