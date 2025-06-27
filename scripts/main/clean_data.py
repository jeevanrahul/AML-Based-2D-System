import pandas as pd
import os

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    print(f"📊 Initial shape: {df.shape}")

    # Diagnostic Info
    print("\n🩺 Missing values per column:")
    print(df.isnull().sum())

    if 'area' in df.columns:
        print(f"\n❓ Negative area values: {(df['area'] < 0).sum()}")

    if 'perimeter' in df.columns:
        print(f"❓ Negative perimeter values: {(df['perimeter'] < 0).sum()}")

    # Clean Steps
    df = df.drop_duplicates()
    df = df.dropna(subset=['area', 'perimeter', 'shape_type'])
    df = df[(df['area'] >= 0) & (df['perimeter'] >= 0)]

    print(f"\n✅ Cleaned shape: {df.shape}")

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    clean_data(
        input_path=r"data/processed/master_dataset/master_dataset.csv",
        output_path=r"data/processed/master_dataset/cleaned_master_data.csv"
    )
