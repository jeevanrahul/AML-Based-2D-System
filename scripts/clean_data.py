import pandas as pd

def clean_data(input_path=r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\master_dataset\master_dataset.csv", output_path=r"C:\AML-kaushal\AML-2D-System-MyWorks\AML_project\cleaned_master_data.csv"):
    df = pd.read_csv(input_path)
    print("Initial shape:", df.shape)

    # Debug info
    print("\nMissing values per column:")
    print(df.isnull().sum())

    if 'area' in df.columns:
        print("\nAny negative areas?")
        print((df['area'] < 0).sum())

    if 'perimeter' in df.columns:
        print("\nAny negative perimeters?")
        print((df['perimeter'] < 0).sum())

    # Drop duplicates
    df = df.drop_duplicates()

    # Keep only rows with valid 'area', 'perimeter', and 'shape_type'
    df = df.dropna(subset=['area', 'perimeter', 'shape_type'])

    # Filter invalid numeric values
    df = df[df['area'] >= 0]
    df = df[df['perimeter'] >= 0]

    print("\nCleaned shape:", df.shape)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

# ✅ Make sure this line is present so it runs!
if __name__ == "__main__":
    clean_data(output_path=r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\master_dataset\cleaned_master_data.csv")

