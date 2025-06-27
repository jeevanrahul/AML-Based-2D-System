import os
import pandas as pd

# === Config ===
normalized_folder = r"data/processed/processed_normalized"
master_save_path = r"data/processed/master_dataset/master_dataset.csv"

# === Load and Combine Normalized CSVs ===
all_dataframes = []

for file in sorted(os.listdir(normalized_folder)):
    if file.endswith(".csv"):
        file_path = os.path.join(normalized_folder, file)
        df = pd.read_csv(file_path)

        # Extract shape type from filename like 'normalized_circle_1.csv'
        shape_type = file.replace("normalized_", "").replace(".csv", "").split("_")[0]
        df["shape_type"] = shape_type

        all_dataframes.append(df)

# === Save Combined Dataset ===
if all_dataframes:
    os.makedirs(os.path.dirname(master_save_path), exist_ok=True)
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df.to_csv(master_save_path, index=False)
    print(f"✅ Master dataset saved at:\n{master_save_path}")
else:
    print("⚠️ No CSV files found in:", normalized_folder)
