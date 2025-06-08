import os
import pandas as pd

# Folder paths
normalized_folder = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\processed_normalized"
master_save_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\master_dataset\master_dataset.csv"

# List to store all dataframes
all_dataframes = []

# Loop through all files in normalized folder
for file in os.listdir(normalized_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(normalized_folder, file)
        df = pd.read_csv(file_path)

        # Use the file name to extract shape type
        shape_type = file.replace("normalized_", "").replace(".csv", "")
        df["shape_type"] = shape_type  # Add shape type column

        all_dataframes.append(df)

# Combine all dataframes
if all_dataframes:
    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df.to_csv(master_save_path, index=False)
    print(f"Master dataset saved successfully at: {master_save_path}")
else:
    print("No CSV files found in normalized folder.")
