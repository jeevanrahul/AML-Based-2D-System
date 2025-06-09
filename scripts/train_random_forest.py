import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

# Paths - update if needed
data_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\master_dataset\master_dataset.csv"
model_save_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models"
os.makedirs(model_save_path, exist_ok=True)

# 1. Load the combined normalized dataset
df = pd.read_csv(data_path)

# Check dataset structure (uncomment to debug)
# print(df.head())
# print(df.columns)

# 2. Prepare features and labels
# Assuming 'shape_name' column contains labels
if 'shape_type' not in df.columns:
    raise ValueError("Column 'shape_type' not found in dataset. Check your CSV.")

X = df.drop(columns=['shape_type'])
y = df['shape_type']

# 3. Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_mat)
print("Classification Report:")
print(class_report)

# 6. Save the trained model for later use
model_file = os.path.join(model_save_path, "random_forest_model.pkl")
joblib.dump(model, model_file)
print(f"Trained model saved at: {model_file}")