# evaluate_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import os
from sklearn.metrics import confusion_matrix
import joblib
import os

# Paths
master_csv_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\master_dataset\cleaned_master_data.csv"
model_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models\random_forest_model.pkl"
encoder_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models\label_encoder.pkl"

# Load data and model
df = pd.read_csv(master_csv_path)
model = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Features and target
X = df.drop(columns=["shape_type"])
y = df["shape_type"]
y_encoded = label_encoder.transform(y)

# Predict
y_pred = model.predict(X)

# Confusion Matrix
conf_mat = confusion_matrix(y_encoded, y_pred)
labels = label_encoder.classes_

plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png")
plt.close()

# Classification Report
report = classification_report(y_encoded, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv("visuals/classification_report.csv")
print("✅ Classification report and confusion matrix saved.")

# Feature Importance
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="viridis")
plt.title("Top Feature Importances")
plt.tight_layout()

# Ensure the 'visuals' directory exists
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/feature_importance.png")
plt.close()

print("✅ Feature importance chart saved.")
