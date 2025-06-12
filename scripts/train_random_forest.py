import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    # Paths
    data_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\data\processed\master_dataset\cleaned_master_data.csv"
    model_save_path = r"C:\Users\welcome\Desktop\New folder\AML-Based-2D-System\models"
    os.makedirs(model_save_path, exist_ok=True)

    # 1. Load the dataset
    df = pd.read_csv(data_path)
    if 'shape_type' not in df.columns:
        raise ValueError("Column 'shape_type' not found in dataset.")

    # 2. Features and Labels
    X = df.drop(columns=['shape_type'])
    y = df['shape_type']

    # 3. Encode Labels (important for later usage)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    # 5. Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy:.4f}")
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Save model & encoder
    joblib.dump(model, os.path.join(model_save_path, "random_forest_model.pkl"))
    joblib.dump(le, os.path.join(model_save_path, "label_encoder.pkl"))
    print(f"\n✅ Model saved at: {model_save_path}\\random_forest_model.pkl")
    print(f"✅ Label encoder saved at: {model_save_path}\\label_encoder.pkl")

    # 8. Optional: Feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n🔥 Top 5 Important Features:")
    print(importances.head(5))

if __name__ == "__main__":
    main()
