import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
ds = os.getenv("DATASET")


dataset = pd.read_csv(ds)

x = dataset.drop(columns=["Crop", "Production", "Yield"])  #Using features other than the dropped ones
y = dataset["Crop"]  # Crop is my target(NOT GROWN OR YIELD REMEMBER!!!)

categorical_features = ["Season", "State"]

ct = ColumnTransformer(transformers=
                       [("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)], 
                       remainder='passthrough')

x_processed = ct.fit_transform(x)

sc = StandardScaler()
x_scaled = sc.fit_transform(x_processed)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


# Train Model
clf = RandomForestClassifier(class_weight="balanced",n_estimators=500, random_state=42)
clf.fit(x_train, y_train)


# Model Evaluation
y_pred = clf.predict(x_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))


# Save the Model
# joblib.dump(clf, "rfm.pkl")
# joblib.dump(ct, "ct.pkl")
# joblib.dump(sc, "sc.pkl")


# Predict the crop from the user
if __name__ == "__main__":
    while True:
        print('\n')

        input = {
            "Crop_Year": int(input("Enter Crop Year: ")),
            "Season": input("Enter Season: "),
            "State": input("Enter State: "),
            "Area": float(input("Enter Area(in hectares): ")),
            "Annual_Rainfall": float(input("Enter Annual Rainfall(in mm): ")),
            "Fertilizer": float(input("Enter Fertilizer Used(in kg): ")),
            "Pesticide": float(input("Enter Pesticide Used(in kg): "))
        }

        df = pd.DataFrame([input])

        processed = ct.transform(df)
        scaled = sc.transform(processed)

        recommended_crop = clf.predict(scaled)[0]
        # print(recommended_crop)
        print(f"\nRecommended Crop to Grow: {recommended_crop}")