# STEP 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# STEP 2: Load Dataset

df = pd.read_csv("D:\\Sem5\\MP2\\StudentStreamDS.csv")
print("Dataset Loaded Successfully")
print(df.head())


# STEP 3: Data Cleaning


# Count missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

df = df[(df["Grade_10"] <= 100) & (df["Grade_12"] <= 100)]


# 3.1 - Handle missing values
def clean_missing_values(data):
    cleaned_df = data.copy()

    # Rows with only 1 missing numerical value → fill with column mean
    for col in ['Grade_10', 'Grade_12', 'JEE', 'CUET']:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())

    # Remove rows with more than 1 missing numerical value
    cleaned_df = cleaned_df.dropna(thresh=len(cleaned_df.columns) - 1)
    return cleaned_df

df = clean_missing_values(df)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())



# STEP 4:  Feature Engineering

df['JEE_100'] = df['JEE'] / 3.0
df['CUET_100'] = df['CUET'] / 8.0
df['Acad_Avg'] = (df['Grade_10'] + df['Grade_12']) / 2
df['Entrance_Avg'] = (df['JEE_100'] + df['CUET_100']) / 2
df['Combined_Score'] = 0.5 * df['Acad_Avg'] + 0.5 * df['Entrance_Avg']

print("\n Feature Engineering Complete")
print(df.head())

df["JEE_100"] = df["JEE_100"] * 2
df["CUET_100"] = df["CUET_100"] *2



X_temp = df.drop(columns=['Name', 'Phone', 'EmailID', 'Stream'])
y_temp = df['Stream']

# Encode categorical features in X_temp
X_temp_encoded = X_temp.copy()
for col in X_temp_encoded.columns:
    if X_temp_encoded[col].dtype == 'object':
        X_temp_encoded[col] = LabelEncoder().fit_transform(X_temp_encoded[col])

# Encode target
le = LabelEncoder()
y_temp_encoded = le.fit_transform(y_temp)

rf_temp = RandomForestClassifier(random_state=42)
rf_temp.fit(X_temp_encoded, y_temp_encoded)
importances = rf_temp.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': X_temp_encoded.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Optional plot
plt.figure(figsize=(10,5))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'],color='pink',edgecolor='black')
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance (RandomForest)")
plt.show()



le = LabelEncoder()
y_encoded = le.fit_transform(df["Stream"])

X = df.drop(columns=["Name", "Phone", "EmailID", "Stream","City"])
y = df["Stream"]

# Ensure only numeric features are used for correlation
numeric_X = X.select_dtypes(include=['number'])

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_X.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap", fontsize=14)
plt.show()



# STEP 6: Train/Test Split

X = df.drop(columns=["Name", "Phone", "EmailID", "Stream","Age","Location","City","Gender","JEE","CUET"])
X = pd.get_dummies(X, drop_first=True)

#  Handle missing values (fill with column mean)
X = X.fillna(X.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Step 7 Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "SVM (RBF Kernel)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(n_estimators=400, random_state=42))
    ])
}


accuracies = {}
predictions = {}

# 🔹 Train & evaluate
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    predictions[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("-" * 60)
    

# Step 8 Confusion Matrix Heatmaps
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()



# STEP 9: Auto-pick Best Model & Save
best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
best_model.fit(X, y)  # Retrain on full dataset

print(f"\n Best Model Selected: {best_model_name} (Accuracy: {accuracies[best_model_name] * 100:.2f}%)")

# Save model & label encoder
joblib.dump(best_model, "model.pkl")
joblib.dump(le, "label_encoder.pkl")
print(" Model & encoder saved successfully.")



# -------------------- Import Libraries --------------------
import streamlit as st
import pandas as pd
import joblib
import os

# -------------------- Load Model --------------------
m1 = joblib.load("model.pkl")   # make sure your trained model file is here

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="Student Stream Predictor", layout="centered")

st.title("🎓 Student Stream Prediction App")
st.write("Enter your details below to predict your suitable stream:")

# -------------------- User Inputs --------------------

# Student basic details
st.subheader("📝 Student Information")
name = st.text_input("Name")
age = st.number_input("Age", min_value=18, max_value=30, step=1)

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    grade_10 = st.number_input("Grade 10 Marks (%)", min_value=0, max_value=100, step=1)
    jee = st.number_input("JEE (out of 300)", min_value=0, max_value=300, step=1)

with col2:
    city = st.text_input("City")
    grade_12 = st.number_input("Grade 12 Marks (%)", min_value=0, max_value=100, step=1)
    cuet = st.number_input("CUET (out of 800)[Optional]", min_value=0, max_value=800, step=1)


# -------------------- Button Style --------------------
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color:#ffe6e6;
        color:#ff4d4d;
        font-weight:600;
        border:2px solid #ffcccc;
        border-radius:12px;
        padding:8px 20px;
    }
    div.stButton > button:first-child:hover {
        background-color:#ffcccc;
        color:#800000;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- Prediction --------------------
if st.button("🔮 Predict Stream"):
    # Step 1: Create DataFrame with raw inputs
    input_data = pd.DataFrame([[
        grade_10, grade_12, jee, cuet
    ]], columns=["Grade_10", "Grade_12", "JEE", "CUET"])

        # Step 2: Apply SAME feature engineering as training
    input_data["JEE_100"] = input_data["JEE"] / 3.0

    # Handle CUET as optional
    if cuet > 0:  # if CUET is entered
        input_data["CUET_100"] = input_data["CUET"] / 8.0
        input_data["Entrance_Avg"] = (input_data["JEE_100"] + input_data["CUET_100"]) / 2
    else:  # no CUET given → only JEE
        input_data["CUET_100"] = 0
        input_data["Entrance_Avg"] = input_data["JEE_100"]

    input_data["Acad_Avg"] = (input_data["Grade_10"] + input_data["Grade_12"]) / 2
    input_data["Combined_Score"] = 0.5 * input_data["Acad_Avg"] + 0.5 * input_data["Entrance_Avg"]

    # Step 3: Keep only trained features
    input_data = input_data[[
        "Grade_10", "Grade_12", "JEE_100", "CUET_100",
        "Acad_Avg", "Entrance_Avg", "Combined_Score"
    ]]


    # Step 3: Keep only trained features
    input_data = input_data[["Grade_10", "Grade_12", "JEE_100", "CUET_100", "Acad_Avg", "Entrance_Avg", "Combined_Score"]]

    # Step 4: Predict
    pred_stream = m1.predict(input_data)[0]

    # Styled output
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; margin-top:20px;">
            <div style="
                background: radial-gradient(circle, #7c3aed, #5b21b6, #4c1d95);
                color: white;
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                padding: 40px;
                border-radius: 50%;
                width: 220px;
                height: 220px;
                display: flex;
                justify-content: center;
                align-items: center;
                box-shadow: 0 0 25px rgba(124,58,237,1), 0 0 40px rgba(167,139,250,1);
            ">
                🎓 {pred_stream}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Extract engineered values
    JEE_100 = input_data["JEE_100"].iloc[0]
    CUET_100 = input_data["CUET_100"].iloc[0]
    Acad_Avg = input_data["Acad_Avg"].iloc[0]
    Entrance_Avg = input_data["Entrance_Avg"].iloc[0]
    Combined_Score = input_data["Combined_Score"].iloc[0]

    # Save input + prediction
    save_data = pd.DataFrame([[
        name, age, grade_10, grade_12, JEE_100, CUET_100, Acad_Avg, Entrance_Avg, Combined_Score, gender, city, pred_stream
    ]], columns=[
        "Name", "Age", "Grade_10", "Grade_12", "JEE_100", "CUET_100",
        "Acad_Avg", "Entrance_Avg", "Combined_Score", "Gender", "City", "Predicted_Stream"
    ])

    if os.path.exists("predictions.csv"):
        save_data.to_csv("predictions.csv", mode="a", header=False, index=False)
    else:
        save_data.to_csv("predictions.csv", index=False)

    st.info("📂 Data saved successfully!")
