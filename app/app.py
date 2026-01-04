import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(__file__) 
MODEL_PATH = os.path.join(BASE_DIR, "titanic_pipeline.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

st.title("Titanic Survival Predictor")


 

col1,col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 0, 100)
    fare = st.number_input("Fare", 0.0, 1000.0)
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])
    sibsp = st.number_input("Siblings/Spouses aboard", 0, 10)
    parch = st.number_input("Parents/Children aboard", 0, 10)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Rare"])
    familysize = sibsp + parch + 1
    isalone = 1 if familysize == 1 else 0
    if st.button("Predict"):
         if age <= 0 or fare <= 0:
            st.warning("Age and Fare must be greater than 0")
         else:
            sample = pd.DataFrame([{
                "age": age, "fare": fare, "FamilySize": familysize, "pclass": pclass,
                "sex": sex, "embarked": embarked, "sibsp": sibsp, "parch": parch,
                "Title": title, "IsAlone": isalone
            }])
            pred = pipeline.predict(sample)[0]
            proba = pipeline.predict_proba(sample)[0][1]
            st.success(f"Prediction: {'Survived' if pred==1 else 'Did not survive'}")
            st.info(f"Probability: {proba:.2f}")

with col2:
    st.header("Upload a CSV file for batch prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if st.button("Predict Batch", key="batch_predict"):
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # Validate columns
            required_columns = ["age","fare","pclass","sex","embarked","sibsp","parch","Title"]

            df["FamilySize"] = df["sibsp"] + df["parch"] + 1
            df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
            df["Title"] = df["name"].str.extract(r' ([A-Za-z]+)\.', expand=False)

            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain columns: {required_columns}")
            else:
                predictions = pipeline.predict(df)
                probabilities = pipeline.predict_proba(df)[:,1]

                df["Prediction"] = ["Survived" if p==1 else "Did not survive" for p in predictions]
                df["Probability"] = probabilities

                st.success("Batch predictions completed!")
                st.dataframe(df)

