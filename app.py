import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("clf.joblib")

clf = load_model()
x = st.slider("Feature X", -5.0, 5.0, 0.0)
y = st.slider("Feature Y", -5.0, 5.0, 0.0)
proba = clf.predict_proba([[x, y]])[0,1]
st.metric("P(class=1)", f"{proba:.2%}")
