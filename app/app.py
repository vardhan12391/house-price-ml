import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data
df = pd.read_csv('../data/train.csv')

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'OverallQual']
target = 'SalePrice'

df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Train model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# UI
st.title("House Price Predictor")

area = st.number_input("Living Area (sq ft)", 500, 5000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 5)
quality = st.number_input("Overall Quality (1-10)", 1, 10)

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, quality]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Estimated Price: ₹{int(prediction[0])}")