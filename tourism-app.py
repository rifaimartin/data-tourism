import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
load_clf = pickle.load(open('tourism_rating.pkl', 'rb'))

st.write("""
# Tourism Rating Predict App

This app predicts the **Rating for place**!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

# Load data
data = pd.read_csv('tourism_rating_cleaned.csv')
data.fillna(value="0", inplace=True)

# Load additional data
data2 = pd.read_csv('sumatera.csv')
data2.fillna(value="0", inplace=True)

# Display uploaded data
st.subheader('Uploaded Data')
st.write(data)

features = ['User_Id', 'Place_Id', 'Time_Minutes', 'Price']  # Sesuaikan dengan fitur yang ingin digunakan

# Prepare data for prediction
data_for_prediction = data[features]
prediction = load_clf.predict(data_for_prediction)

data_for_prediction2 = data2[features]
prediction2 = load_clf.predict(data_for_prediction2)

# Sort predictions to find the highest values
sorted_indices = np.argsort(prediction)[::-1]
sorted_indices2 = np.argsort(prediction2)[::-1]

# Display prediction with record index
st.subheader('Top Predictions Jawa Barat')
for idx, pred in zip(sorted_indices[:10], prediction[sorted_indices[:10]]):
    st.write(f"Record index: {idx}, Prediction: {pred}")

st.subheader('Top Predictions Sumatra')
for idx, pred in zip(sorted_indices2[:10], prediction2[sorted_indices2[:10]]):
    st.write(f"Record index: {idx}, Prediction: {pred}")