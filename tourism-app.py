import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Tourism Rating Predict App

This app predicts the **Rating for place**!

Data obtained from kaagle [data tourism](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination) .
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
# else:
#     def user_input_features():
#         location = st.sidebar.selectbox('Location',('Semarang','Jawa Tengah'))
#         place_name = st.sidebar.selectbox('Place Name',('Candi Ratu Boko'))
#         description = st.sidebar.slider('Description', (''))
#         category = st.sidebar.slider('Category', ('Budaya','Bahari'))
#         city = st.sidebar.slider('City', ('Jakarta','Semarang'))
#         data = {'location': location,
#                 'place_name': place_name,
#                 'description': description,
#                 'category': category,
#                 'city': city}
#         features = pd.DataFrame(data, index=[0])
#         return features
    # input_df = user_input_features()

# Load trained model
load_clf = pickle.load(open('tourism_rating.pkl', 'rb'))

# Load data
data = pd.read_csv('tourism_rating_cleaned.csv')

data.fillna(value="0", inplace=True)


# Display uploaded data
st.subheader('Uploaded Data')
st.write(data)

# tujuan dari pemodelan adalah untuk melakukan prediksi nilai 'Place_Ratings' yang 
# merupakan variabel numerik atau kontinu. Linear Regression sering digunakan 
# untuk memodelkan hubungan linier antara variabel independen (fitur) dan variabel 
# dependen (target) dengan asumsi hubungan tersebut dapat dijelaskan melalui suatu garis lurus.


features = ['User_Id', 'Place_Id', 'Time_Minutes','Price']  # Sesuaikan dengan fitur yang ingin digunakan # Adjust features accordingly
# Prepare data for prediction (select only relevant features)
data_for_prediction = data[features]
# .iloc[[0]]

# Perform prediction using the loaded model
prediction = load_clf.predict(data_for_prediction)

# Display prediction
st.subheader('Prediction')
st.write(prediction)
