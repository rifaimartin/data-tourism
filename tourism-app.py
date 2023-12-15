import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Tourism Rating Predict App

This app predicts the **Rating for place**!

Data obtained from kaagle [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
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
#     input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
tourism_raw = pd.read_csv('tourism_rating_cleaned.csv')
tourism = tourism_raw.drop(columns=['Age'])
tourism = tourism_raw.drop(columns=['Category'])
tourism = tourism_raw.drop(columns=['City'])
tourism = tourism_raw.drop(columns=['Coordinate'])
tourism = tourism_raw.drop(columns=['Description'])

# df = tourism_raw 
df = tourism

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
# encode = ['sex','island']
# for col in encode:
#     dummy = pd.get_dummies(df[col], prefix=col)
#     df = pd.concat([df,dummy], axis=1)
#     del df[col]
# df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
# else:
#     st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
#     st.write(df)

# Memuat model yang benar (misalnya RandomForestClassifier)
load_clf = pickle.load(open('tourism_rating.pkl', 'rb'))

# Mengecek kembali fitur-fitur yang akan digunakan untuk prediksi
features = ['User_Id', 'Location', 'Place_Id', 'Time_Minutes']  # Sesuaikan dengan fitur yang digunakan saat melatih model

# Memastikan dataframe yang digunakan sesuai dengan fitur yang diharapkan
df = pd.read_csv('tourism_rating_cleaned.csv')
df = df[features]  # Mengambil hanya fitur-fitur yang digunakan saat pelatihan

# Menampilkan fitur-fitur yang akan digunakan untuk prediksi
st.subheader('User Input features')
st.write(df)

categorical_columns = ['Location', 'Place_Name', 'Description', 'Category', 'City']

# Lakukan One-Hot Encoding pada kolom-kolom kategorikal
for col in categorical_columns:
    encoded_cols = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, encoded_cols], axis=1)
    df.drop(col, axis=1, inplace=True)

# Melakukan prediksi dengan model yang telah dimuat
prediction = load_clf.predict(df)

# Menampilkan hasil prediksi
st.subheader('Prediction')
st.write(prediction)
