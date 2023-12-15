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
df = pd.read_csv('tourism_rating_cleaned.csv')
# tourism = tourism_raw.drop(columns=['Unnamed: 11'])
# df = tourism_raw 

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

# Reads in saved classification model
load_clf = pickle.load(open('tourism_rating.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
# prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['User_Id','Location','Place_Id', 'Time_Minutes'])
st.write(penguins_species[prediction])

# st.subheader('Prediction Probability')
# st.write(prediction_proba)
