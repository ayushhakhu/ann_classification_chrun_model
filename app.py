import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.models import load_model
import pickle



# Load trained model
model = load_model('model.h5') 


# load geoencoder
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

# load labelencoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)



## streamlit app
st.title('Customer Churn Prediction')

## user input
credit_score = st.number_input('Credit Score', value=600)
geography = st.selectbox('Geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18,92)
tenure = st.number_input('Tenure', value=3)
balance = st.number_input('Balance', value=100000)
num_of_products = st.number_input('Number of Products', value=2)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', value=60000)

#Prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': age,
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]}

# one hot encode geography
geo_encoded_data = one_hot_encoder.transform([input_data['Geography']]).toarray()
geo_encoded_data = pd.DataFrame(geo_encoded_data, columns=one_hot_encoder.get_feature_names_out(['Geography']))

# combine data
combined_data = pd.concat([pd.DataFrame(input_data), geo_encoded_data], axis=1)
combined_data = combined_data.drop(columns=['Geography'])

combined_data = scaler.transform(combined_data)

# make prediction
prediction = model.predict(combined_data)
prediction_probability = prediction[0][0]

# display prediction
st.write(f"Prediction probability: {prediction_probability:.4f}")
st.write(f"Predicted class: {'Customer will churn' if prediction_probability > 0.5 else 'Customer will stay'}")
