import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st

# Load encoders and model
onehot_encoder_geo = pickle.load(open('onehot_encoder_geo.pkl', 'rb'))
label_encoder_gender = pickle.load(open('label_encoder_gender.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = tf.keras.models.load_model('model.h5')

# Streamlit UI
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Base input dataframe
input_df = pd.DataFrame([{
    'CreditScore': credit_score,
    'Gender': gender_encoded,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,  # ✅ fixed spelling
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}])

# One-hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine
input_df = pd.concat([input_df, geo_df], axis=1)


input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# Predict
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

# Output
if prediction_proba > 0.5:
    st.error(f"Customer is likely to churn ({prediction_proba:.2f})")
else:
    st.success(f"Customer is NOT likely to churn ({prediction_proba:.2f})")