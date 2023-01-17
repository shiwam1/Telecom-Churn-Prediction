#https://docs.streamlit.io/library/get-started/create-an-app

import streamlit as st
import numpy as np
import pandas as pd

st.title('Telecom Churn Prediction')

st.sidebar.header('User Input Parameters')




def user_input_features():
    gen = st.sidebar.selectbox("Gender",("Male","Female"))
    ss = st.sidebar.selectbox("SeniorCitizen",(0,1))
    dep = st.sidebar.selectbox("Dependents",("Yes","No"))
    ten = st.slider("Tenure",min_value=0,max_value=75,step=1)
    
    isr = st.sidebar.selectbox('InternetService',('DSL', 'Fiber optic' ,'No'))
    osr = st.sidebar.selectbox('OnlineSecurity',('No', 'Yes',"No internet service"))
    ob = st.sidebar.selectbox('OnlineBackup',('No', 'Yes',"No internet service"))
    dp = st.sidebar.selectbox('DeviceProtection',('No', 'Yes',"No internet service"))

    ts = st.sidebar.selectbox('TechSupport',('No', 'Yes',"No internet service"))
    
    cr = st.sidebar.selectbox('Contract',('Month-to-month', 'One year' ,'Two year'))
    
    pb = st.sidebar.selectbox('PaperlessBilling',('Yes','No'))
    pm = st.sidebar.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'))
    
    
    mc = st.sidebar.number_input("Insert the MonthlyCharges",min_value=10,max_value=1000,step=1)
    tc = st.sidebar.number_input("Insert TotalCharges",min_value=10,max_value=1000,step=1)
    
    
    new = {"gender":gen,
         'SeniorCitizen': ss,
         'Dependents':dep,
         'tenure': ten,
         'InternetService': isr,
         'OnlineSecurity': osr,
         'OnlineBackup': ob,
         'DeviceProtection': dp,
         'TechSupport': ts,
         'Contract': cr,
         'PaperlessBilling': pb,
         'PaymentMethod': pm,
         'MonthlyCharges': mc,
         'TotalCharges': tc,
            }
    features = pd.DataFrame(new,index = [0])
    return features 
    
df = user_input_features()
st.write(df)
 



import pickle
    
with open("Final_model.pkl",mode="rb") as f:
    model = pickle.load(f)


    
st.write("Model Loaded")



result = model.predict(df)

st.subheader('Predicted Result')

if result[0]=="No":
    st.write("Customer will not Churn")
    
else:
    st.write("Customer will Churn")










    
    
    



