import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np
model = pickle.load(open('multiple.sav', 'rb'))

st.title('Multiple Linear Regression')
# FUNCTION
def user_report():
    r_n_d = st.number_input("R&D Spend:", min_value=0.0, max_value=170000.0, value=100000.0, step=2.0)
    admin = st.number_input("Administration:", min_value=0.0, max_value=160000.0, value=100000.0, step=2.0)
    market = st.number_input("Marketing Spend:", min_value=0.0, max_value=480000.0, value=100000.0, step=2.0)
    state = st.number_input("State:", min_value=0, max_value=2, value=0, step=1)
    
    user_report_data = {
        'R&D Spend':r_n_d,
        'Administration':admin,
        'Marketing Spend':market,
        'State':state
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()
if st.button("Submit"):
    st.header("Capital Data")
    st.write(user_data)
    
    salary = model.predict(user_data)
    st.subheader('Profit Prediction')
    st.subheader('$'+str(np.round(salary[0], 2)))


#state = st.number_input("State:", min_value=0, max_value=2, value=0, step=2)