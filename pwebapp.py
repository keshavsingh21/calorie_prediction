# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 06:10:38 2025

@author: Acer
"""

import numpy as np
import pickle
import streamlit as st

loaded_m = pickle.load(open('C:/Users/Acer/Downloads/trained_model.sav', 'rb'))


# Use the same encoding as during training!


def calories_prrediction(input_data):
    # input_data: [Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]
    gender = input_data[0]
    # Encode gender
    if gender == 'male':
        gender_encoded = 0
    elif gender == 'female':
        gender_encoded = 1
    else:
        raise ValueError("Gender must be 'male' or 'female'")
    # Replace gender string with encoded value
    numeric_data = [gender_encoded] + input_data[1:]
    input_data_np = np.array(numeric_data, dtype=np.float32).reshape(1, -1)
    prediction = loaded_m.predict(input_data_np)
    return prediction


def main():
    
    
    st.title('calories prediction web app')
    
    Gender = st.text_input('enter the gender')
    Age = st.text_input('enter age')
    Height = st.text_input('enter height')
    Weight = st.text_input('enter weight')
    Duration = st.text_input('enter workout duration')
    Heart_Rate = st.text_input('enter heart rate')
    Body_Temp = st.text_input('enter body temp')
    
    
    diagnosis = ''
    
    if st.button('total calories value'):
       diagnosis = calories_prrediction([Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp])

    st.success(diagnosis)

if __name__ == '__main__':  
    main()      
        
        
        
        
        
        