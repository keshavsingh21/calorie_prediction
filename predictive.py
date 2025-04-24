# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
loaded_m = pickle.load(open('C:/Users/Acer/Downloads/trained_model.sav', 'rb'))


input_data =(0,69,190.0,95.0,39.0,110.0,50.8)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = loaded_m.predict(input_data_reshaped)
print(prediction)