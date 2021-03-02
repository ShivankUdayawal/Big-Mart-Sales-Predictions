# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle



# Loading model to compare the results
model = pickle.load(open('new_model.pkl','rb'))
print(model.predict([[249.8092,1,0,1,0,0]]))