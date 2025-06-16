'''
Predict for the vailidative data
'''

import os
import keras
from keras.models import Model,load_model
import scipy.io as sio
import mysub
import numpy as np
import sys

#load in data
x_test,y_test=mysub.load_dataCNN(mypath='input_data/data_test.mat',shuffle='False')

#reshape for keras input
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)
y_test=y_test.reshape(y_test.shape[1],)
print("test data'shape:")
print(x_test.shape)
print(y_test.shape)

#load model
model=load_model("models/model_epoch100.h5")
model.summary()
#predict
y_pred=model.predict([x_test])

#output
output_folder="output"
if not os.path.exists(output_folder):
      os.mkdir(output_folder)
filename="%s/predict.mat" %(output_folder)

# save
sio.savemat(filename,{'x_test':x_test,'test_true':y_test,'test_pred':y_pred})

