'''
Deep learining for first arrival picking
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
from keras.models import Model,load_model
import scipy.io as sio
import mysub
import numpy as np
import sys

# load in data
#test_left,test_right,test_true = mysub.load_data(mypath='input_data/data_real_support.mat',shuffle='False')
test_left,test_right,test_true = mysub.load_data(mypath='input_data/data_test_support.mat',shuffle='False')

num=test_left.shape[0]
#num=1000
test_left=test_left[0:num]
test_right=test_right[0:num]
test_true=test_true[:,0:num]

# reshape for keras input (to be 4D)
test_left = test_left.reshape(test_left.shape[0], test_left.shape[1], 1)
test_right = test_right.reshape(test_right.shape[0], test_right.shape[1], 1)
test_true = test_true.reshape(test_true.shape[1], )

print('Test data shape:')
print(test_left.shape)
print(test_true.shape)

def my_loss(y, preds, margin=0.2):
        # explicitly cast the true class label data type to the predicted
        # class label data type (otherwise we run the risk of having two
        # separate data types, causing TensorFlow to error out)
        y = tf.cast(y, preds.dtype)
        # calculate the contrastive loss between the true labels and
        # the predicted labels
        squaredPreds = K.square(1 - preds)
        squaredMargin = K.square(K.maximum(preds - margin, 0))
        loss = K.mean( y * squaredPreds + (1 - y) * squaredMargin)
        #loss = K.mean((1 - y) * squaredPreds + y * squaredMargin)
        #loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
        # return the computed contrastive loss to the calling function
        return loss

# load model
#model = load_model('models/model_epoch050.h5')
model = load_model('models/model_epoch100.h5', custom_objects={'my_loss': my_loss})

# predict
test_pred = model.predict([test_left, test_right],batch_size=256)

test_true=np.array(test_true,dtype=float)
test_pred=np.array(test_pred,dtype=float)

# output
output_folder = "output"
if not os.path.exists(output_folder):
      os.mkdir(output_folder)
filename="%s/predict.mat" %(output_folder)
# save
sio.savemat(filename,{'test_true':test_true, 'test_pred':test_pred})
#sio.savemat(filename,{'test_left':test_left,'test_right':test_right,'test_true':test_true, 'test_pred':test_pred})
