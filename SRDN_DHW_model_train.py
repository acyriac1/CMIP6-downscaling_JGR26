import sys
import os
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
os.environ["NCCL_DEBUG"] = "WARN"

import time
import socket

import tensorflow as tf
import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, MaxPooling2D, Dropout, Reshape, Dense
from tensorflow.keras.optimizers import Adam,SGD,Nadam
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

import math
import numpy as np
import xarray as xr
import pickle
from netCDF4 import Dataset


print(tf.version)

np.set_printoptions(threshold=np.inf)  # print all array elements

#----------------------------------------------------------------------------------------------
#
# Horovod: initialize Horovod.
#
#----------------------------------------------------------------------------------------------

hvd.init()

print ('***hvd.size ', hvd.size(),' hvd.rank', hvd.rank(), 'hvd.local_rank() ', hvd.local_rank())

# Horovod: pin GPU to be used to process local rank (one GPU per process)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(' gpus = ', gpus)
if hvd.local_rank() == 0:
    print("Socket and len gpus = ",socket.gethostname(), len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#----------------------------------------------------------------------------------------------
#
# Build downscaling model using TensorFlow and Horovod - this version uses convolutuions
#
#----------------------------------------------------------------------------------------------

def SRDCN_DHW_v1(numHiddenUnits, numResponses, numFeatures, numLats, numLongs, shrink):

    concat_axis = -1
    activation = 'relu'
    shrink = shrink
    reg_val  = 0.000000001

    inputs  = tf.keras.layers.Input(shape = (int(numLats/shrink), int(numLongs/shrink), numFeatures) )
    
    # Downscale model using conv2dTranspose layers
    
    x = layers.Conv2DTranspose(numHiddenUnits, (7, 7), strides=2, activation="relu", padding="same",
                kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(inputs)
                
    x = layers.Conv2DTranspose(numHiddenUnits, (7, 7), strides=2, activation="relu", padding="same",
                kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(x)
               
    x = layers.Conv2DTranspose(numHiddenUnits, (7, 7), strides=2, activation="relu", padding="same",
                kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(x)
     
    x = layers.Conv2D(numResponses, (1, 1), activation="linear", padding="same",
        kernel_regularizer=l2(reg_val), activity_regularizer=l2(reg_val), bias_regularizer=l2(reg_val))(x) 

    model = tf.keras.models.Model(inputs=inputs, outputs=x)

# Horovod: adjust learning rate based on number of GPUs.
    opt = tf.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999,
               epsilon=None, decay=0.0, amsgrad=False)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(loss='mse', optimizer=opt, metrics=['msle','mae'], experimental_run_tf_function=False)
    
    print( model.summary() ) if hvd.rank() == 0 else None

    return model

# 

t0 = [0]*hvd.size()
t0[hvd.rank()] = time.time()

####--------------------- Set total number of images, training epoch size, test data size
#
#  - must be Scalable/divisible to multi nodes to ensure load balance

Total_images = 13149 # 36 years of daily data [1979-2014]📍️
Epoch_size = 10560 #approx. 80% and divisible by 80GPUs
Test_size  = Total_images - Epoch_size 

# Batch size - aim to fill GPU memory to achieve best computational performance
batch_size = 16

# Set key parameters for Conv2DLSTM
numHiddenUnits = 64
numResponses = 1
numFeatures  = 1
shrink = 8

numLats        = 512 
numLongs       = 512

if hvd.rank() == 0:
    print ('*** rank = ', hvd.rank(),' Epoch size = ', Epoch_size)
    print ('*** rank = ', hvd.rank(),' Test_size = ', Test_size)
    print ('*** rank = ', hvd.rank(),' Batch size = ', batch_size)
    print ('*** rank = ', hvd.rank(),' numHiddenUnits = ', numHiddenUnits)
    print ('*** rank = ', hvd.rank(),' numResponses= ', numResponses)
    print ('*** rank = ', hvd.rank(),' numFeatures = ', numFeatures)
    print ('*** rank = ', hvd.rank(),' numLats = ', numLats)
    print ('*** rank = ', hvd.rank(),' numLongs = ', numLongs)
#----------------------------------------------------------------------------------------------
#
# 	Open the input data files using xarray
#
#----------------------------------------------------------------------------------------------


###------------------ Use the normalized data
gbr_past = xr.open_dataset("DHW_norm_GBR.nc")
lon = gbr_past.lon.data
lat = gbr_past.lat.data

time_past = gbr_past.time.data
dhw_data = gbr_past.DHWnorm.data #saved normalized data

#-----------------------------------------
# Compute and print file open time
elapsed_time = time.time() - t0[hvd.rank()]
print ('*** rank = ', hvd.rank(),' Dataset Intitialise Elapsed Time (sec) = ', elapsed_time)

#----------------------------------------------------------------------------------------------
#
# Horovod: Split the test data across multiple processors   
#
#----------------------------------------------------------------------------------------------

istart = int(hvd.rank()*Epoch_size/hvd.size())
istop  = int((hvd.rank()+1)*Epoch_size/hvd.size())

i_test_start = int(hvd.rank()*Test_size/hvd.size()+Epoch_size) + 1
i_test_stop  = int((hvd.rank()+1)*Test_size/hvd.size()+Epoch_size)

if i_test_stop >= Total_images:
  i_test_stop = Total_images - 1

print ( '*** rank = ', hvd.rank(),' istart = ', istart, ' istop = ', istop)
print ( '*** rank = ', hvd.rank(),' i_test_start = ', i_test_start, ' i_test_stop = ', i_test_stop)

t0[hvd.rank()] = time.time()
#----------------------------------------------------------------------------------------------
#
# Horovod: Read in the train and test data across multiple processors
#
#----------------------------------------------------------------------------------------------

#  Downscale experiments start with low resolution data

#add a tensor to make the data 4D
x_tr_n = np.expand_dims(sst_data[istart:istop,:,:], axis=3)
x_te_n = np.expand_dims(sst_data[i_test_start:i_test_stop,:,:], axis=3)

x_train = tf.keras.layers.AveragePooling2D(
          pool_size=(shrink, shrink), strides=None, padding='same', data_format=None)(x_tr_n)
x_test = tf.keras.layers.AveragePooling2D(
          pool_size=(shrink, shrink), strides=None, padding='same', data_format=None)(x_te_n)

#  Target is the high resolution data
y_train = x_tr_n
y_test = x_te_n

print(' Training data shapes = ',x_train.shape, y_train.shape)
print(' Test data shapes = ',x_test.shape, y_test.shape)

print ('*** rank = ', hvd.rank(),' Y data')

print ('*** rank = ', hvd.rank(),' y_train', sys.getsizeof(y_train))
print ('*** rank = ', hvd.rank(),' y_test', sys.getsizeof(y_test))


print ('*** rank = ', hvd.rank(),' X data')

print ('*** rank = ', hvd.rank(),' x_train', sys.getsizeof(x_train))
print ('*** rank = ', hvd.rank(),' x_test', sys.getsizeof(x_test))

#   Elapsed time for read operation

elapsed_time = time.time() - t0[hvd.rank()]
print ('*** rank = ', hvd.rank(),' Dataset Read Elapsed Time (sec) = ', elapsed_time)

# Determine how many batches are there in train and test sets

train_batches = len(x_train) // batch_size
test_batches = len(x_test) // batch_size

print ('*** rank = ', hvd.rank(),' train_batches', train_batches)
print ('*** rank = ', hvd.rank(),' test_batches', test_batches)
#----------------------------------------------------------------------------------------------
#
# Horovod: create callbacks required for horovod model run
#
#----------------------------------------------------------------------------------------------

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.

    hvd.callbacks.MetricAverageCallback(),

    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),

    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, monitor='val_loss', mode='min', patience=10, min_lr=0.00001, verbose=1),
]
#----------------------------------------------------------------------------------------------
#
# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#
#----------------------------------------------------------------------------------------------

if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoints/checkpoint-{epoch}.h5', monitor='val_loss', save_best_only=True))

#----------------------------------------------------------------------------------------------
#
# 	Build the model
#
#----------------------------------------------------------------------------------------------

model = SRDCN_DHW_v1(numHiddenUnits, numResponses, numFeatures, numLats, numLongs, shrink)

#----------------------------------------------------------------------------------------------
#
# 	Train the model
#
#----------------------------------------------------------------------------------------------

# Setup timer for training step

t0[hvd.rank()] = time.time()

# Add a barrier to sync all processes before starting training

hvd.allreduce([0], name="Barrier")
print ('*** rank = ', hvd.rank(),' Train model')

history = model.fit(x_train, y_train, callbacks=callbacks, epochs=200, verbose=2, 
                      validation_data = (x_test, y_test)) 

# Elapsed time for training operation
elapsed_time = time.time() - t0[hvd.rank()]
print ('*** rank = ', hvd.rank(),' Total Training Elapsed Time (sec) = ', elapsed_time)

#for history in histories:
with open('./trainHistoryDict_DHW_history_highres_{}_GPU'.format(hvd.size()), 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
if hvd.rank() == 0:
   model.save('./downscale_DHW_norm_GBR.h5')
