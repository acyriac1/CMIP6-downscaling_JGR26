#---------------------------------------------------
import os
import random
os.environ["TF_DISABLE_NVTX_RANGES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"

import time
import tensorflow as tf
import numpy as np
import sys
import xarray as xr

import horovod.tensorflow.keras as hvd
from tensorflow.keras import layers
import time
import socket
import math
import cupy as cp

from skimage.metrics import structural_similarity as ssim_ski


#------ initiate horovod

hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)

gpus = tf.config.experimental.list_physical_devices('GPU')
if hvd.local_rank() == 0:
	print("Socket and len gpus = ",socket.gethostname(), len(gpus))
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')   

###---Normalized input data

gbr_past = xr.open_dataset("DHW_norm_GBR.nc")
time_past = gbr_past.time.data
dhw_data = gbr_past.DHWnorm.data #saved normalized data
dhw_r = gbr_past.DHWnorm

#-------------------------------------------------------------------------
Total_images = len(time_past)

if hvd.rank() == 0:
	#nworkers = int(hvd.size())
	istart = int(hvd.rank()*Total_images/hvd.size())
	istop  = int((hvd.rank()+1)*Total_images/hvd.size())
else:
	istart = int(hvd.rank()*Total_images/hvd.size())
	istop  = int((hvd.rank()+1)*Total_images/hvd.size())
if istop >= Total_images:
	istop = Total_images - 1
print ( '*** rank = ', hvd.rank(),' istart = ', istart, ' istop = ', istop)
#-------
#--------------------------- expand the dimensions of the data

x_norm = np.expand_dims(dhw_data[istart:istop,:,:], axis=3)


#------------------shrink the input data 
shrink = 8
x = tf.keras.layers.AveragePooling2D(
          pool_size=(shrink, shrink), strides=None, padding='same', data_format=None)(x_norm)

#--------------------- Load the trained model and predict 

model = tf.keras.models.load_model('downscale_dhw_srdn.h5', compile=False)
if hvd.rank() == 0:
    model.summary()
model_dhw_norm = model.predict(x)

print('*** rank = ', hvd.rank(),'predict shape:',model_dhw_norm.shape)

dhw_model = xr.zeros_like(dhw_r[istart:istop,:,:])
dhw_model[:,:,:] = model_dhw_norm[:,:,:,0]
#-------------------------

#---save the predicted data
dhw_model.to_netcdf('DHW_model_{}.nc'.format(hvd.rank()))

#-------------------------
print('*** rank = ', hvd.rank(),'prediction completed')
