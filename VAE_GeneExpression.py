# Variational AutoEncoder

**Author:** Maximiliano Bless<br>
**Date created:** 2022/05/14<br>
**Last modified:** 2022/05/14<br>
**Description:** Convolutional Variational AutoEncoder (VAE) to trained on PCA images from expresion dataset without grouping.
Image dimension 256x256.
Class categories: 4

Convolutional Kernel sizes: 32-64-128.

Latent dimension: 256

**Conclusion**:

**Resumen**:

En este código se entreó el VAE de expresion con los datos individuales.
"""

from google.colab import drive
drive.mount('/content/drive')

"""## Setup"""

!pip3 install visualkeras

import numpy as np
from numpy import array, zeros, newaxis
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import pandas as pd
import datetime, os
import glob
from numba import cuda
import random
import visualkeras
from PIL import Image,ImageDraw,ImageFont

font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 42, encoding="unic")
from tensorflow.keras.layers import InputLayer,Concatenate,Conv2D,BatchNormalization,LeakyReLU,MaxPooling2D,Flatten,Dense

from collections import defaultdict

color_map = defaultdict(dict)
color_map[InputLayer]['fill'] = (255,200,100,255)
color_map[Concatenate]['fill'] = 'gray'
color_map[Conv2D]['fill'] = (255,101,101,255)
color_map[BatchNormalization]['fill'] = (0,255,204,255)
color_map[LeakyReLU]['fill'] = (101,175,255,255)
color_map[MaxPooling2D]['fill'] = (0,0,51,255)
color_map[Flatten]['fill'] = (150,0,51,255)
color_map[Dense]['fill'] = (101,255,101,255)

#visualkeras.layered_view(decoder, legend=True, font=font,spacing=10, scale_xy=10, scale_z=1, max_z=100,color_map=color_map)

"""## Create a sampling layer"""

class Sampler(layers.Layer):
    def call(self, z_mean,z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""## Build the encoder

### Build the encoder - **Transfer learning**
"""

latent_dim = 256
img_dim = 256
down_ratio = 4
kernel_sizes=[32,64,128]
use_bias=True

RNA_encSub1_inputs = keras.Input(shape=(img_dim, img_dim, 1), name="x_RNA")

RNA_encSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias, name="RNA_encSub1_conv2D.1.0")(RNA_encSub1_inputs)
RNA_encSub1 = layers.BatchNormalization(name="RNA_encSub1_conv2D.1.1")(RNA_encSub1)
RNA_encSub1 = layers.LeakyReLU(name="RNA_encSub1_conv2D.1.2")(RNA_encSub1)

RNA_encSub1 = layers.MaxPool2D(pool_size=down_ratio, name="RNA_encSub1_MaxPool.1")(RNA_encSub1) # MaxPool -1-

RNA_encSub1 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias, name="RNA_encSub1_conv2D.2.0")(RNA_encSub1)
RNA_encSub1 = layers.BatchNormalization(name="RNA_encSub1_conv2D.2.1")(RNA_encSub1)
RNA_encSub1 = layers.LeakyReLU(name="RNA_encSub1_conv2D.2.2")(RNA_encSub1)

RNA_encSub1_outputs = layers.MaxPool2D(pool_size=down_ratio,name="RNA_encSub1_MaxPool.3")(RNA_encSub1) # MaxPool -2-

# Genero submodelo del encoder que utilizaré para transfer learning
RNA_encoder_sub1 = keras.Model(RNA_encSub1_inputs, RNA_encSub1_outputs, name="RNA_encoder_sub1")
print(RNA_encoder_sub1.summary())
keras.utils.plot_model(RNA_encoder_sub1, show_shapes = True)

"""### Build the encoder with all the subparts - **latent space**"""

RNA_inputs = keras.Input(shape=(256, 256, 1),name="x_RNA_input")
encSub2_inputs = RNA_encoder_sub1(RNA_inputs)

encSub2 = layers.Conv2D(kernel_sizes[2], down_ratio, strides=1, padding="same",use_bias=False, name="encSub2_conv2D.1.0")(encSub2_inputs)
encSub2 = layers.BatchNormalization(name="encSub2_conv2D.1.1")(encSub2)
encSub2 = layers.LeakyReLU(name="encSub2_conv2D.1.2")(encSub2)

encSub2 = layers.MaxPool2D(pool_size=down_ratio,name="encSub2_MaxPool.1")(encSub2) # MaxPool -3-

encSub2 = layers.Flatten(name="encSub2_Flatten")(encSub2)

z_mean = layers.Dense(latent_dim, name="z_mean")(encSub2)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encSub2)

encoder = keras.Model(RNA_inputs, [z_mean,z_log_var], name="encoder")

print(encoder.summary())
keras.utils.plot_model(encoder, show_shapes = True)

#CREO MAPA PARA EL ENCODER

latent_dim = 256
img_dim = 256
down_ratio = 4
kernel_sizes=[32,64,128]
use_bias=True

RNA_encSub1_inputs = keras.Input(shape=(img_dim, img_dim, 1), name="x_RNA")

RNA_encSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias, name="RNA_encSub1_conv2D.1.0")(RNA_encSub1_inputs)
RNA_encSub1 = layers.BatchNormalization(name="RNA_encSub1_conv2D.1.1")(RNA_encSub1)
RNA_encSub1 = layers.LeakyReLU(name="RNA_encSub1_conv2D.1.2")(RNA_encSub1)

RNA_encSub1 = layers.MaxPool2D(pool_size=down_ratio, name="RNA_encSub1_MaxPool.1")(RNA_encSub1) # MaxPool -1-

RNA_encSub1 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias, name="RNA_encSub1_conv2D.2.0")(RNA_encSub1)
RNA_encSub1 = layers.BatchNormalization(name="RNA_encSub1_conv2D.2.1")(RNA_encSub1)
RNA_encSub1 = layers.LeakyReLU(name="RNA_encSub1_conv2D.2.2")(RNA_encSub1)

RNA_encSub1_outputs = layers.MaxPool2D(pool_size=down_ratio,name="RNA_encSub1_MaxPool.3")(RNA_encSub1) # MaxPool -2-

encSub2 = layers.Conv2D(kernel_sizes[2], down_ratio, strides=1, padding="same",use_bias=False, name="encSub2_conv2D.1.0")(RNA_encSub1_outputs)
encSub2 = layers.BatchNormalization(name="encSub2_conv2D.1.1")(encSub2)
encSub2 = layers.LeakyReLU(name="encSub2_conv2D.1.2")(encSub2)

encSub2 = layers.MaxPool2D(pool_size=down_ratio,name="encSub2_MaxPool.1")(encSub2) # MaxPool -3-

encSub2 = layers.Flatten(name="encSub2_Flatten")(encSub2)

z_mean = layers.Dense(latent_dim, name="z_mean")(encSub2)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encSub2)

encoder = keras.Model(RNA_encSub1_inputs, [z_mean,z_log_var], name="encoder")

visualkeras.layered_view(encoder, legend=True, font=font,spacing=10,
                         scale_xy=1, scale_z=.1, max_z=1e10,
                         color_map=color_map,
                         draw_volume=False)

"""## Build the decoder

### Build the decoder - **Transfer learning**
"""

# Genero submodelo del decoder que utilizaré para transfer learning
up_ratio = 4
use_bias=True

decSub1_inputs = keras.Input(shape=(16, 16, 64), name="decSub1_inputs")

decSub1 = layers.Conv2DTranspose(kernel_sizes[1], 3, strides=up_ratio, padding="same",use_bias=use_bias,name="decSub1_convTrans2D.1.0")(decSub1_inputs)
decSub1 = layers.BatchNormalization(name="decSub1_convTrans2D.1.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_convTrans2D.1.2")(decSub1)

decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="decSub1_conv2D.2.0")(decSub1)
decSub1 = layers.BatchNormalization(name="decSub1_conv2D.2.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_conv2D.2.2")(decSub1)

decSub1 = layers.Conv2DTranspose(kernel_sizes[0], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="decSub1_convTrans2D.3.0")(decSub1)
decSub1 = layers.BatchNormalization(name="decSub1_convTrans2D.3.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_convTrans2D.3.2")(decSub1)

decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="decSub1_conv2D.4.0")(decSub1)
decSub1 = layers.BatchNormalization(name="decSub1_conv2D.4.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_conv2D.4.2")(decSub1)

decSub1_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same",name="decSub1_output")(decSub1)

decoder_sub1 = keras.Model(decSub1_inputs, decSub1_outputs, name="decoder_sub1")
print(decoder_sub1.summary())
keras.utils.plot_model(decoder_sub1, show_shapes = True)

"""### Build the decoder - **latent space**"""

# Genero el input del encoder (z_sample)

z_sample = keras.Input(shape=(latent_dim,),name="z_sample")

decSub2 = layers.Dense(4 * 4 * kernel_sizes[2], activation="relu",name="decSub2_Dense")(z_sample)

decSub2 = layers.Reshape((4, 4, kernel_sizes[2]),name="decSub2_Reshape")(decSub2)

decSub2 = layers.Conv2DTranspose(kernel_sizes[2], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="decSub2_convTrans2D.1.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_convTrans2D.1.1")(decSub2)
decSub2 = layers.LeakyReLU(name="decSub2_convTrans2D.1.2")(decSub2)

decSub2 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias,name="decSub2_conv2D.2.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_conv2D.2.1")(decSub2)
decSub2_outputs = layers.LeakyReLU(name="decSub2_conv2D.2.2")(decSub2)
RNA_decoder_sub1_output = decoder_sub1(decSub2_outputs)

decoder = keras.Model(z_sample, RNA_decoder_sub1_output, name="decoder")
print(decoder.summary())
keras.utils.plot_model(decoder, show_shapes = True)

#CREO MAPA PARA EL DECODER

z_sample = keras.Input(shape=(latent_dim,),name="z_sample")

decSub2 = layers.Dense(4 * 4 * kernel_sizes[2], activation="relu",name="decSub2_Dense")(z_sample)

decSub2 = layers.Reshape((4, 4, kernel_sizes[2]),name="decSub2_Reshape")(decSub2)

decSub2 = layers.Conv2DTranspose(kernel_sizes[2], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="decSub2_convTrans2D.1.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_convTrans2D.1.1")(decSub2)
decSub2 = layers.LeakyReLU(name="decSub2_convTrans2D.1.2")(decSub2)

decSub2 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias,name="decSub2_conv2D.2.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_conv2D.2.1")(decSub2)
decSub2_outputs = layers.LeakyReLU(name="decSub2_conv2D.2.2")(decSub2)

decSub1 = layers.Conv2DTranspose(kernel_sizes[1], 3, strides=up_ratio, padding="same",use_bias=use_bias,name="decSub1_convTrans2D.1.0")(decSub2_outputs)
decSub1 = layers.BatchNormalization(name="decSub1_convTrans2D.1.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_convTrans2D.1.2")(decSub1)

decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="decSub1_conv2D.2.0")(decSub1)
decSub1 = layers.BatchNormalization(name="decSub1_conv2D.2.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_conv2D.2.2")(decSub1)

decSub1 = layers.Conv2DTranspose(kernel_sizes[0], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="decSub1_convTrans2D.3.0")(decSub1)
decSub1 = layers.BatchNormalization(name="decSub1_convTrans2D.3.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_convTrans2D.3.2")(decSub1)

decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="decSub1_conv2D.4.0")(decSub1)
decSub1 = layers.BatchNormalization(name="decSub1_conv2D.4.1")(decSub1)
decSub1 = layers.LeakyReLU(name="decSub1_conv2D.4.2")(decSub1)

decSub1_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same",name="decSub1_output")(decSub1)
decoder = keras.Model(z_sample, decSub1_outputs, name="decoder")

#visualkeras.layered_view(decoder, legend=True, font=font,spacing=10,
#                         scale_xy=1, scale_z=.1, max_z=1e10,
#                         color_map=color_map,
#                         draw_volume=False)
print(decoder.summary())
keras.utils.plot_model(decoder, show_shapes = True)

visualkeras.layered_view(decoder, legend=True, font=font,spacing=10,
                         scale_xy=1, scale_z=.1, max_z=1e10)

"""## Define the VAE as a `Model` with a custom `train_step`




"""

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    #@tf.function #improves speed without debugging
    def train_step(self, data):
        # with tf.GradientTape() I define tape as a function to be derivated
        with tf.GradientTape() as tape:
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean,z_log_var)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
            #total_loss = reconstruction_loss + kl_loss
        
        # calculate the gradients using our tape and then update the model weights. total_loss es la función y self.trainable_weights son los valores de los weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Compute our own loss metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss_vae": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

"""## Callbacks

Tensorflow callbacks are functions or blocks of code which are executed during a specific instant while training a Deep Learning Model. The following callbacks will be used during the training:

 * Early Stopping

 * Checkpoint

 * Training logger

 * TensorBoard
"""

!mkdir '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/'
!mkdir '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/VAE_V6/'

# Borra carpetas necesarias
#!rm -rf '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/best_weight_model_v2.h5'
!rm -rf "logs"
#!rm -rf '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/training_v2.log'

# Commented out IPython magic to ensure Python compatibility.
#------------------------------------------------
# Early Stopping
#------------------------------------------------
earlyStop = EarlyStopping(monitor='loss_vae', 
                          mode='min', # indicate that we want to follow decreasing of the metric
                          patience=10, # adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement
                          verbose=1)

#------------------------------------------------
# Checkpoint
#------------------------------------------------
#!mkdir '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_chollet_v6/'
checkPath = '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/model.V6.1_RNA.h5'
checkpoint = ModelCheckpoint(filepath=checkPath, 
                             monitor='loss_vae',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True, #only weights because it is a subModel
                             mode='min')

#------------------------------------------------
# Training logger
#------------------------------------------------
logPath='/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/VAE_V6/training.V6.1_RNA.csv'
csv_logger = CSVLogger(logPath,
                       separator=",",
                       append=True)

#------------------------------------------------
# TensorBoard
#------------------------------------------------

# Realtime plot
#-rf "/content/logs"
#%load_ext tensorboard
# %reload_ext tensorboard

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# Codigo para abrir pantalla tensorboard
#%tensorboard --logdir logs
print("Ubicacions logs de Tensorboard",logdir)

#------------------------------------------------
# Callback group
#------------------------------------------------
callbacks=[tensorboard_callback,
           checkpoint,
           earlyStop,
           csv_logger]

"""### Función para generar back up de archivos"""

# Función para averiguar si los archivos de los callbacks existen y generar un back up
from tensorflow.python.ops.gen_array_ops import empty
from os import path, rename

def check_if_duplicated(filePath):
    if path.exists(filePath):
        numb = 1
        while True:
            newPath = "{0}backUp_{2}{1}".format(*path.splitext(filePath) + (numb,"backUp"))
            if path.exists(newPath):
                numb += 1
            else:
                break
    else:
        return print(" *No back up generated")               

    rename(filePath, newPath) 
    return print(" *BackUp generate: "+newPath)

if False:
    check_if_duplicated(checkPath)
    check_if_duplicated(logPath)

"""## Data upload and revision"""

#Identifico todos los archivos de expresión y su metadata y los guardo en orden alfabético
relevant_path = "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/single/"

# Listado de datos de expresión
list_of_dataExp = sorted( filter( os.path.isfile,
                        glob.glob(relevant_path + 'dataExp_*') ) )

# Listado de metadatos de expresión
list_of_metadataExp = sorted( filter( os.path.isfile,
                        glob.glob(relevant_path + 'metadataExp_*') ) )


print("Length of uploaded dataExp:",len(list_of_dataExp))
print("Length of uploaded metadataExp:",len(list_of_dataExp))

# remove file in folder data
#!rm -rf "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/"
#!mkdir "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data"

# Loop donde cargo archivo por archivo y los apilo uno arriba del otro
from numpy import array, zeros, newaxis
dataExp_array=np.zeros((256,256,0))
metadataExp_array=np.zeros((0,29))

counter=0
print("Uploading file:")
for file_data,file_metadata in zip(list_of_dataExp,list_of_metadataExp):
      
    counter=counter+1
    print(counter,end=" ")
    
    tempData=np.load(file_data,allow_pickle=True)
    if len(tempData.shape) == 2: #si solo hay una muestra, la dimensión del dato es 2D, con newaxis transformo a 3D
        tempData=tempData[:,:,newaxis]
    #print("temp shape:",tempData.shape)

    dataExp_array  = np.append(dataExp_array,tempData,axis=2)
    #print("dataExp_array shape:",dataExp_array.shape)
    
    tempMeta=np.load(file_metadata,allow_pickle=True)
    #print("temp shape:",temp.shape)
    #print("metadataExp_array shape:",metadataExp_array.shape)
    metadataExp_array = np.append(metadataExp_array,tempMeta,axis=0)

print("Final shape of dataExp_array:",dataExp_array.shape)
print("Final shape of metadataExp_array:",metadataExp_array.shape)

# Reviso de las frecuencias de cada tipo de tumor
import pandas as pd
unique, counts = np.unique(metadataExp_array[:,25], return_counts=True)
sample=np.asarray((unique, counts))
#for i in range(sample.shape[1]):
#    print(sample[:,i])

#df = pd.DataFrame([counts],columns = unique)
#df
df_labels=pd.DataFrame(metadataExp_array[:,25])
df_labels.columns=['labels']
print(df_labels.value_counts())
df_labels.value_counts().to_csv('/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/other/sampleFrequency_VAE_V6.1_RNA.csv', index=True, header=True)

#Código en donde extraigo el índice de las tipo de muestras deseadas
samples=["Primary Tumor","Metastatic","Solid Tissue Normal","Recurrent Tumor"]
sample_index = []
for i, x in enumerate(metadataExp_array[:,25]):
    if any(x == c for c in samples):
        sample_index.append(i)
print(pd.DataFrame(metadataExp_array[:,25][sample_index]).value_counts())
sample_label=metadataExp_array[:,25][sample_index]

#Convert to one-hot encoding

# libraries
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
labels = metadataExp_array[:,25]
labels = array(labels)
print("labels:",labels)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print("integer encoded:",integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("onehot:", onehot_encoded)
print("onehot shape:", onehot_encoded.shape)

# Keras necesita tensores de entrada de la siguiente estructura:
# * (batch, altura imagen, anchura imagen, filtros)
# sample_index: ínidice de filas del tipo de muestra que quiero predecir

 #función que cambio la dimensión 0 por la 2
dataExp = np.swapaxes(dataExp_array[:,:,sample_index],0,2)
#BCRA_img = np.swapaxes(BCRA_img,1,2) #función que cambio la dimensión 1 por la 2

#Cambio la escala de expresion a [0,1]
maxValue=np.max(dataExp).astype("float64")

#Agrego la dimension de los filtros
dataExp_train = np.expand_dims(dataExp, -1).astype("float64") / maxValue
#dataExp_train = min_max_range(np.expand_dims(dataExp, -1).astype("float64"))
#Corroboro dimensiones del input
dataExp_train.shape

# Reviso valores maximos y minimos
print(np.max(dataExp_train))
print(np.min(dataExp_train))

"""##Plot"""

#Genero gráfico para 10 muestras aleatorias

import matplotlib.pyplot as plt
from matplotlib import colors
import random

plt.figure(figsize=(30, 15))
plt.rcParams.update({'font.size': 16})

image_index = random.sample(range(0,dataExp_train.shape[0]),10)

#nDim1=dataExp_train.shape[1]
#nDim2=dataExp_train.shape[2]

# make a color map of fixed colors
cmap = colors.ListedColormap(['blue', 'red'])


n = 5  # How many digits we will display
#plt.figure(figsize=(20, 4))
for i,index in enumerate(image_index):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(dataExp_train[index].reshape(img_dim, img_dim),
               cmap='gist_rainbow')
               #cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title(metadataExp_array[index,25]+'\n'+metadataExp_array[index,18])

    
plt.show()

"""## Train the VAE"""

# Resumen de parametros del enconder y decoder 
print(encoder.summary())
print(decoder.summary())

# Commented out IPython magic to ensure Python compatibility.
# Abrir ventana de Tensorboard
# %tensorboard --logdir logs

#Configuro el entrenamiento
EPOCH = 1000
batch_size = 64


#Creo el
VAE_RNA=VAE(encoder, decoder)

# Path to the weights

#load from last checkpoint
#path_lastWeights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_chollet_v4.0/model.05-3037.12_v3.h5"
#if os.path.exists(path_lastWeights):
#  vae.built = True
#  vae.load_weights(filepath = path_lastWeights)

#Initiate from beginning
VAE_RNA.compile(optimizer=keras.optimizers.Adam(0.0001))
VAE_RNA.metrics_names
VAE_RNA.fit(dataExp_train,
        epochs=EPOCH,
        batch_size = batch_size,
        callbacks = callbacks
        )

"""## Cargo modelo entrenado:"""

# Path to the weights
print("Pesos cargados del siguiente archivo: ",checkPath)
#Creo el modelo
VAE_RNA=VAE(encoder, decoder)
VAE_RNA.built = True
VAE_RNA.load_weights(filepath = checkPath)
#Initiate from beginning
VAE_RNA.compile(optimizer=keras.optimizers.Adam(0.0001))

"""## Resumen de la evolución de los entrenamientos"""

# Cargo el tracker
df = pd.read_csv(logPath)

#Código para corregir si se reinició el entrenamiento (ya que las epocas vuelven a 0)
#df["epoch"]=np.arange(len(df.iloc[:,0]))
#df.to_csv(logPath)
df=df.iloc[236:726,:]
df.to_csv(logPath)
df

# Multiple plots 
# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
# Horizontal graphs
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = [20, 10]

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig.suptitle('Horizontally stacked subplots')

ax1.plot(df["epoch"], df["loss_vae"],'tab:red')
ax1.set_title('loss_vae')
ax2.plot(df["epoch"], df["reconstruction_loss"],'tab:green')
ax2.set_title('reconstruction_loss')
ax3.plot(df["epoch"], df["kl_loss"],'tab:blue')
ax3.set_title('kl_loss')

"""## Reconstructing images:"""

# Contruction of n random samples
# How many digits we will display/reconstruct
n = 10  
# Generate random samples according n
image_index = random.sample(range(0,dataExp_train.shape[0]),n)

data=dataExp_train[image_index,:,:,:]
metadata=metadataExp_array[image_index,:]

data.shape
#metadata.shape

#Reconstruction of images

#Encoder
z_mean,var_mean=VAE_RNA.encoder.predict(data)
z_sample=VAE_RNA.sampler(z_mean,var_mean)  

#Decoder
reconstruct = VAE_RNA.decoder.predict(z_sample)

plt.imshow(np.squeeze(reconstruct[0]), cmap="gist_rainbow")
plt.axis('off')
plt.title(metadataExp_array[image_index[0],25]+'\n'+metadataExp_array[image_index[0],18])

"""## Plot first 10 reconstructions"""

ndim1=ndim2=256
plt.figure(figsize=(30, 15))
plt.rcParams.update({'font.size': 16})

plt.suptitle('RNA reconstruction',fontsize = 48)
plt.subplots_adjust(top=0.9)
n=5
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(np.squeeze(data[i]),
               cmap="gist_rainbow")
    #plt.gray()
    #ax.get_xticklabels([])
    #ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel("Original", rotation=90, fontsize=24)
    plt.title(metadataExp_array[image_index[i],25]+'\n'+metadataExp_array[image_index[i],18])

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(np.squeeze(reconstruct[i]),
               cmap="gist_rainbow")
    #plt.gray()
    ax.set_yticklabels([])
    #ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("Reconstructed", rotation=90, fontsize=24)       
plt.show()

"""## Guardando solamente parte del encoder y decoder para transfer learning"""

# Itero sobre las capas de Encoder y Decoder
print("Layers in submodel Encoder\n")
for i,layer in enumerate(VAE_RNA.layers[0].layers):
    #print(layer.name, layer)
    print(i,layer.name)
print("\n*--------------------------*\n")
# Itero sobre las capas de decSub1
print("Layers in submodel Decoder\n")
for i,layer in enumerate(VAE_RNA.layers[1].layers):
    #print(layer.name, layer)
    print(i,layer.name)

# Summary de los submodelos
VAE_RNA.layers[0].layers[1].summary()
VAE_RNA.layers[1].layers[9].summary()

#Extraigo subsubmodelo encoder_sub1 y guardo sus pesos
saveSubEncoder_weight="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubEncoder_weight_RNA_V6.1.h5"
#check_if_duplicated(saveSubEncoder_weight)
print("Modelo: " + VAE_RNA.name)
print("  -> capa 1: " + VAE_RNA.layers[0].name)
print("       -> Submodelo: " + VAE_RNA.layers[0].layers[0].name)
print("       -> Submodelo: " + VAE_RNA.layers[0].layers[1].name)
print("       -> Submodelo: " + VAE_RNA.layers[0].layers[2].name)

print("\n *Saving weights of: "+VAE_RNA.layers[0].layers[1].name)
print(  " ",saveSubEncoder_weight)
VAE_RNA.layers[0].layers[1].save_weights(saveSubEncoder_weight)

#Extraigo subsubmodelo decoder_sub1 y guardo sus pesos
saveSubDecoder_weight="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubDecoder_weight_RNA_V6.1.h5"
#check_if_duplicated(saveSubEncoder_weight)
print("Modelo: "+VAE_RNA.name)
print("  -> capa 1: "+VAE_RNA.layers[1].name)
print("       -> Submodelo: "  +VAE_RNA.layers[1].layers[7].name)
print("       -> Submodelo: "  +VAE_RNA.layers[1].layers[8].name)
print("       -> Submodelo: "  +VAE_RNA.layers[1].layers[9].name)


print("\n *Saving weights of: "+VAE_RNA.layers[1].layers[9].name)
print(  " ",saveSubDecoder_weight)
VAE_RNA.layers[1].layers[9].save_weights(saveSubDecoder_weight)

"""## Preguntas

## Bibliografía

https://towardsdatascience.com/reparameterization-trick-126062cfd3c3

https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0

https://keras.io/guides/writing_a_training_loop_from_scratch/

https://www.tensorflow.org/guide/basics

https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
"""
