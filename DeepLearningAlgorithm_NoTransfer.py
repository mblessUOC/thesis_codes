# Variational AutoEncoder

**Author:** Maximiliano Bless<br>
**Date created:** 2022/05/14<br>
**Last modified:** 2022/05/23<br>
**Description:** Convolutional Variational AutoEncoder (VAE) + classifier to trained on PCA images from 3 different datasets: 
 
 * expression
 * methylation
 * miRNA

Images dimension:

  * 256x256
  * 256x256
  * 64x64

Class categories: 4

Muestras: 8985

Convolutional Kernel sizes: 32-64-128.

Latent dimension: 256

It was set to train in 3 different phases (differenciate trainable layers and learning rates) with *transfer learning* and *fine tuning*

**Conclusion**:

**Resumen**:

En este código se entreó el VAE con los 3 datasets junto con su clasificación
"""

from google.colab import drive
drive.mount('/content/drive')

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)

"""## Setup"""

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

#Limiting GPU memory growth
#https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

#delete GPU memory
from keras import backend as K
K.clear_session()
#device = cuda.get_current_device()
#device.reset()
#del VAE_classifier

# https://github.com/keras-team/keras/issues/12625
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc
import tensorflow

#reset Keras Session
def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

reset_keras()

"""## Create a sampling layer"""

class Sampler(layers.Layer):
    def call(self, z_mean,z_log_var):
        batch_size = tf.shape(z_mean)[0]
        z_size = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""## Build the encoder

### Build the encoder - **RNA**
"""

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

"""### Build the encoder - **Meth**



"""

img_dim = 256
down_ratio = 4
kernel_sizes=[32,64,128]
use_bias=True

Meth_encSub1_inputs = keras.Input(shape=(img_dim, img_dim, 1), name="x_Meth")

Meth_encSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias, name="Meth_encSub1_conv2D.1.0")(Meth_encSub1_inputs)
Meth_encSub1 = layers.BatchNormalization(name="Meth_encSub1_conv2D.1.1")(Meth_encSub1)
Meth_encSub1 = layers.LeakyReLU(name="Meth_encSub1_conv2D.1.2")(Meth_encSub1)

Meth_encSub1 = layers.MaxPool2D(pool_size=down_ratio, name="Meth_encSub1_MaxPool.1")(Meth_encSub1) # MaxPool -1-

Meth_encSub1 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias, name="Meth_encSub1_conv2D.2.0")(Meth_encSub1)
Meth_encSub1 = layers.BatchNormalization(name="Meth_encSub1_conv2D.2.1")(Meth_encSub1)
Meth_encSub1 = layers.LeakyReLU(name="Meth_encSub1_conv2D.2.2")(Meth_encSub1)

Meth_encSub1_outputs = layers.MaxPool2D(pool_size=down_ratio,name="Meth_encSub1_MaxPool.3")(Meth_encSub1) # MaxPool -2-

# Genero submodelo del encoder que utilizaré para transfer learning
Meth_encoder_sub1 = keras.Model(Meth_encSub1_inputs, Meth_encSub1_outputs, name="Meth_encoder_sub1")
print(RNA_encoder_sub1.summary())
keras.utils.plot_model(RNA_encoder_sub1, show_shapes = True)

"""### Build the encoder - **miRNA**"""

img_dim = 64
down_ratio = 2
kernel_sizes=[32,64,128]
use_bias=True

miRNA_encSub1_inputs = keras.Input(shape=(img_dim, img_dim, 1), name="x_miRNA")

miRNA_encSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias, name="miRNA_encSub1_conv2D.1.0")(miRNA_encSub1_inputs)
miRNA_encSub1 = layers.BatchNormalization(name="miRNA_encSub1_conv2D.1.1")(miRNA_encSub1)
miRNA_encSub1 = layers.LeakyReLU(name="miRNA_encSub1_conv2D.1.2")(miRNA_encSub1)

miRNA_encSub1 = layers.MaxPool2D(pool_size=down_ratio, name="miRNA_encSub1_MaxPool.1")(miRNA_encSub1) # MaxPool -1-

miRNA_encSub1 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias, name="miRNA_encSub1_conv2D.2.0")(miRNA_encSub1)
miRNA_encSub1 = layers.BatchNormalization(name="miRNA_encSub1_conv2D.2.1")(miRNA_encSub1)
miRNA_encSub1 = layers.LeakyReLU(name="miRNA_encSub1_conv2D.2.2")(miRNA_encSub1)

miRNA_encSub1_outputs = layers.MaxPool2D(pool_size=down_ratio,name="miRNA_encSub1_MaxPool.2")(miRNA_encSub1) # MaxPool -2-

# Genero resto del encoder
miRNA_encoder_sub1 = keras.Model(miRNA_encSub1_inputs,miRNA_encSub1_outputs, name="miRNA_encoder_sub1")
print(miRNA_encoder_sub1.summary())
keras.utils.plot_model(miRNA_encoder_sub1, show_shapes = True)

"""### Build the encoder with all the subparts - **latent space**"""

# Genero el los inputs correspondiente a cada ómica
latent_dim=256
RNA_inputs = keras.Input(shape=(256, 256, 1),name="x_RNA_input")
Meth_inputs = keras.Input(shape=(256, 256, 1),name="x_Methylation_input")
miRNA_inputs = keras.Input(shape=(64, 64, 1),name="x_miRNA_input")

# Uno cada input con su correspondiente parte de transfer learning
RNA_output=RNA_encoder_sub1(RNA_inputs)
Meth_output=Meth_encoder_sub1(Meth_inputs)
miRNA_output=miRNA_encoder_sub1(miRNA_inputs)

# Combino los 3 modelos en la capa "mergeSubEncoder"
mergeSubEncoder = layers.concatenate([RNA_output, Meth_output, miRNA_output],name="mergeSubEncoder",axis=(2)) # 0: por batch / 1: dimension 1 / 2: dimension 2 / 3: filtros / -1: filtros

down_ratio=2

# Genero el resto del encoder en donde se calculan las variables del espacio latente/latent space: z_mean y z_log_var

encSub2 = layers.Conv2D(kernel_sizes[2], down_ratio, strides=1, padding="same",use_bias=False, name="encSub2_conv2D.1.0")(mergeSubEncoder)
encSub2 = layers.BatchNormalization(name="encSub2_conv2D.1.1")(encSub2)
encSub2 = layers.LeakyReLU(name="encSub2_conv2D.1.2")(encSub2)

encSub2 = layers.MaxPool2D(pool_size=down_ratio*2,name="encSub2_MaxPool.1")(encSub2) # MaxPool -3-
#encSub2 = layers.MaxPool2D(pool_size=(down_ratio*2,down_ratio*6,),name="encSub2_MaxPool.1")(encSub2) # MaxPool -3-

encSub2 = layers.Flatten(name="encSub2_Flatten")(encSub2)

z_mean = layers.Dense(latent_dim, name="z_mean")(encSub2)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encSub2)

#encoder_2 = keras.Model(encSub2_inputs, [z_mean,z_log_var], name="encoder_sub2")
encoder = keras.Model([RNA_inputs, Meth_inputs, miRNA_inputs], [z_mean,z_log_var], name="encoder")
print(encoder.summary())
keras.utils.plot_model(encoder, show_shapes = True)

"""## Build the decoder

### Build the decoder - **RNA**
"""

# Genero submodelo del decoder que utilizaré para transfer learning
up_ratio = 4
use_bias=True

RNA_decSub1_inputs = keras.Input(shape=(16, 16, 64), name="RNA_decSub1_inputs")

RNA_decSub1 = layers.Conv2DTranspose(kernel_sizes[1], 3, strides=up_ratio, padding="same",use_bias=use_bias,name="RNA_decSub1_convTrans2D.1.0")(RNA_decSub1_inputs)
RNA_decSub1 = layers.BatchNormalization(name="RNA_decSub1_convTrans2D.1.1")(RNA_decSub1)
RNA_decSub1 = layers.LeakyReLU(name="RNA_decSub1_convTrans2D.1.2")(RNA_decSub1)

RNA_decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="RNA_decSub1_conv2D.2.0")(RNA_decSub1)
RNA_decSub1 = layers.BatchNormalization(name="RNA_decSub1_conv2D.2.1")(RNA_decSub1)
RNA_decSub1 = layers.LeakyReLU(name="RNA_decSub1_conv2D.2.2")(RNA_decSub1)

RNA_decSub1 = layers.Conv2DTranspose(kernel_sizes[0], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="RNA_decSub1_convTrans2D.3.0")(RNA_decSub1)
RNA_decSub1 = layers.BatchNormalization(name="RNA_decSub1_convTrans2D.3.1")(RNA_decSub1)
RNA_decSub1 = &2138 & 827 & 2097\\layers.LeakyReLU(name="RNA_decSub1_convTrans2D.3.2")(RNA_decSub1)

RNA_decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="RNA_decSub1_conv2D.4.0")(RNA_decSub1)
RNA_decSub1 = layers.BatchNormalization(name="RNA_decSub1_conv2D.4.1")(RNA_decSub1)
RNA_decSub1 = layers.LeakyReLU(name="RNA_decSub1_conv2D.4.2")(RNA_decSub1)

RNA_decSub1_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same",name="RNA_decSub1_output")(RNA_decSub1)

RNA_decoder_sub1 = keras.Model(RNA_decSub1_inputs, RNA_decSub1_outputs, name="RNA_decoder_sub1")
print(RNA_decoder_sub1.summary())
keras.utils.plot_model(RNA_decoder_sub1, show_shapes = True)

"""### Build the decoder - **Methylation**"""

# Genero submodelo del decoder que utilizaré para transfer learning
up_ratio = 4
use_bias=True

Meth_decSub1_inputs = keras.Input(shape=(16, 16, 64), name="Meth_decSub1_inputs")

Meth_decSub1 = layers.Conv2DTranspose(kernel_sizes[1], 3, strides=up_ratio, padding="same",use_bias=use_bias,name="decSub1_convTrans2D.1.0")(Meth_decSub1_inputs)
Meth_decSub1 = layers.BatchNormalization(name="Meth_decSub1_convTrans2D.1.1")(Meth_decSub1)
Meth_decSub1 = layers.LeakyReLU(name="Meth_decSub1_convTrans2D.1.2")(Meth_decSub1)

Meth_decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="decSub1_conv2D.2.0")(Meth_decSub1)
Meth_decSub1 = layers.BatchNormalization(name="Meth_decSub1_conv2D.2.1")(Meth_decSub1)
Meth_decSub1 = layers.LeakyReLU(name="Meth_decSub1_conv2D.2.2")(Meth_decSub1)

Meth_decSub1 = layers.Conv2DTranspose(kernel_sizes[0], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="Meth_decSub1_convTrans2D.3.0")(Meth_decSub1)
Meth_decSub1 = layers.BatchNormalization(name="Meth_decSub1_convTrans2D.3.1")(Meth_decSub1)
Meth_decSub1 = layers.LeakyReLU(name="Meth_decSub1_convTrans2D.3.2")(Meth_decSub1)

Meth_decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="Meth_decSub1_conv2D.4.0")(Meth_decSub1)
Meth_decSub1 = layers.BatchNormalization(name="Meth_decSub1_conv2D.4.1")(Meth_decSub1)
Meth_decSub1 = layers.LeakyReLU(name="Meth_decSub1_conv2D.4.2")(Meth_decSub1)

Meth_decSub1_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same",name="Methylation_recontructed")(Meth_decSub1)

Meth_decoder_sub1 = keras.Model(Meth_decSub1_inputs, Meth_decSub1_outputs, name="Meth_decoder_sub1")
print(Meth_decoder_sub1.summary())
keras.utils.plot_model(Meth_decoder_sub1, show_shapes = True)

"""### Build the decoder - **miRNA**

"""

# Genero submodelo del decoder que utilizaré para transfer learning
up_ratio = 2
use_bias=True

miRNA_decSub1_inputs = keras.Input(shape=(16, 16, 64), name="decSub1_inputs")

miRNA_decSub1 = layers.Conv2DTranspose(kernel_sizes[1], 3, strides=up_ratio, padding="same",use_bias=use_bias,name="miRNA_decSub1_convTrans2D.1.0")(miRNA_decSub1_inputs)
miRNA_decSub1 = layers.BatchNormalization(name="miRNA_decSub1_convTrans2D.1.1")(miRNA_decSub1)
miRNA_decSub1 = layers.LeakyReLU(name="miRNA_decSub1_convTrans2D.1.2")(miRNA_decSub1)

miRNA_decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="miRNA_decSub1_conv2D.2.0")(miRNA_decSub1)
miRNA_decSub1 = layers.BatchNormalization(name="miRNA_decSub1_conv2D.2.1")(miRNA_decSub1)
miRNA_decSub1 = layers.LeakyReLU(name="miRNA_decSub1_conv2D.2.2")(miRNA_decSub1)

miRNA_decSub1 = layers.Conv2DTranspose(kernel_sizes[0], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="miRNA_decSub1_convTrans2D.3.0")(miRNA_decSub1)
miRNA_decSub1 = layers.BatchNormalization(name="miRNA_decSub1_convTrans2D.3.1")(miRNA_decSub1)
miRNA_decSub1 = layers.LeakyReLU(name="miRNA_decSub1_convTrans2D.3.2")(miRNA_decSub1)

miRNA_decSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias,name="miRNA_decSub1_conv2D.4.0")(miRNA_decSub1)
miRNA_decSub1 = layers.BatchNormalization(name="miRNA_decSub1_conv2D.4.1")(miRNA_decSub1)
miRNA_decSub1 = layers.LeakyReLU(name="miRNA_decSub1_conv2D.4.2")(miRNA_decSub1)

miRNA_decSub1_outputs = layers.Conv2D(1, 3, activation="sigmoid", padding="same",name="miRNA_decSub1_output")(miRNA_decSub1)

miRNA_decoder_sub1 = keras.Model(miRNA_decSub1_inputs, miRNA_decSub1_outputs, name="miRNA_decoder_sub1")
print(miRNA_decoder_sub1.summary())
keras.utils.plot_model(miRNA_decoder_sub1, show_shapes = True)

"""### Build the decoder - **latent space**"""

# Genero el input del encoder (z_sample)
up_ratio = 4 
use_bias=True


z_sample = keras.Input(shape=(latent_dim,),name="z_sample")

decSub2 = layers.Dense(12 * 4 * kernel_sizes[2], activation="relu",name="decSub2_Dense")(z_sample)

decSub2 = layers.Reshape((4, 12, kernel_sizes[2]),name="decSub2_Reshape")(decSub2)

decSub2 = layers.Conv2DTranspose(kernel_sizes[2], 3, activation="relu", strides=up_ratio, padding="same",use_bias=use_bias,name="decSub2_convTrans2D.1.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_convTrans2D.1.1")(decSub2)
decSub2 = layers.LeakyReLU(name="decSub2_convTrans2D.1.2")(decSub2)

decSub2 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias,name="decSub2_conv2D.2.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_conv2D.2.1")(decSub2)
decSub2_outputs = layers.LeakyReLU(name="decSub2_conv2D.2.2")(decSub2)

RNA_decSub2_input, Meth_decSub2_input, miRNA_decSub2_input = tf.split(decSub2_outputs, num_or_size_splits=3, axis=(2), name="decSub2_split.3")

RNA_decoder_sub1_output = RNA_decoder_sub1(RNA_decSub2_input)
Meth_decoder_sub1_output = Meth_decoder_sub1(Meth_decSub2_input)
miRNA_decoder_sub1_output = miRNA_decoder_sub1(miRNA_decSub2_input)

decoder = keras.Model(z_sample, [RNA_decoder_sub1_output,Meth_decoder_sub1_output,miRNA_decoder_sub1_output], name="decoder")
print(decoder.summary())
keras.utils.plot_model(decoder, show_shapes = True)



"""## Build the classifier"""

no_classes=4

latent_z_mean = keras.Input(shape=(latent_dim,),name="classifier_layerIn1")
#latent_z_var = keras.Input(shape=(latent_dim,),name="classifier_layerIn2")
#c = layers.concatenate([latent_z_mean, latent_z_var],name="classifier_layer3")
#c = layers.Dense(64, activation='relu',name="classifier_layer4")(c)
c = layers.Dense(latent_dim*1.1, activation='relu',name="classifier_layer4")(latent_z_mean)
c = layers.Dropout(0.  )(c)
c = layers.Dense(latent_dim*0.7, activation='relu',name="classifier_layer5")(c)
c = layers.Dropout(0.2)(c)
classifier_output = layers.Dense(no_classes,activation="softmax",name="classifier_layerOut")(c)
#classifier = keras.Model([latent_z_mean, latent_z_var], classifier_output, name="classifier")
classifier = keras.Model(latent_z_mean, classifier_output, name="classifier")
classifier.summary()
keras.utils.plot_model(classifier, show_shapes = True)

"""## Define the VAE as a `Model` with a custom `train_step`




"""

class VAE_contructor(keras.Model):
    def __init__(self, encoder, decoder, classifier,**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.sampler = Sampler()
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        #------------------------------
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.RNA_reconstruction_loss_tracker = keras.metrics.Mean(name="RNA_reconstruction_loss")
        self.Meth_reconstruction_loss_tracker = keras.metrics.Mean(name="Meth_reconstruction_loss")
        self.miRNA_reconstruction_loss_tracker = keras.metrics.Mean(name="miRNA_reconstruction_loss")
        #------------------------------
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.cce_loss_tracker = keras.metrics.Mean(name="class_loss")
        self.class_acc_tracker = keras.metrics.CategoricalAccuracy(name="class_acc")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            #------------------------------
            self.reconstruction_loss_tracker,
            self.RNA_reconstruction_loss_tracker,
            self.Meth_reconstruction_loss_tracker,
            self.miRNA_reconstruction_loss_tracker,
            #------------------------------
            self.kl_loss_tracker,
            self.cce_loss_tracker,
            self.class_acc_tracker
        ]

    # one batch train
    def train_step(self, data):
        with tf.GradientTape() as tape:
            input, targets = data
            #input[0]: RNA_input
            #input[1]: Meth_input
            #input[2]: miRNA_input
            #target: type of sample+tumor
            
            # VAE loss
            z_mean, z_log_var = self.encoder(input)
            z = self.sampler(z_mean,z_log_var)
            RNA_reconstruction, Meth_reconstruction, miRNA_reconstruction = self.decoder(z)

            # Reconstruction loss calculation
            # - For each omics
            RNA_reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(input[0], RNA_reconstruction), axis=(1, 2)))
            Meth_reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(input[1], Meth_reconstruction), axis=(1, 2)))
            miRNA_reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(input[2], miRNA_reconstruction), axis=(1, 2)))
            # - Total 
            reconstruction_loss = tf.reduce_mean([RNA_reconstruction_loss,Meth_reconstruction_loss,miRNA_reconstruction_loss])

            #------------------------------
            # KL loss calculation
            # - Total 
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

            #------------------------------
            # Classification loss
            predictions = self.classifier(z_mean)
            classfication_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(targets,predictions))

            #------------------------------
            # Total loss = VAE + classification
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss) + classfication_loss 
                   
        # calculate the gradients using our tape and then update the model weights
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        #updating performance values
        self.total_loss_tracker.update_state(total_loss)
        #------------------------------
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.RNA_reconstruction_loss_tracker.update_state(RNA_reconstruction_loss)
        self.Meth_reconstruction_loss_tracker.update_state(Meth_reconstruction_loss)
        self.miRNA_reconstruction_loss_tracker.update_state(miRNA_reconstruction_loss)
        #------------------------------
        self.kl_loss_tracker.update_state(kl_loss)
        self.cce_loss_tracker.update_state(classfication_loss)
        self.class_acc_tracker.update_state(targets,predictions)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "RNA_reconstruction_loss": self.RNA_reconstruction_loss_tracker.result(),
            "Meth_reconstruction_loss": self.Meth_reconstruction_loss_tracker.result(),
            "miRNA_reconstruction_loss": self.miRNA_reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "class_loss": self.cce_loss_tracker.result(),
            "class_acc": self.class_acc_tracker.result()
        }

    #def predict_step(self, input):
    #    z_mean, z_log_var = self.encoder(input,training=False)
    #    predictions = self.classifier([z_mean,z_log_var],training=False)
    #    return (x)

    def predict(self, *input,**kwargs):
        result = []
        for xs in input:
            #z_mean, z_log_var = self.encoder(xs,training=False)
            z_mean, _ = self.encoder(xs,training=False)
            #predictions = self.classifier([z_mean,z_log_var],training=False)
            predictions = self.classifier(z_mean,training=False)
            result.append(predictions)
            return tf.concat(result, axis=0)
            
    # one batch test     
    def test_step(self, data):
        input, targets = data
        z_mean, z_log_var = self.encoder(input, training=False)
        #z_mean, _ = self.encoder(input, training=False)
        #z = self.sampler(z_mean,z_log_var)
        #predictions = self.classifier([z_mean,z_log_var], training=False)
        predictions = self.classifier(z_mean, training=False)
        classfication_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(targets,predictions))
        #self.val_CCE_loss_tracker.update_state(classfication_loss)
        self.cce_loss_tracker.update_state(classfication_loss)
        #return {"CCE_loss": self.val_CCE_loss_tracker.result()}
        self.class_acc_tracker.update_state(targets,predictions)
        return {"class_loss": self.cce_loss_tracker.result(),
                "class_acc": self.class_acc_tracker.result()}

"""## Callbacks

Tensorflow callbacks are functions or blocks of code which are executed during a specific instant while training a Deep Learning Model. The following callbacks will be used during the training:

 * Early Stopping

 * Checkpoint

 * Training logger

 * TensorBoard
"""

!mkdir '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/superVAE_V3.0/'
!mkdir '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/superVAE_V3.0/'

# Borra carpetas necesarias
#!rm -rf '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/model.SuperVAE_V3.h5'
!rm -rf "logs"
#!rm -rf '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/VAE_V6/training.SuperVAE_V3.csv'

# Commented out IPython magic to ensure Python compatibility.
#------------------------------------------------
# Early Stopping
#------------------------------------------------

earlyStop = EarlyStopping(monitor='total_loss', 
                          mode='min', # indicate that we want to follow decreasing of the metric
                          patience=10, # adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement
                          verbose=1)
#------------------------------------------------
# Checkpoint
#------------------------------------------------

#!mkdir '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_chollet_v6/'
checkPath = '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/superVAE_V3.0/model.SuperVAE_V3.h5'
checkpoint = ModelCheckpoint(filepath=checkPath, 
                             monitor='total_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True, #only weights because it is a subModel
                             mode='min')
#------------------------------------------------
# Training logger
#------------------------------------------------

logPath='/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/superVAE_V3.0/training.SuperVAE_V3.csv'
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
relevant_path = "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/all/"

# Listado de datos de expresión
list_of_dataExp = sorted( filter( os.path.isfile,glob.glob(relevant_path + 'dataExp_*') ) )
list_of_dataMeth = sorted( filter( os.path.isfile,glob.glob(relevant_path + 'dataMeth_*') ) )
list_of_datamiRNA = sorted( filter( os.path.isfile,glob.glob(relevant_path + 'datamiRNA_*') ) )

#Listado de metadatos de expresión
list_of_metadata = sorted( filter( os.path.isfile,glob.glob(relevant_path + 'metadataExp_*') ) )


print("Length of uploaded Exp:",len(list_of_dataExp))
print("Length of uploaded Meth:",len(list_of_dataMeth))
print("Length of uploaded miRNA:",len(list_of_datamiRNA))

print("Length of uploaded metadata:",len(list_of_metadata))

# remove file in folder data
#!rm -rf "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/"
#!mkdir "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data"

# Loop donde cargo datos de las omicas y etiquetas
from numpy import array, zeros, newaxis
data_arrayExp=np.zeros((256,256,0))
data_arrayMeth=np.zeros((256,256,0))
data_arraymiRNA=np.zeros((64,64,0))
metadata_array=np.zeros((0,29))

labels_array=np.zeros((0,1))
counter=0
print("Uploading file:")
for file_Exp,file_Meth,file_miRNA,file_metadata in zip(list_of_dataExp,list_of_dataMeth,list_of_datamiRNA,list_of_metadata):
    
    counter=counter+1
    print(counter,end=" ")
    #---------------------------
    #RNA
    tempData=np.load(file_Exp,allow_pickle=True)
    if len(tempData.shape) == 2: #si solo hay una muestra, la dimensión del dato es 2D, con newaxis transformo a 3D
        tempData=tempData[:,:,newaxis]
    data_arrayExp = np.append(data_arrayExp,tempData,axis=2)
    #---------------------------
    #Meth
    tempData=np.load(file_Meth,allow_pickle=True)
    if len(tempData.shape) == 2: #si solo hay una muestra, la dimensión del dato es 2D, con newaxis transformo a 3D
        tempData=tempData[:,:,newaxis]
    data_arrayMeth = np.append(data_arrayMeth,tempData,axis=2)
    #---------------------------
    #miRNA
    tempData=np.load(file_miRNA,allow_pickle=True)
    if len(tempData.shape) == 2: #si solo hay una muestra, la dimensión del dato es 2D, con newaxis transformo a 3D
        tempData=tempData[:,:,newaxis]
    data_arraymiRNA = np.append(data_arraymiRNA,tempData,axis=2)
    #print("temp shape:",tempData.shape)
    #---------------------------
        
    tempMeta=np.load(file_metadata,allow_pickle=True)
    #print("temp shape:",temp.shape)
    #print("metadata_array shape:",metadata_array.shape)
    metadata_array = np.append(metadata_array,tempMeta,axis=0)

#Genero etiquetas por tumor y por tipo
#labels_array=metadata_array[:,18] + '-' + metadata_array[:,25]
labels_array=metadata_array[:,25]

# Solo vuelvo a cargar las etiquetas

metadata_array=np.zeros((0,29))
labels_array=np.zeros((0,1))
counter=0
print("Uploading file:")
for file_metadata in list_of_metadata:
    counter=counter+1
    print(counter,end=" ")
    tempMeta=np.load(file_metadata,allow_pickle=True)
    metadata_array = np.append(metadata_array,tempMeta,axis=0)

#Genero etiquetas por tumor y por tipo
labels_array=metadata_array[:,25]

print("Final shape of data_arrayExp:",data_arrayExp.shape)
print("Final shape of data_arrayMeth:",data_arrayMeth.shape)
print("Final shape of data_arraymiRNA:",data_arraymiRNA.shape)

print("Final shape of metadata_array:",metadata_array.shape)
print("Final shape of labels_array:",labels_array.shape)
labels_array[0]

# Reviso de las frecuencias de cada tipo de tumor

#unique, counts = np.unique(metadata_array[:,25], return_counts=print("   ",data_miRNA_train.shape,y_train.shape)

unique, counts = np.unique(labels_array, return_counts=True)
sample=np.asarray((unique, counts))
#for i in range(sample.shape[1]):
#    print(sample[:,i])

#df = pd.DataFrame(counts,index = unique)
df = pd.DataFrame(counts,unique)
singleSample = df.loc[df[0]==1].index
df_labels=pd.DataFrame(labels_array)
df_labels.columns=['labels']
print(df_labels.value_counts())
df_labels.value_counts().to_csv('/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/other/sampleFrequency_SuperVAE_V2.0.csv', index=True, header=True)



import matplotlib.pyplot as plt
plt.hist(df_labels.value_counts(), bins = 76,edgecolor = 'black')
plt.show()

#Muestro posición de las muestras con 1 unidad
sample_alone=df_labels[df_labels['labels'].isin(df_labels['labels'].value_counts()[df_labels['labels'].value_counts()==1].index)].labels
print(sample_alone)
index = sample_alone.index
a_list = list(index)
print(a_list)

#Código en donde extraigo el índice de las tipo de muestras deseadas
samples=["Primary Tumor","Metastatic","Solid Tissue Normal","Recurrent Tumor"]
sample_index = []
for i, x in enumerate(labels_array):
    if any(x == c for c in samples):
        sample_index.append(i)
print(pd.DataFrame(labels_array[sample_index]).value_counts())
sample_label=labels_array[sample_index]

sample_label.shape

#Convert to one-hot encoding

# libraries
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
#labels = metadata_array[:,25]
labels = sample_label
labels = array(labels)
print("labels:",labels)
print("labels shape:",labels.shape)

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
# (batch, altura imagen, anchura imagen, filtros)

data_RNA = np.swapaxes(data_arrayExp,0,2) #función que cambio la dimensión 0 por la 2
data_Meth = np.swapaxes(data_arrayMeth,0,2) #función que cambio la dimensión 0 por la 2
data_miRNA = np.swapaxes(data_arraymiRNA,0,2) #función que cambio la dimensión 0 por la 2

#Normalizo las expresion de [0,1]
maxValueRNA=np.max(data_RNA).astype("float32")
maxValueMeth=np.max(data_Meth).astype("float32")
maxValuemiRNA=np.max(data_miRNA).astype("float32")


#Agrego la dimension de los filtros
data_RNA_train = np.expand_dims(data_RNA, -1).astype("float32") / maxValueRNA
data_Meth_train = np.expand_dims(data_Meth, -1).astype("float32") / maxValueMeth
data_miRNA_train = np.expand_dims(data_miRNA, -1).astype("float32") / maxValuemiRNA
#data_train = min_max_range(np.expand_dims(data, -1).astype("float64"))
#Corroboro dimensiones del input
print(data_RNA_train.shape)
print(data_Meth_train.shape)
print(data_miRNA_train.shape)

#sin escalamiento min/max
if False:
    data = np.swapaxes(data_array,0,2) #función que cambio la dimensión 0 por la 2
    data_train = np.expand_dims(data, -1).astype("float64")
    data_train.shape

# Reviso valores maximos y minimos
print("data_RNA_train:",np.min(data_RNA_train),"-",np.max(data_RNA_train))
print("data_Meth_train:",np.min(data_Meth_train),"-",np.max(data_Meth_train))
print("data_miRNA_train:",np.min(data_miRNA_train),"-",np.max(data_miRNA_train))

#Spliting in training and testing in a balanced manner
from sklearn.model_selection import train_test_split
#data_train_idx = np.arange(0, data_RNA_train.shape[0])
data_train_idx=sample_index

X_train_id, X_test_id, y_train, y_test = train_test_split(data_train_idx, onehot_encoded, test_size=0.2, random_state=1, stratify=onehot_encoded)

#data_RNA_train=data_RNA_train[X_train_id,:,:,:]

# Extract test data (20%)
data_RNA_test=data_RNA_train[X_test_id,:,:,:]
data_Meth_test=data_Meth_train[X_test_id,:,:,:]
data_miRNA_test=data_miRNA_train[X_test_id,:,:,:]
test_labels = labels_array[X_test_id]
#y_test array with oneHot encoding


# Extract validation data from training set (20% from training / 16% from total)
X_train_id, X_test_id, y_train, y_test = train_test_split(X_train_id, y_train, test_size=0.2, random_state=1, stratify=y_train)

data_RNA_val=data_RNA_train[X_test_id,:,:,:]
data_Meth_val=data_Meth_train[X_test_id,:,:,:]
data_miRNA_val=data_miRNA_train[X_test_id,:,:,:]
y_val=y_test
val_labels = labels_array[X_test_id]

#Extract trainig data (64%)
data_RNA_train = data_RNA_train[X_train_id,:,:,:]
data_Meth_train = data_Meth_train[X_train_id,:,:,:]
data_miRNA_train = data_miRNA_train[X_train_id,:,:,:]
#y_train
train_labels = labels_array[X_train_id]



print("Dimensiones RNA:")
print(" - Train")
print("   ",data_RNA_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_RNA_val.shape,y_val.shape)
print(" - Test")
print("   ",data_RNA_test.shape,y_test.shape)
print("")
print("Dimensiones Meth:")
print(" - Train")
print("   ",data_Meth_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_Meth_val.shape,y_val.shape)
print(" - Test")
print("   ",data_Meth_test.shape,y_test.shape)
print("")
print("Dimensiones miRNA:")
print(" - Train")
print("   ",data_miRNA_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_miRNA_val.shape,y_val.shape)
print(" - Test")
print("   ",data_miRNA_test.shape,y_test.shape)
print("")

print("Dimensiones RNA:")
print(" - Train")
print("   ",data_RNA_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_RNA_val.shape,y_val.shape)
print(" - Test")
print("   ",data_RNA_test.shape,y_test.shape)
print("")
print("Dimensiones Meth:")
print(" - Train")
print("   ",data_Meth_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_Meth_val.shape,y_val.shape)
print(" - Test")
print("   ",data_Meth_test.shape,y_test.shape)
print("")
print("Dimensiones miRNA:")
print(" - Train")
print("   ",data_miRNA_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_miRNA_val.shape,y_val.shape)
print(" - Test")
print("   ",data_miRNA_test.shape,y_test.shape)
print("")

#!mkdir "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2"

# Guardo dataset modificados
if False:
    # Datasets
    #training
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_RNA_train"  , data_RNA_train)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_Meth_train" , data_Meth_train)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_miRNA_train", data_miRNA_train) 

    #validation
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_RNA_val"  , data_RNA_val)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_Meth_val" , data_Meth_val)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_miRNA_val", data_miRNA_val)

    #test
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_RNA_test"  , data_RNA_test)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_Meth_test" , data_Meth_test)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_miRNA_test", data_miRNA_test) 

    # One hot labels
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/y_train", y_train)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/y_val"  , y_val)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/y_test" , y_test)

    # String labels
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/train_labels", train_labels)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/val_labels"  , val_labels)
    np.save("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/test_labels" , test_labels)

"""## Uploading saved data"""

if True:
    # Datasets

    #training
    data_RNA_train  =  np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_RNA_train.npy")
    data_Meth_train =  np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_Meth_train.npy")
    data_miRNA_train = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_miRNA_train.npy") 
    
    #test
    data_RNA_test  =  np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_RNA_test.npy")
    data_Meth_test=   np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_Meth_test.npy")
    data_miRNA_test = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_miRNA_test.npy") 

    #validation
    data_RNA_val  =  np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_RNA_val.npy")
    data_Meth_val=   np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_Meth_val.npy")
    data_miRNA_val = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/data_miRNA_val.npy") 

    # One hot labels
    y_train = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/y_train.npy")
    y_test =  np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/y_test.npy")
    y_val =   np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/y_val.npy")


    # String labels
    train_labels = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/train_labels.npy",allow_pickle=True)
    val_labels = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/val_labels.npy",allow_pickle=True)
    test_labels = np.load("/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/superVAE_V2/test_labels.npy",allow_pickle=True)

print("Dimensiones RNA:")
print(" - Train")
print("   ",data_RNA_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_RNA_val.shape,y_val.shape)
print(" - Test")
print("   ",data_RNA_test.shape,y_test.shape)
print("")

print("Dimensiones Meth:")
print(" - Train")
print("   ",data_Meth_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_Meth_val.shape,y_val.shape)
print(" - Test")
print("   ",data_Meth_test.shape,y_test.shape)
print("")

print("Dimensiones miRNA:")
print(" - Train")
print("   ",data_miRNA_train.shape,y_train.shape)
print(" - Validation")
print("   ",data_miRNA_val.shape,y_val.shape)
print(" - Test")
print("   ",data_miRNA_test.shape,y_test.shape)
print("")

"""##Plot"""

#Ploting n random constructed sample with their originals
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl

n=5 # imágenes por fila
pad = 0 # in points

ticklabelpad = mpl.rcParams['xtick.major.pad']

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(20, 10))

for i in range(n):
    
    # Display RNA
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(data_RNA_train[i].reshape(256, 256),
               cmap="gist_rainbow")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    plt.title(train_labels[i]+ '\n' )
    if i == 0:
        ax.set_ylabel("RNA", rotation=90, size='large')
        #ax.annotate('XLabel', xy=(0,0.5), xytext=(-100, -ticklabelpad), ha='left', va='top',xycoords='axes fraction', textcoords='offset points')
    

    # Display Meth
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(data_Meth_train[i].reshape(256, 256),
               cmap="gist_rainbow")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.set_yticklabels([])
    #ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("Meth", rotation=90, size='large')

    # Display miRNA
    ax = plt.subplot(3, n, i + 1 + n*2)
    plt.imshow(data_miRNA_train[i].reshape(64, 64),
               cmap="gist_rainbow")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.set_yticklabels([])
    #ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("MiRNA", rotation=90, size='large')

plt.show()

"""## Train the VAE

### Transfer learning

https://keras.io/guides/transfer_learning/
https://www.tensorflow.org/tutorials/images/transfer_learning
https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/
https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

**Transfer learning**
  
  * Inicial el modelo y cargar los pesos pre-entrenado
  * Congelar todas las capas del modelo con los nuevos pesos colocando `.traibale=False`.
    * Los cambios en los estados de las capas solo se actualizan cada vez que se aplique `compile()`
  * Entrenar el modelo con los datos hasta que converja

**Fine tuninig**

  * Descongelar las capas y volver a reentrenar todo el modelo con un learning rate menor.
      * Puede producir mejorías muy grandes, pero a costa de overfitting.

  * Es importante que el modelo con las capas congeladas haya convergido antes de hacer esto. Sino las capas con inicialización aleatoria generarán cambios grandes en los gradientes y se perderá la información ganada en las capas previamente congeladas
  * Es importante utilizar learning rate chiquitos en el fine tuninig, ya el modelo aumentoó de tamaño pudiendo generar overfitting.

**BatchNormalization layer**

BatchNormalization contiene 2 pesos no entrenables que se actualizan durante los entrenamientos. Estas son las variables que documentan la media y la varianza de los inputs.

Cuando se setean `bn_layer.traiable=False`, la capa BatchNormalization corre en modo inferencia y no actualizará su media y varianza según los inputs.

Cuando se descongela el modelo que contiene capas BatchNormalization para realizar fine tuning, se debe mantener las capas BatchNormalization con `.training=False` cuando se llama al modelo base. Sino las actualizaciones aplicadas a los pesos no entrenables destruiran el modelo aprendido.

### Función donde defino estado de entrenamiento de las capas
"""

#trainableStatus(subModel,model_Trainable=True,batchNorm_Trainable=True,showPrint=False)
def trainableStatus(subModel,model_Trainable=True,batchNorm_Trainable=True,showPrint=False):
  # subModel: modelo que quiero analizar e iterar para cambiar .trainable
  # model_Trainable: bool que indica el estado .trainable en todas las capas
  # batchNorm_Trainable: bool que indica el estado .trainable SOLO de 
  #                      layers.BatchNormalization()

  subModel.trainable=model_Trainable

  if model_Trainable == False: #si es False quiero que todo el modelo no pueda ser entrenado
    if showPrint == True: print(" - Model in inference mode - \n")
  
  else: #si es True quiero que todo el modelo ser entrenado
    if showPrint == True: print(" - Model in training mode - \n")
  
  #Analizo primera capa del modelo
  for l in subModel.layers: 
    if showPrint == True: print(l.name,"-",l.trainable)
    if isinstance(l, layers.BatchNormalization):
      l.trainable = batchNorm_Trainable
      if showPrint == True: print(  "** BatchNorm Layer **")      
      if batchNorm_Trainable==False:
        if showPrint == True: print("   **Inference Mode - ",l.trainable)
      else:
        if showPrint == True: print("   **Trainable Mode - ",l.trainable)

    try:
      for l2 in l.layers: #Analizo segunda capa del modelo
        if showPrint == True: print(l2.name,"-",l2.trainable)
        if isinstance(l2, layers.BatchNormalization):
          l2.trainable = batchNorm_Trainable
          if showPrint == True: print("   ** BatchNorm Layer **")
          if batchNorm_Trainable==False:            
            if showPrint == True: print("      **Inference Mode - ",l2.trainable)
          else:            
            if showPrint == True: print("      **Trainable Mode - ",l2.trainable)
      try:
        for l3 in l2.layers: #Analizo tercera capa del modelo
          print("   **CUIDADO NO SE ANALIZA ESTA CAPA**",l3.name,"-",l2.trainable) 
          print("   -",l3.name,"-",l2.trainable)    
      except:
        pass

    except:
      pass

"""### Creo modelo para cargar los pesos"""

#Creo el modelo base para cargar los datos
VAE_classifier=VAE_contructor(encoder, decoder, classifier)
VAE_classifier.built = True

"""###Transfer learning - **Encoder**"""

# Ubicación de los submodelos del encoder para hacer transfer learning
print("Encoder submodel to be transfered")
print(VAE_classifier.encoder.layers[3].name)
print(VAE_classifier.encoder.layers[4].name)
print(VAE_classifier.encoder.layers[5].name)

#paths para los pesos de los submodelos en el encoder
RNA_encoder_weights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubEncoder_weight_RNA_V6.1.h5"
Meth_encoder_weights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubEncoder_weight_Meth_V6.1.h5"
miRNA_encoder_weights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubEncoder_weight_miRNA_V6.1.h5"

for i,path_weights in enumerate([RNA_encoder_weights,Meth_encoder_weights,miRNA_encoder_weights]):
     #print(i,path)
    print(i,"-",VAE_classifier.encoder.layers[i+3].name)
    print("    -> Pesos cargados:",path_weights,"\n")
    VAE_classifier.encoder.layers[i+3].load_weights(filepath = path_weights)
    trainableStatus(VAE_classifier.encoder.layers[i+3],model_Trainable=False,batchNorm_Trainable=False,showPrint=False)

if False:
    print(encoder.layers[3].summary())
    print(encoder.layers[4].summary())
    print(encoder.layers[5].summary())

"""###Transfer learning - **Decoder**"""

# Ubicación de los submodelos del decoder para hacer transfer learning
print("Decoder submodel to be transfered")
print(decoder.layers[10].name)
print(decoder.layers[11].name)
print(decoder.layers[12].name)

#paths para los pesos de los submodelos en el decoder
RNA_decoder_weights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubDecoder_weight_RNA_V6.1.h5"
Meth_decoder_weights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubDecoder_weight_Meth_V6.1.h5"
miRNA_decoder_weights="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubDecoder_weight_miRNA_V6.1.h5"

for i,path_weights in enumerate([RNA_decoder_weights,Meth_decoder_weights,miRNA_decoder_weights]):
     #print(i,path)
    print(i,"-",VAE_classifier.decoder.layers[i+10].name)
    print("    -> Pesos cargados:",path_weights,"\n")
    VAE_classifier.decoder.layers[i+10].load_weights(filepath = path_weights)
    trainableStatus(VAE_classifier.decoder.layers[i+10],model_Trainable=False,batchNorm_Trainable=False,showPrint=False)

if False:
    print(decoder.layers[10].summary())
    print(decoder.layers[11].summary())
    print(decoder.layers[12].summary())

# Muestro estado de los parámetros
encoder.summary()
decoder.summary()
classifier.summary()

for i in range(3):
    trainableStatus(VAE_classifier.encoder.layers[i+3],model_Trainable=False,batchNorm_Trainable=False,showPrint=False)
for i in range(3):
    trainableStatus(VAE_classifier.decoder.layers[i+10],model_Trainable=False,batchNorm_Trainable=False,showPrint=False)

"""### Training the superVAE"""

# Commented out IPython magic to ensure Python compatibility.
# Abrir ventana de Tensorboard
# %tensorboard --logdir logs

# Creating and compiling the model
#VAE_classifier = VAE_contructor(encoder, decoder, classifier)
#VAE_classifier.built = True
#VAE_classifier.summary()
VAE_classifier.compile(optimizer=keras.optimizers.Adam(1e-4))
print("\nAvailable metrics:\n",pd.DataFrame(VAE_classifier.metrics_names,columns = [""]).to_string(index=False, header=True))

print("Dimensiones RNA:",data_RNA_train.shape)
print("Dimensiones Meth:",data_Meth_train.shape)
print("Dimensiones miRNA:",data_miRNA_train.shape)

"""### Training settings"""

verbosity = 1
batch_size = 20

"""### Fase 1 entrenamiento

Solamente se entrena la mitad del superVAE, sin los modelos a los cuales se hicieron transfer training ni el clasificador
"""

# Phase 1
K.clear_session()
VAE_classifier=VAE_contructor(encoder, decoder, classifier)
VAE_classifier.built = True

# Classfier submodel is freezed
VAE_classifier.classifier.trainable=False
VAE_classifier.encoder.trainable=True
VAE_classifier.decoder.trainable=True

#Loop through layers 3,4 and 5 and freeze the layers that has been learn transfered
for i in range(3):
    trainableStatus(VAE_classifier.encoder.layers[i+3],model_Trainable=False,batchNorm_Trainable=False,showPrint=False)
    #Loop through layers 10,11 and 12 and freeze the layers that has been learn transfered
for i in range(3):
    trainableStatus(VAE_classifier.decoder.layers[i+10],model_Trainable=False,batchNorm_Trainable=False,showPrint=False)


VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))
VAE_classifier.summary()

K.clear_session()

VAE_classifier.fit(x=[data_RNA_train,data_Meth_train,data_miRNA_train],y=y_train,
                       initial_epoch=0,
                       epochs = 20,
                       verbose = verbosity,
                       batch_size = batch_size,
                       callbacks = callbacks,
                       validation_data=([data_RNA_val,data_Meth_val,data_miRNA_val],y_val)
                       #validation_split=validation_split
                       )

K.clear_session()

if True:
    del VAE_classifier
    VAE_classifier=VAE_contructor(encoder, decoder, classifier)
    VAE_classifier.built = True    
    VAE_classifier.load_weights(filepath = checkPath)
    VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))

reset_keras()
if True:
    del VAE_classifier
    VAE_classifier=VAE_contructor(encoder, decoder, classifier)
    VAE_classifier.built = True    
    VAE_classifier.load_weights(filepath = checkPath)
    VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))

"""### Fase 2 entrenamiento
En esta fase se hace fine tuning del VAE solamente, congelando el classifier
"""

# Phase 2
VAE_classifier=VAE_contructor(encoder, decoder, classifier)
VAE_classifier.built = True

# Classfier submodel is freezed
VAE_classifier.classifier.trainable=False
VAE_classifier.encoder.trainable=True
VAE_classifier.decoder.trainable=True

#Loop through layers 3,4 and 5 to freeze batchNormalization layer in the subEncoders
for i in range(3):
    trainableStatus(VAE_classifier.encoder.layers[i+3],model_Trainable=True,batchNorm_Trainable=False,showPrint=False)
    #Loop through layers 10,11 and 12 to freeze batchNormalization layer in the subDecoders
for i in range(3):
    trainableStatus(VAE_classifier.decoder.layers[i+10],model_Trainable=True,batchNorm_Trainable=False,showPrint=False)


VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.00001))
VAE_classifier.summary()

VAE_classifier.fit(x=[data_RNA_train,data_Meth_train,data_miRNA_train],y=y_train,
                       initial_epoch=20,
                       epochs = 20,
                       verbose = verbosity,
                       batch_size = batch_size,
                       callbacks = callbacks,
                       validation_data=([data_RNA_val,data_Meth_val,data_miRNA_val],y_val)
                       #validation_split=validation_split
                       )

reset_keras()
if True:
    del VAE_classifier
    VAE_classifier=VAE_contructor(encoder, decoder, classifier)
    VAE_classifier.built = True    
    VAE_classifier.load_weights(filepath = checkPath)
    VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))

"""### Fase 3 entrenamiento
En esta fase se entrena el clasificador con el VAE congelado

Se define un nuevo callback sobre el validation accuracy
"""

#------------------------------------------------
# New Early Stopping - Validation Accuracy
#------------------------------------------------

earlyStop_ValAcc = EarlyStopping(monitor='val_class_acc', 
                          mode='min', # indicate that we want to follow decreasing of the metric
                          patience=10, # adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement
                          verbose=1)
#------------------------------------------------
# Callback group 2
#------------------------------------------------
callbacks2=[tensorboard_callback,
           checkpoint,
           earlyStop_ValAcc,
           csv_logger]

"""Se define las capas a entrenar y las que quedan congeladas"""

# Phase 3
#VAE_classifier=VAE_contructor(encoder, decoder, classifier)
#VAE_classifier.built = True

VAE_classifier.classifier.trainable=True
VAE_classifier.encoder.trainable=False
VAE_classifier.decoder.trainable=False

VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))
VAE_classifier.summary()

VAE_classifier.fit(x=[data_RNA_train,data_Meth_train,data_miRNA_train],y=y_train,
                       initial_epoch=0,
                       epochs = 40,
                       verbose = verbosity,
                       batch_size = batch_size,
                       callbacks = callbacks2,
                       validation_data=([data_RNA_val,data_Meth_val,data_miRNA_val],y_val)
                       #validation_split=validation_split
                       )

reset_keras()
if True:
    del VAE_classifier
    VAE_classifier=VAE_contructor(encoder, decoder, classifier)
    VAE_classifier.built = True    
    VAE_classifier.load_weights(filepath = checkPath)
    VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))

"""### Fase 4 entrenamiento
Se entrena el modelo completo pero se coloca las capas BatchNormalization en .trainable=False
"""

K.clear_session()
#VAE_classifier=VAE_contructor(encoder, decoder, classifier)
#VAE_classifier.built = True

trainableStatus(VAE_classifier.classifier,model_Trainable=True,batchNorm_Trainable=False,showPrint=False)
trainableStatus(VAE_classifier.encoder,model_Trainable=True,batchNorm_Trainable=False,showPrint=False)
trainableStatus(VAE_classifier.decoder,model_Trainable=True,batchNorm_Trainable=False,showPrint=False)

#VAE_classifier.load_weights(filepath = checkPath)
VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.00001))
VAE_classifier.summary()

VAE_classifier.fit(x=[data_RNA_train,data_Meth_train,data_miRNA_train],y=y_train,
                       initial_epoch=80,
                       epochs = 10000,
                       verbose = verbosity,
                       batch_size = batch_size,
                       callbacks = callbacks,
                       validation_data=([data_RNA_val,data_Meth_val,data_miRNA_val],y_val)
                       #validation_split=validation_split
                       )

"""Cargo modelo entrenado:"""

# Path to the weights
print("Pesos cargados desde: ",checkPath)

#Creo el modelo
VAE_classifier=VAE_contructor(encoder, decoder, classifier)
VAE_classifier.built = True
VAE_classifier.load_weights(filepath = checkPath)
#Initiate from beginning
VAE_classifier.compile(optimizer=keras.optimizers.Adam(0.0001))

"""En el caso que el entorno se reinicie ejecutar:

## Resumen de la evolución de los entrenamientos
"""

# Cargo el tracker
df = pd.read_csv(logPath)

#Código para corregir si se reinició el entrenamiento (ya que las epocas vuelven a 0)
#df["epoch"]=np.arange(len(df.iloc[:,0]))
#df.to_csv(logPath)
df=df.iloc[20:118,]
df["epoch"]=np.arange(len(df.iloc[:,0]))
# Elimino las primeras 3 filas
#df.drop(index=df.index[0:3], axis=0,inplace=True)
df

# Multiple plots 
# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
# Horizontal graphs
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 16})

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
#fig.suptitle('superVAE Loss Evolution ')

ax1.plot(df["epoch"], df["total_loss"],'tab:red')
ax1.set_title("Total Loss (VAE+classifier)")
ax1.axvline(x=20, color='r', linestyle="--",lw=1)
ax1.axvline(x=40, color='r', linestyle="--",lw=1)
ax1.axvline(x=51, color='r', linestyle="--",lw=1)

ax2.plot(df["epoch"], df["reconstruction_loss"],color='green',linestyle="-",label='mean reconstruction')
ax2.plot(df["epoch"], df["RNA_reconstruction_loss"],color='green',linestyle=":",label='RNA')
ax2.plot(df["epoch"], df["Meth_reconstruction_loss"],color='green',linestyle="--",label='Methylation')
ax2.plot(df["epoch"], df["miRNA_reconstruction_loss"],color='green',linestyle="-.",label='miRNA')
ax2.set_title('Mean Reconstruction Loss')
ax2.axvline(x=20, color='r', linestyle="--",lw=1)
ax2.axvline(x=40, color='r', linestyle="--",lw=1)
ax2.axvline(x=51, color='r', linestyle="--",lw=1)
ax2.legend()

ax3.plot(df["epoch"], df["kl_loss"],'tab:blue')
ax3.set_title('KL Loss')
ax3.axvline(x=20, color='r', linestyle="--",lw=1)
ax3.axvline(x=40, color='r', linestyle="--",lw=1)
ax3.axvline(x=51, color='r', linestyle="--",lw=1)
fig.text(0.5, 0.04, 'Epochs', ha='center')
fig.text(0.04, 0.5, 'Loss value', va='center', rotation='vertical')

matplotlib.rcParams['figure.figsize'] = [20, 10]
plt.rcParams.update({'font.size': 22})
fig, (ax1,ax2) = plt.subplots(1,2)
#fig.suptitle('Classificator evolution')

# Plotting both the curves simultaneously
ax1.plot(df["epoch"], df["val_class_loss"],'tab:blue', label='Validation')
ax1.plot(df["epoch"], df["class_loss"],    color='orange', label='Training')
#ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss value")
ax1.set_title('Categorical Cross-Entropy Loss')
ax1.axvline(x=20, color='r', linestyle="--",lw=1)
ax1.axvline(x=40, color='r', linestyle="--",lw=1)
ax1.axvline(x=51, color='r', linestyle="--",lw=1)

ax2.plot(df["epoch"], df["val_class_acc"],'tab:blue', label='Validation')
ax2.plot(df["epoch"], df["class_acc"],    color='orange', label='Training')
#ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax2.set_title('Accuracy values')
ax2.axvline(x=20, color='r', linestyle="--",lw=1)
ax2.axvline(x=40, color='r', linestyle="--",lw=1)
ax2.axvline(x=51, color='r', linestyle="--",lw=1)
  
# Adding legend, which helps us recognize the curve according to it's color
ax1.legend(loc='upper right')
ax2.legend(loc='lower right')
fig.text(0.5, 0.04, 'Epochs', ha='center')

"""## Reconstructing images and classification:"""

# Contruction of n random balanced samples 
samples=["Primary Tumor","Metastatic","Solid Tissue Normal","Recurrent Tumor"]
index = []
n=2 #2 samples per tumor type
for i, x in enumerate(samples):
    ind=np.where(test_labels==x)[0]
    temp=random.sample(range(0, len(ind)-1),n)
    index.append(ind[temp])
index=np.array(index).reshape(-1)
#test_labels[index]
data_reduced=[data_RNA_test[index,:,:,:],data_Meth_test[index,:,:,:],data_miRNA_test[index,:,:,:]]
labels_reduced=test_labels[index]

test_labels[index]

#Reconstruction of images

#Encoder
z_mean,var_mean=VAE_classifier.encoder.predict(data_reduced)
z_sample=VAE_classifier.sampler(z_mean,var_mean)  

#Decoder
reconstruct = VAE_classifier.decoder.predict(z_sample)

#Classifier
onehot_reconstructed=VAE_classifier.classifier.predict(z_mean)

"""## Reconstructing labels:"""

labels_reduced #observed

# Invierto de oneHot a palabras
labels_reconstructed = label_encoder.inverse_transform(argmax(onehot_reconstructed,axis=1))
labels_reconstructed #predicted
#print(pd.DataFrame(labels_reconstructed).value_counts())

plt.rcParams['figure.figsize'] = [30, 15]
plt.rcParams.update({'font.size': 32})

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
fig.suptitle('Image reconstruction\n of a '+labels_reconstructed[0])

ax1.imshow(np.squeeze(reconstruct[0][0,:,:,:]), cmap="gist_rainbow")
ax1.set_title('RNA')
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax2.imshow(np.squeeze(reconstruct[1][0,:,:,:]), cmap="gist_rainbow")
ax2.set_title('Methylation')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)

ax3.imshow(np.squeeze(reconstruct[2][0,:,:,:]), cmap="gist_rainbow")
ax3.set_title('miRNA')
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

"""## Plot first 10 reconstructions"""

#Ploting n random constructed sample with their originals - RNA
ndim1=ndim2=256
plt.figure(figsize=(30, 10))
plt.rcParams.update({'font.size': 16})

#plt.suptitle('RNA reconstruction',fontsize = 48)
plt.subplots_adjust(top=0.9)
n=8
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data_reduced[0][i].reshape(ndim1, ndim2),
               cmap="gist&2138 & 827 & 2097\\_rainbow",vmin=0, vmax=1)
    #plt.gray()
    #ax.get_xticklabels([])
    #ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel("Original", rotation=90, fontsize=24)
    
    #if i == 7:
        #plt.colorbar()

    plt.title(labels_reduced[i])

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstruct[0][i].reshape(ndim1, ndim2),
               cmap="gist_rainbow",vmin=0, vmax=1)
    #plt.gray()
    ax.set_yticklabels([])
    #ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("Reconstructed/Predicted", rotation=90, fontsize=24)
    plt.title(labels_reconstructed[i])        
plt.show()

#Ploting n random constructed sample with their originals - Methylation
ndim1=ndim2=256
plt.figure(figsize=(30, 10))
plt.rcParams.update({'font.size': 16})

#plt.suptitle('Methylation reconstruction',fontsize = 48)
plt.subplots_adjust(top=0.9)
n=8
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data_reduced[1][i].reshape(ndim1, ndim2),
               cmap="gist_rainbow",vmin=0, vmax=1)
    #plt.gray()
    #ax.get_xticklabels([])
    #ax.get_yaxis().set_vi&2138 & 827 & 2097\\sible(False)
    ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel("Original", rotation=90, fontsize=24)
     #if i == 7:
        #plt.colorbar()

    plt.title(labels_reduced[i])

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstruct[1][i].reshape(ndim1, ndim2),
               cmap="gist_rainbow",vmin=0, vmax=1)
    #plt.gray()
    ax.set_yticklabels([])
    #ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("Reconstructed/Predected", rotation=90, fontsize=24)
    plt.title(labels_reconstructed[i])        
plt.show()

#Ploting n random constructed sample with their originals - miRNA
ndim1=ndim2=64
plt.figure(figsize=(30, 10))
plt.rcParams.update({'font.size': 16})

#plt.suptitle('miRNA reconstruction',fontsize = 48)
plt.subplots_adjust(top=0.9)
n=8
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data_reduced[2][i].reshape(ndim1, ndim2),
               cmap="gist_rainbow",vmin=0, vmax=1)
    #plt.gray()
    #ax.get_xticklabels([])
    #ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel("Original", rotation=90, fontsize=24)
    
    #if i == 7:
        #plt.colorbar()

    plt.title(labels_reduced[i])

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstruct[2][i].reshape(ndim1, ndim2),
               cmap="gist_rainbow",vmin=0, vmax=1)
    #plt.gray()
    ax.set_yticklabels([])
    #ax.get_yaxis().set_visible(False)
    if i == 0:
        ax.set_ylabel("Reconstructed/Predicted", rotation=90, fontsize=24)
    plt.title(labels_reconstructed[i])        
plt.show()

"""## Matrices de confusión"""

#Reconstruction of images
#test data

data_test=[data_RNA_test,data_Meth_test,data_miRNA_test]

#Encoder
z_mean,var_mean=VAE_classifier.encoder.predict(data_test)
z_sample=VAE_classifier.sampler(z_mean,var_mean)  

#Decoder
reconstruct_test = VAE_classifier.decoder.predict(z_sample)

z_sample.shape

# Predigo las etiquetas según los datos a onehot
onehot_pred=VAE_classifier.classifier.predict(z_sample)
print(onehot_pred.shape)

# Invierto de oneHot a palabras
labels_pred = label_encoder.inverse_transform(argmax(onehot_pred,axis=1))
labels_pred.shape

from sklearn.metrics import confusion_matrix

#Get the confusion matrix
cf_matrix = confusion_matrix(test_labels, labels_pred)

print(cf_matrix)
names=np.unique(test_labels)
print(names)

import seaborn as sns

sns.set(font_scale=5) # for label size
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
labels = np.asarray(labels).reshape(4,4)

ax = sns.heatmap(cf_matrix, annot=labels, fmt="",cmap="Blues",cbar=False)
#ax.figure.axes[-1].set_ylabel('Samples', size=10)
#ax.set_title('Test Confusion Matrix with labels\n\n');

ax.set_xlabel('\nPredicted Tumor Type')
ax.set_ylabel('Actual Tumor Type\n');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['M', 'PT', 'RT','Normal'])
ax.yaxis.set_ticklabels(['M', 'PT', 'RT','Normal'])

## Display the visualization of the Confusion Matrix.
plt.show()

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')

ax.set_title('Test Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Tumor Type')
ax.set_ylabel('Actual Tumor Type\n');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['M', 'PT', 'RT','Normal'])
ax.yaxis.set_ticklabels(['M', 'PT', 'RT','Normal'])

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix,roc_curve, auc
print("F1 score:  ",f1_score(test_labels, labels_pred, average="weighted"))
print("Precision: ",precision_score(test_labels, labels_pred, average="weighted"))
print("Recall:    ",recall_score(test_labels, labels_pred, average="weighted"))  
print("Accuracy:  ",accuracy_score(test_labels, labels_pred))

for i,label in enumerate(samples):
    print(label)

"""## Reviso estructura del superVAE

### Estructura - **Encoder**
"""

print("Modelo: "+VAE_classifier.name)
print("  -> capa 1: "+VAE_classifier.layers[0].name)
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[0].name)
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[1].name)
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[2].name)
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[3].name)
print("          ->", VAE_classifier.layers[0].layers[3].summary())
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[4].name)
print("          ->", VAE_classifier.layers[0].layers[4].summary())
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[5].name)
print("          ->", VAE_classifier.layers[0].layers[5].summary())
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[6].name)
print("       -> Submodelo: "+VAE_classifier.layers[0].layers[7].name)

#print("\n *Saving weights of: "+vae.layers[0].layers[1].name)
#vae.layers[0].layers[1].save_weights(saveSubEncoder_weight)

"""### Estructura - **Decoder**"""

print("Modelo: "+VAE_classifier.name)
print("  -> capa 1: "+VAE_classifier.layers[1].name)

print("       -> Submodelo: "+VAE_classifier.layers[1].layers[0].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[1].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[2].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[3].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[4].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[5].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[6].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[7].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[8].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[9].name)
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[10].name)
print("          ->", VAE_classifier.layers[1].layers[10].summary())
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[11].name)
print("          ->", VAE_classifier.layers[1].layers[11].summary())
print("       -> Submodelo: "+VAE_classifier.layers[1].layers[12].name)
print("          ->", VAE_classifier.layers[1].layers[12].summary())

VAE_classifier.layers[1].name

"""## Guardando solamente parte del encoder y decoder para transfer learning"""

#Extraigo subsubmodelo encoder_sub1 y guardo sus pesos
saveSubEncoder_weight="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubEncoder_weight_RNA+class.h5"
check_if_duplicated(saveSubEncoder_weight)
print("Modelo: "+vae.name)
print("  -> capa 1: "+vae.layers[0].name)
print("       -> Submodelo: "+vae.layers[0].layers[0].name)
print("       -> Submodelo: "+vae.layers[0].layers[1].name)
print("       -> Submodelo: "+vae.layers[0].layers[2].name)

print("\n *Saving weights of: "+vae.layers[0].layers[1].name)
vae.layers[0].layers[1].save_weights(saveSubEncoder_weight)

len(VAE_classifier.layers[0].name)

#Extraigo subsubmodelo decoder_sub1 y guardo sus pesos
saveSubEncoder_weight="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubDecoder_weight_RNA+class.h5"
check_if_duplicated(saveSubEncoder_weight)
print("Modelo: "+vae.name)
print("  -> capa 1: "+vae.layers[1].name)
print("       -> Submodelo: "  +vae.layers[1].layers[0].name)
print("       -> Submodelo: "  +vae.layers[1].layers[1].name)
print("       -> Submodelo: "  +vae.layers[1].layers[2].name)

print("\n *Saving weights of: "+vae.layers[1].layers[2].name)
vae.layers[1].layers[2].save_weights(saveSubEncoder_weight)

# Itero sobre las capas de encSub1
print("Layers in submodel encSub1\n")
for layer in vae.layers[0].layers[1].layers:
    #print(layer.name, layer)
    print(layer.name)

# Itero sobre las capas de decSub1
print("Layers in submodel decSub1\n")
for layer in vae.layers[1].layers[2].layers:
    #print(layer.name, layer)
    print(layer.name)

vae.layers[0].layers[1].summary()

vae.layers[1].layers[2].summary()

"""## Preguntas

## Bibliografía

https://towardsdatascience.com/reparameterization-trick-126062cfd3c3

https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0

https://keras.io/guides/writing_a_training_loop_from_scratch/

https://www.tensorflow.org/guide/basics

https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit

https://machinelearningmastery.com/train-test-split-for-evaluating-machine-learning-algorithms/

https://stackoverflow.com/questions/43179429/scikit-learn-error-the-least-populated-class-in-y-has-only-1-member

https://www.tensorflow.org/api_docs/python/tf/split
"""
