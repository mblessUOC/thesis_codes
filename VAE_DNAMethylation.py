**Author:** Maximiliano Bless<br>
**Date created:** 2022/05/14<br>
**Last modified:** 2022/05/14<br>
**Description:** Convolutional Variational AutoEncoder (VAE) to trained on PCA images from methylation dataset that are single.
Image dimension 256x256.
Class categories: 4

Convolutional Kernel sizes: 32-64-128.

Latent dimension: 256

**Conclusion**:
*Epoch 129 - loss_vae: 21843.6050 - reconstruction_loss: 21972.6367 - kl_loss: 8.7992*

**Resumen**:

En este código se entreó el VAE de expresion con los datos individuales.
"""

from google.colab import drive
drive.mount('/content/drive')

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

Meth_encSub1_inputs = keras.Input(shape=(img_dim, img_dim, 1), name="x_Meth")

Meth_encSub1 = layers.Conv2D(kernel_sizes[0], 3, strides=1, padding="same",use_bias=use_bias, name="Meth_encSub1_conv2D.1.0")(Meth_encSub1_inputs)
Meth_encSub1 = layers.BatchNormalization(name="Meth_encSub1_conv2D.1.2")(Meth_encSub1)
Meth_encSub1 = layers.LeakyReLU(name="Meth_encSub1_conv2D.1.3")(Meth_encSub1)

Meth_encSub1 = layers.MaxPool2D(pool_size=down_ratio, name="Meth_encSub1_MaxPool.1")(Meth_encSub1) # MaxPool -1-

Meth_encSub1 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=use_bias, name="Meth_encSub1_conv2D.2.0")(Meth_encSub1)
Meth_encSub1 = layers.BatchNormalization(name="Meth_encSub1_conv2D.2.1")(Meth_encSub1)
Meth_encSub1 = layers.LeakyReLU(name="Meth_encSub1_conv2D.2.2")(Meth_encSub1)

Meth_encSub1_outputs = layers.MaxPool2D(pool_size=down_ratio,name="Meth_encSub1_MaxPool.2")(Meth_encSub1) # MaxPool -2-
 
# Genero submodelo del encoder que utilizaré para transfer learning
Meth_encoder_sub1 = keras.Model(Meth_encSub1_inputs, Meth_encSub1_outputs, name="encoder_sub1")
print(Meth_encoder_sub1.summary())
keras.utils.plot_model(Meth_encoder_sub1, show_shapes = True)

"""### Build the encoder with all the subparts - **latent space**"""

Meth_inputs = keras.Input(shape=(256, 256, 1),name="x_Meth_input")
encSub2_inputs = Meth_encoder_sub1(Meth_inputs)

encSub2 = layers.Conv2D(kernel_sizes[2], down_ratio, strides=1, padding="same",use_bias=use_bias, name="encSub2_conv2D.1.0")(encSub2_inputs)
encSub2 = layers.BatchNormalization(name="encSub2_conv2D.1.1")(encSub2)
encSub2 = layers.LeakyReLU(name="encSub2_conv2D.1.2")(encSub2)

encSub2 = layers.MaxPool2D(pool_size=down_ratio,name="encSub2_MaxPool.1")(encSub2) # MaxPool -3-

encSub2 = layers.Flatten(name="encSub2_Flatten")(encSub2)

z_mean = layers.Dense(latent_dim, name="z_mean")(encSub2)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encSub2)

encoder = keras.Model(Meth_inputs, [z_mean,z_log_var], name="encoder")
print(encoder.summary())
keras.utils.plot_model(encoder, show_shapes = True)

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

# Genero resto del submodelo del decoder que no se transferira

z_sample = keras.Input(shape=(latent_dim,),name="z_sample")

decSub2 = layers.Dense(up_ratio * up_ratio * kernel_sizes[2], activation="relu",name="decSub2_Dense")(z_sample)

decSub2 = layers.Reshape((up_ratio, up_ratio, kernel_sizes[2]),name="decSub2_Reshape")(decSub2)

decSub2 = layers.Conv2DTranspose(kernel_sizes[2], 3, activation="relu", strides=up_ratio, padding="same",use_bias=False,name="decSub2_convTrans2D.1.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_convTrans2D.1.1")(decSub2)
decSub2 = layers.LeakyReLU(name="decSub2_convTrans2D.1.2")(decSub2)

decSub2 = layers.Conv2D(kernel_sizes[1], 3, strides=1, padding="same",use_bias=False,name="decSub2_conv2D.2.0")(decSub2)
decSub2 = layers.BatchNormalization(name="decSub2_conv2D.2.1")(decSub2)
decSub2_outputs = layers.LeakyReLU(name="decSub2_conv2D.2.2")(decSub2)
Meth_decoder_sub1_output = decoder_sub1(decSub2_outputs)

decoder = keras.Model(z_sample, Meth_decoder_sub1_output, name="decoder")
print(decoder.summary())
keras.utils.plot_model(decoder, show_shapes = True)

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
#!rm -rf '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/model.V6.1_Meth.h5'
!rm -rf "logs"
#!rm -rf '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/VAE_V6/training.V6.1_Meth.csv'

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
checkPath = '/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/model.V6.1_Meth.h5'
checkpoint = ModelCheckpoint(filepath=checkPath, 
                             monitor='loss_vae',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True, #only weights because it is a subModel
                             mode='min')

#------------------------------------------------
# Training logger
#------------------------------------------------
logPath='/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Run/VAE_V6/training.V6.1_Meth.csv'
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
list_of_data = sorted( filter( os.path.isfile,
                        glob.glob(relevant_path + 'dataMeth_*') ) )

# Listado de metadatos de expresión
list_of_metadata = sorted( filter( os.path.isfile,
                        glob.glob(relevant_path + 'metadataMeth_*') ) )


print("Length of uploaded data:",len(list_of_data))
print("Length of uploaded metadata:",len(list_of_metadata))

# remove file in folder data
#!rm -rf "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/"
#!mkdir "/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data"

# Loop donde cargo archivo por archivo y los apilo uno arriba del otro

from numpy import array, zeros, newaxis
data_array=np.zeros((256,256,0))
metadata_array=np.zeros((0,28))

for file_data,file_metadata in zip(list_of_data,list_of_metadata):
    tempData=np.load(file_data,allow_pickle=True)
    if len(tempData.shape) == 2: #si solo hay una muestra, la dimensión del dato es 2D, con newaxis transformo a 3D
        tempData=tempData[:,:,newaxis]
    #print("temp shape:",tempData.shape)

    data_array  = np.append(data_array,tempData,axis=2)
    #print("data_array shape:",data_array.shape)
    
    tempMeta=np.load(file_metadata,allow_pickle=True)
    #print("temp shape:",temp.shape)
    #print("metadata_array shape:",metadata_array.shape)
    metadata_array = np.append(metadata_array,tempMeta,axis=0)

print("Final shape of data_array:",data_array.shape)
print("Final shape of metadata_array:",metadata_array.shape)

# Reviso de las frecuencias de cada tipo de tumor
# en metadata para methilación:
#  - tumor: posición 17
#  - tipo muestra: posición 24  
import pandas as pd
unique, counts = np.unique(metadata_array[:,24], return_counts=True)
sample=np.asarray((unique, counts))
#for i in range(sample.shape[1]):
#    print(sample[:,i])

df = pd.DataFrame([counts],columns = unique)
#df
df_labels=pd.DataFrame(metadata_array[:,24])
df_labels.columns=['labels']
print(df_labels.value_counts())
df_labels.value_counts().to_csv('/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/data/other/sampleFrequency_VAE_V6.1_Meth.csv', index=True, header=True)

#Código en donde extraigo el índice de las tipo de muestras deseadas
samples=["Primary Tumor","Metastatic","Solid Tissue Normal","Recurrent Tumor"]
sample_index = []
for i, x in enumerate(metadata_array[:,24]):
    if any(x == c for c in samples):
        sample_index.append(i)
print(pd.DataFrame(metadata_array[:,24][sample_index]).value_counts())
sample_label=metadata_array[:,24][sample_index]

#Convert to one-hot encoding

# libraries
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# define example
labels = metadata_array[:,24]
labels = array(labels)
print("labels:",labels[0:10])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print("integer encoded:",integer_encoded[0:10])
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print("onehot:", onehot_encoded)
print("onehot shape:", onehot_encoded.shape)

# Keras necesita tensores de entrada de la siguiente estructura:
# (batch, altura imagen, anchura imagen, filtros)
data = np.swapaxes(data_array[:,:,sample_index],0,2) #función que cambio la dimensión 0 por la 2
#BCRA_img = np.swapaxes(BCRA_img,1,2) #función que cambio la dimensión 1 por la 2

#Normalizo las expresion de [0,1]
maxValue=np.max(data).astype("float64")

#Agrego la dimension de los filtros
data_train = np.expand_dims(data, -1).astype("float64") / maxValue
#data_train = min_max_range(np.expand_dims(data, -1).astype("float64"))
#Corroboro dimensiones del input
data_train.shape

#sin escalamiento min/max
if False:
    data = np.swapaxes(data_array,0,2) #función que cambio la dimensión 0 por la 2
    data_train = np.expand_dims(data, -1).astype("float64")
    data_train.shape

# Reviso valores maximos y minimos
print(np.max(data_train))
print(np.min(data_train))

"""##Plot"""

#Genero gráfico para 10 muestras aleatorias

import matplotlib.pyplot as plt
from matplotlib import colors
import random
plt.figure(figsize=(30, 15))
plt.rcParams.update({'font.size': 16})

n=10
image_index = random.sample(range(0,data_train.shape[0]),n)

# make a color map of fixed colors
cmap = colors.ListedColormap(['white', 'red'])


n = 5  # How many digits we will display per row
for i,index in enumerate(image_index):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data_train[index].reshape(img_dim, img_dim),cmap='gist_rainbow')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.title(metadata_array[index,24]+'\n'+metadata_array[index,17])

    
plt.show()

"""## Train the VAE"""

# Resumen de parametros del enconder y decoder 
print(encoder.summary())
print(decoder.summary())

# Commented out IPython magic to ensure Python compatibility.
# Abrir ventana de Tensorboard
# %tensorboard --logdir logs

#Configuro el entrenamiento
EPOCH = 2000
batch_size = 128


#Creo el modelo
VAE_Meth=VAE(encoder, decoder)

# Path to the weights

#load from last checkpoint
#path_lastWeights=checkPath
#if os.path.exists(path_lastWeights):
#    vae.built = True
#    vae.load_weights(filepath = path_lastWeights)

#Initiate from beginning
VAE_Meth.compile(optimizer=keras.optimizers.Adam(0.0001))
VAE_Meth.metrics_names
VAE_Meth.fit(data_train,
        epochs=EPOCH,
        batch_size = batch_size,
        callbacks = callbacks
        )

"""## Resumen de la evolución de los entrenamientos"""

# Cargo el tracker
df = pd.read_csv(logPath)

#Código para corregir si se reinició el entrenamiento (ya que las epocas vuelven a 0)
#df["epoch"]=np.arange(len(df.iloc[:,0]))
#df.to_csv(logPath)
df

# Multiple plots 
# https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
# Horizontal graphs
#Horizontal graphs
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

"""## Cargo modelo entrenado:"""

# Path to the weights
print("Pesos cargados del siguiente archivo: ",checkPath)
#Creo el modelo
VAE_Meth=VAE_Meth(encoder, decoder)
VAE_Meth.built = True
VAE_Meth.load_weights(filepath = checkPath)
#Initiate from beginning
VAE_Meth.compile(optimizer=keras.optimizers.Adam(0.0001))

"""## Reconstructing images:"""

# Contruction of n random samples
# How many digits we will display/reconstruct
n = 10  
# Generate random samples according n
image_index = random.sample(range(0,data_train.shape[0]),n)

data_reduced=data_train[image_index,:,:,:]
z_mean,var_mean=VAE_Meth.encoder.predict(data_reduced)
z_sample=VAE_Meth.sampler(z_mean,var_mean)  
reconstruct = VAE_Meth.decoder.predict(z_sample)

plt.imshow(np.squeeze(reconstruct[0]), cmap="gist_rainbow")
plt.axis('off')
plt.title(metadata_array[image_index[0],24]+'\n'+metadata_array[image_index[0],17])

"""## Plot first 10 reconstructions"""

#Ploting n random constructed sample with their originals
plt.figure(figsize=(30, 15))
plt.rcParams.update({'font.size': 24})

plt.suptitle('Methylation reconstruction',fontsize = 48)
plt.subplots_adjust(top=0.9)
n=5
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(data_reduced[i].reshape(img_dim, img_dim),
               cmap="gist_rainbow")
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])
    if i == 0:
        ax.set_ylabel("Original", rotation=90, fontsize=24)
    plt.title(metadata_array[image_index[i],24]+'\n'+metadata_array[image_index[i],17])

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstruct[i].reshape(img_dim, img_dim),
               cmap="gist_rainbow")
    #plt.gray()
    if i == 0:
        ax.set_ylabel("Reconstructed", rotation=90, fontsize=24)
    ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    ax.set_yticklabels([])

plt.show()

"""## Guardando solamente parte del encoder y decoder para transfer learning"""

# Itero sobre las capas de Encoder y Decoder
print("Layers in VAE_Meth\n")
for i,layer in enumerate(VAE_Meth.layers):
    #print(layer.name, layer)
    print(i,layer.name)
print("\n*--------------------------*\n")
print("Layers in submodel Encoder\n")
for i,layer in enumerate(VAE_Meth.layers[0].layers):
    #print(layer.name, layer)
    print(i,layer.name)
print("\n*--------------------------*\n")
# Itero sobre las capas de decSub1
print("Layers in submodel Decoder\n")
for i,layer in enumerate(VAE_Meth.layers[1].layers):
    #print(layer.name, layer)
    print(i,layer.name)

#Extraigo subsubmodelo encoder_sub1 y guardo sus pesos
saveSubEncoder_weight="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubEncoder_weight_Meth_V6.1.h5"
#check_if_duplicated(saveSubEncoder_weight)
print("Modelo: "+VAE_Meth.name)
print("  -> capa 0: "+VAE_Meth.layers[0].name)
for i,layer in enumerate(VAE_Meth.layers[0].layers):
    print("       ->",i,"Submodelo:",layer.name)

print("\n *Saving weights of: "+VAE_Meth.layers[0].layers[1].name)
#VAE_Meth.layers[0].layers[1].save_weights(saveSubEncoder_weight)

#Extraigo subsubmodelo decoder_sub1 y guardo sus pesos
saveSubEncoder_weight="/content/drive/MyDrive/Maxi Bless/VAEs/Datos_Colab/Weights/VAE_V6/SubDecoder_weight_Meth_V6.1.h5"
#check_if_duplicated(saveSubEncoder_weight)
print("Modelo: "+VAE_Meth.name)
print("  -> capa 1: "+VAE_Meth.layers[1].name)
for i,layer in enumerate(VAE_Meth.layers[1].layers):
    print("       ->",i,"Submodelo:",layer.name)


print("\n *Saving weights of: "+VAE_Meth.layers[1].layers[9].name)
VAE_Meth.layers[1].layers[9].save_weights(saveSubEncoder_weight)

VAE_Meth.layers[0].layers[1].summary()
VAE_Meth.layers[1].layers[9].summary()

"""## Preguntas

## Bibliografía

https://towardsdatascience.com/reparameterization-trick-126062cfd3c3

https://towardsdatascience.com/variational-autoencoders-as-generative-models-with-keras-e0

https://keras.io/guides/writing_a_training_loop_from_scratch/

https://www.tensorflow.org/guide/basics

https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
"""
