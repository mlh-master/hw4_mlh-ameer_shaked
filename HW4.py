# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# + pycharm={"is_executing": false}
import os
import numpy as np
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
# %matplotlib inline
import matplotlib as mpl
from sklearn import metrics
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"

# + pycharm={"is_executing": false}
import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# + pycharm={"is_executing": false}
def preprocess(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn),1)
        img = resize(src,(32,32),order = 3)
        
        images[ii,:,:,0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# -

def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        images[ii,:,:,0] = imread(os.path.join(datapath, fn),1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y

def save_model(model, model_name):
    if not ("results" in os.listdir()):
        os.mkdir("results")
    save_dir = "results/"
    model_name = model_name + ".h5"
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train , BaseY_train = preprocess_train_and_val(train_path)
BaseX_val , BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)

keras.backend.clear_session()

# ### PART 1: Fully connected layers 
# --------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *NN with fully connected layers. 
#
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification. Use ReLU activation functions for the hidden layers and He_normal for initialization. Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*
#
# ---

# +
#--------------------------Impelment your code here:-------------------------------------
def create_fc_model(input_shape=(32, 32, 1), act_hidden="relu", act_out='softmax', initializer=keras.initializers.he_normal(seed=None), batch_norm=False):
    # define the model
    fc_model = Sequential()
    # define an input layer
    #fc_model.add(InputLayer(input_shape=input_shape))
    #model_relu.add(InputLayer())
    # flatten the input image
    fc_model.add(Flatten(input_shape=input_shape,name="Flatten_input"))
    # add 2 layers with 300 and 150 neurons respectively

    fc_model.add(Dense(300, input_shape=(1024,), kernel_initializer=initializer))
    if (act_hidden=="relu"):
        fc_model.add(tf.keras.layers.ReLU(name="Hidden_1"))
    elif (act_hidden=="leaky_relu"):
        fc_model.add(tf.keras.layers.LeakyReLU(name="Hidden_1"))

    if (batch_norm==True):
        fc_model.add(tf.keras.layers.BatchNormalization(name="Batch_normalizaion1"))


    fc_model.add(Dense(150, input_shape=(300,), kernel_initializer=initializer))
    if (act_hidden=="relu"):
        fc_model.add(tf.keras.layers.ReLU(name="Hidden_2"))
    elif (act_hidden=="leaky_relu"):
        fc_model.add(tf.keras.layers.LeakyReLU(name="Hidden_2"))

    if (batch_norm == True):
        fc_model.add(tf.keras.layers.BatchNormalization(name="Batch_normalizaion2"))

    # add the labels layer - 4 categories
    fc_model.add(Dense(4, input_shape=(150,), activation=act_out, name="Output_Layer"))
#    fc_model.compile(optimizer=optimizer, metrics=['accuracy'], loss='categorical_crossentropy')
    return fc_model

he_init = keras.initializers.he_normal(seed=None)
model_relu = create_fc_model(initializer=he_init)
#----------------------------------------------------------------------------------------
# -
model_relu.summary()
# +
#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# -

# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task. Train your model on the training set and evaluate the model on the testing set. Print the accuracy and loss over the testing set.

# +
#--------------------------Impelment your code here:-------------------------------------
# let us compile the classifier using the parameters given above
#model_relu = KerasClassifier(build_fn=create_fc_model, verbose=2, epochs=epochs, batch_size=batch_size,
#                             initializer=relu_init, optimizer=AdamOpt, input_shape=input_shape)

model_relu = create_fc_model(initializer=he_init, input_shape=input_shape)
model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# save the model for further training
save_model(model_relu, "model_relu")
# fit the training set
history = model_relu.fit(BaseX_train, BaseY_train, epochs=25, batch_size=64, verbose=2, validation_data=(BaseX_val, BaseY_val))
# evaluate the model over the test set
#Y_test_pred = model_relu.predict(X_test)
relu_test_res = model_relu.evaluate(X_test, Y_test)
print('\n Model_relu (principal): \n Test loss: %.3f , Test ACC: %.3f\n' % (relu_test_res[0], relu_test_res[1]))

#----------------------------------------------------------------------------------------
# -

# ---
# <span style="color:red">***Task 2:***</span> *Activation functions.* 
#
# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*
#
# ---

# +
#--------------------------Impelment your code here:-------------------------------------
#new_a_model = create_fc_model(act_hidden=tf.nn.leaky_relu, optimizer=AdamOpt, input_shape=input_shape)
new_a_model = create_fc_model(act_hidden="leaky_relu", input_shape=input_shape)
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

save_model(new_a_model, "new_a_model")
#----------------------------------------------------------------------------------------
# -

new_a_model.summary()

# ---
# <span style="color:red">***Task 3:***</span> *Number of epochs.* 
#
# Train the new model using 25 and 40 epochs. What difference does it makes in term of performance? Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12. Evaluate each trained model on the test set*
#
# ---

# +
#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model_25 = load_model("results/new_a_model.h5")
history25 = new_a_model_25.fit(BaseX_train, BaseY_train, epochs=epochs, batch_size=64,
                                verbose=2, validation_data=(BaseX_val, BaseY_val))
new_a_25_res = new_a_model_25.evaluate(X_test, Y_test)
print('\n new_a_model with 25 epochs performance: \n Test loss: %.3f , Test ACC: %.3f\n' % (new_a_25_res[0], new_a_25_res[1]))
#-----------------------------------------------------------------------------------------

# +
#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)



# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model_40 = load_model("results/new_a_model.h5")
history40 = new_a_model_40.fit(BaseX_train, BaseY_train, epochs=epochs, batch_size=64,
                               verbose=2, validation_data=(BaseX_val, BaseY_val))
new_a_40_res = new_a_model_40.evaluate(X_test, Y_test)
print('\n new_a_model with 40 epochs performance: \n Test loss: %.3f , Test ACC: %.3f\n' % (new_a_40_res[0], new_a_40_res[1]))
#-----------------------------------------------------------------------------------------
# -

# ---
# <span style="color:red">***Task 4:***</span> *Mini-batches.* 
#
# Build the `model_relu` again and run it with a batch size of 32 instead of 64. What are the advantages of the mini-batch vs. SGD?*
#
# ---

keras.backend.clear_session()

# +
#--------------------------Impelment your code here:-------------------------------------

model_relu = create_fc_model(initializer=he_init, input_shape=input_shape)

#----------------------------------------------------------------------------------------

# +
batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# +
#--------------------------Impelment your code here:-------------------------------------

model_relu.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')

# save_model(model_relu, "model_relu")
# fit the training set
history_batch32 = model_relu.fit(BaseX_train, BaseY_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(BaseX_val, BaseY_val))
# evaluate the model over the test set
#Y_test_pred = model_relu.predict(X_test)
relu_test_res = model_relu.evaluate(X_test, Y_test)
print('\n Model_relu with mini-batch of size 32 performance: \n Test loss: %.3f , Test ACC: %.3f\n' % (relu_test_res[0], relu_test_res[1]))

#----------------------------------------------------------------------------------------
# -

# ---
# <span style="color:red">***Task 4:***</span> *Batch normalization.* 
#
# Build the `new_a_model` again and add batch normalization layers. How does it impact your results?*
#
# ---

keras.backend.clear_session()

# +
#--------------------------Impelment your code here:-------------------------------------
new_a_model = create_fc_model(act_hidden="leaky_relu", input_shape=input_shape, batch_norm=True)

#---------------------------------------------------------------------------------------

# +
batch_size = 64
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
#Compile the network: 


# +
#Preforming the training by using fit 
#--------------------------Impelment your code here:-------------------------------------
new_a_model.compile(optimizer=AdamOpt, metrics=['accuracy'], loss='categorical_crossentropy')
history_batch32 = new_a_model.fit(BaseX_train, BaseY_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(BaseX_val, BaseY_val))

leakyrelu_test_res = new_a_model.evaluate(X_test, Y_test)
print('\n Model- leakyrelu with mini-batch of size 32 and with batch normalizaion performance: \n Test loss: %.3f , Test ACC: %.3f\n ' % (leakyrelu_test_res[0], leakyrelu_test_res[1]))

#----------------------------------------------------------------------------------------
# -

# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *2D CNN.* 
#
# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?
#
# ---

def get_net(input_shape,drop,dropRate,reg,filters_num):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=filters_num[0], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters_num[1], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:    
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filters_num[2], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters_num[3], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filters_num[4], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:      
    model.add(Dense(512, activation='elu',name='FCN_1')) 
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model


input_shape = (32,32,1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
# a new parameters list - change was required at the last section
filters_num = [64,128,128,256,256]

NNet = get_net(input_shape,drop,dropRate,reg,filters_num=filters_num)

# + pycharm={"is_executing": false}
# NNet = get_net(input_shape,drop,dropRate,reg)

# + pycharm={"is_executing": false}
from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network: 
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
#Checkpath = os.getcwd()
#Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)
# -

#Preforming the training by using fit 
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)

# + pycharm={"is_executing": false}
# NNet.load_weights('Weights_1.h5')

# + pycharm={"is_executing": false}
results = NNet.evaluate(X_test,Y_test)
print('\n the performance of the principal CNN: \n test loss = %.2f test acc = %.2f \n '%( results[0], results[1]))
# -

# ---
# <span style="color:red">***Task 2:***</span> *Number of filters* 
#
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers, i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`. Now train the model with the number of filters reduced by half. What were the results.
#
# ---

# +
#--------------------------Impelment your code here:-------------------------------------
filters_num = [32,64,64,128,128]

NNet = get_net(input_shape,drop,dropRate,reg,filters_num=filters_num)

#Compile the network:
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
#Checkpath = os.getcwd()
#Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)
# -

#Preforming the training by using fit
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)

# + pycharm={"is_executing": false}
# NNet.load_weights('Weights_1.h5')

# + pycharm={"is_executing": false}
results = NNet.evaluate(X_test,Y_test)
print('\n the performance of CNN with number of filters reduced by half: \n test loss = %.2f test acc = %.2f \n '%( results[0], results[1]))
#----------------------------------------------------------------------------------------
# -

# That's all folks! See you :)
