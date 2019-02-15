
####################################################################################################
#### Some imports
####################################################################################################

import pickle
import random
import numpy as np
from numpy.linalg import norm as n
import scipy
from random import shuffle
from sklearn.model_selection import train_test_split
import time
import tables as tb   ### for hdf5 file
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import os
from os.path import isfile
import shutil
import hashlib
from sys import platform
import json
from tensorflow.python import debug as tf_debug
from keras.models import Model, load_model
from keras.layers import UpSampling1D,Add,Masking,Flatten,Concatenate,LeakyReLU,PReLU,Input,LSTM, core, Bidirectional, CuDNNLSTM, CuDNNGRU,Reshape,Lambda,Permute,TimeDistributed,RepeatVector,ConvLSTM2D,Conv3D,Dense,UpSampling3D,Embedding, SpatialDropout1D,GRU,Add,Activation,multiply
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose,UpSampling2D,AtrousConvolution2D,Conv1D,SeparableConv2D,SeparableConv1D
from keras.layers.pooling import MaxPooling2D,MaxPooling3D,MaxPooling1D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.utils import to_categorical
from keras import backend as K
from keras.engine.topology import Layer
from keras import losses
from keras import initializers
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import load_model as LM
from keras.preprocessing import text, sequence
from keras.backend import squeeze,sum
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.constraints import maxnorm

#################################################################################
#### version for single output
#################################################################################
#################################################################################################################################################
### def IGLOO1D
#################################################################################################################################################

'''

# Advanced Usage


IGLOO1D(input_layer,nb_patches,nb_filters_conv1d,return_sequences,patch_size=4,
        padding_style="causal",stretch_factor=1,nb_stacks=1,l2reg=0.00001,conv1d_kernel=3,
        max_pooling_kernel=1,DR=0.0,add_residual=True,nb_sequences=-1,build_backbone=False,psy=0.15)

input_layer:        Keras layer used as input
nb_patches:         number L of patches taken at random from the feature space. This is the main dimension parameter.
                    a fair value to use is around the number of steps in the sequence to study. This can vary depending
                    on the task
nb_filters_conv1d:  Size of the internal convolution K which transforms the input_layer
return_sequences:   False to return only the full sequence representation
patch_size:         This is the number of slices taken to form a patch .Typical value is 4. Adding more increases fitting
                    and adds the number of parameters.
padding_style:      "causal", "same"
stretch_factor:     Stretch factor from the paper. When returning a full sequence this allows to reuse weights. The stretch
                    factor should divide the number of steps exactly. A value around 10-20 usualy works and allows to divide
                    the number of parameters by as much.
nb_stacks:          number of levels of granularity that IGLOO will consider. More stacks increases accuracy but also the number
                    of parameters. Most of the time 1 stack should be enough, unless the number of paramaters is low to start with.
                    Setting more than 1 stack should be used only when return_sequences=False, otherwise the number of paramters
                    will turn out to be too large.
l2_reg:             L2 regularization factor. Can use a value around 0.1 to combat over fitting.
conv1d_kernel       Kernel of the initial convolution. Few reasons to change that.
max_pooling_kernel  Only when return_sequences=False, this allows to reduce the number of steps and therefore the number of paramters.
                    Some tasks work well with this. A typical value can be 2 to6.
DR                  Dropout rate.
add_residual        If return_sequences=True, this improves convergence and generally should be set to True. If return_sequences=False,
                    this does nothing.
nb_sequences        If return_sequences=True, this allows to return the last "nb_sequences" steps of the sequences
                    (For the copy memory task for example).
build_backbone      When the number of patches is large, IGLOO can use some patches arranged in a non random way to better cover the
                    input space. This is set to True by default for return_sequences=False and to False for  return_sequences=True.
                    If set to True, then nb_steps/3 patches are required as a minimum.
psy                 When return_sequences=True, this is the proportion of Local Patches (as per paper description) with respect to the
                    number L of global patches. A typical value is 0.15.


'''

def IGLOO1D(input_layer,nb_patches,nb_filters_conv1d,return_sequences,patch_size=4,padding_style="causal",stretch_factor=1,nb_stacks=1,l2reg=0.00001,conv1d_kernel=3,max_pooling_kernel=1,DR=0.0,add_residual=True,nb_sequences=-1,build_backbone=False,psy=0.15):

    if return_sequences==True:

        M = IGLOO_RETURNFULLSEQ(input_layer,nb_patches,nb_filters_conv1d,patch_size=patch_size,padding_style=padding_style,stretch_factor=stretch_factor,nb_stacks=nb_stacks,l2reg=l2reg,conv1d_kernel=conv1d_kernel,DR=DR,add_residual=add_residual,nb_sequences=nb_sequences,build_backbone=False,psy=psy)

    else:

        M = IGLOO(input_layer,nb_patches,nb_filters_conv1d,patch_size=patch_size,padding_style=padding_style,nb_stacks=nb_stacks,l2reg=l2reg,conv1d_kernel=conv1d_kernel,max_pooling_kernel=max_pooling_kernel,DR=DR,build_backbone=True)

    return M


#################################################################################
#### version for single output
#################################################################################
def IGLOO(input_layer,nb_patches,nb_filters_conv1d,return_sequences=False,patch_size=4,padding_style="causal",nb_stacks=1,l2reg=0.00001,conv1d_kernel=3,max_pooling_kernel=1,DR=0.0,build_backbone=True):

    LAYERS=[]

#    print("return sequences;",return_sequences)


    if return_sequences and nb_sequences==1:
        print("cannot have return sequences and slice last ==1 at the same time")
        nb_sequences=0


    if return_sequences and max_pooling_kernel>1:
        print("When generating sequences rather than representation, pooling cannot be used.")
        sys.exit()


    x = Conv1D(nb_filters_conv1d, conv1d_kernel , padding=padding_style) (input_layer)
#        x = BatchNormalization(axis=-1) (x)
    x = LeakyReLU(alpha=0.1) (x)
    x = SpatialDropout1D(DR)(x)
    x = MaxPooling1D(pool_size=max_pooling_kernel, strides=None, padding="valid") (x)      ### cah nge to 3
    LAYERS.append( PATCHY_LAYER_CNNTOP_LAST(patch_size,nb_patches,DR,l2reg=l2reg) (x) )


    if nb_stacks>1:

        for extra_l in range(nb_stacks-1):

            ###create a new layer for a different granularity view
            x = Conv1D(nb_filters_conv1d, conv1d_kernel , padding=padding_style, dilation_rate=3) (x)       ####this conv is dilated
#                x = BatchNormalization(axis=-1) (x)
            x = LeakyReLU(alpha=0.1) (x)
            x = SpatialDropout1D(DR)(x)
#            print("x after max pooling",x)

            LAYERS.append( PATCHY_LAYER_CNNTOP_LAST(patch_size,nb_patches,DR,l2reg=l2reg) (x) )

    if nb_stacks>1:

        MPI=Concatenate()(LAYERS)

    else:
        MPI=LAYERS[0]


    return MPI


#################################################################################
#### IGLOO_RETURNFULLSEQ
#################################################################################
def IGLOO_RETURNFULLSEQ(input_layer,nb_patches,nb_filters_conv1d,patch_size=4,padding_style="causal",stretch_factor=1,nb_stacks=1,l2reg=0.00001,conv1d_kernel=3,DR=0.0,add_residual=True,
                        nb_sequences=-1,build_backbone=False,psy=0.15):

    LAYERS=[]

    return_sequences=True

    assert not (nb_sequences>0 and stretch_factor>1), "Cannot use the stretch factor when returning a partial sequence"

    if nb_sequences>0 and add_residual==True:
        add_residual=False
        print("Cannot have a residual when returning a partial sequence")




    x = Conv1D(nb_filters_conv1d, conv1d_kernel , padding=padding_style) (input_layer)
#        x = BatchNormalization(axis=-1) (x)    ###
    x = LeakyReLU(alpha=0.1) (x)
    x = SpatialDropout1D(DR)(x)


    LAYERS.append( PATCHY_LAYER_RETURNFULLSEQ(patch_size,nb_patches,DR=DR,stretch_factor=stretch_factor,add_residual=add_residual,l2reg=l2reg,nb_sequences=nb_sequences,build_backbone=build_backbone,psy=psy) (x) )

    if nb_stacks>1:

        for extra_l in range(nb_stacks-1):

            x = Conv1D(nb_filters_conv1d, conv1d_kernel , padding=padding_style,dilation_rate=3) (input_layer)
#                x = BatchNormalization(axis=-1) (x)
            x = LeakyReLU(alpha=0.1) (x)
            x = SpatialDropout1D(DR)(x)
            LAYERS.append( PATCHY_LAYER_RETURNFULLSEQ(patch_size,nb_patches,DR=DR,stretch_factor=stretch_factor,add_residual=add_residual,l2reg=l2reg,nb_sequences=nb_sequences,build_backbone=build_backbone,psy=psy) (x) )


    if len(LAYERS)>1:
        return Concatenate()(LAYERS)
    else:
        return LAYERS[0]


#################################################################################################################################################
### PATCHY_LAYER_CNNTOP_LAST
#################################################################################################################################################


class PATCHY_LAYER_CNNTOP_LAST(Layer):

    def __init__(self,patch_size,nb_patches,DR,initializer='glorot_normal',l2reg=0.000001,activation='relu',**kwargs):

        ### add support for masking
        self.supports_masking = True
        self.nb_patches=nb_patches

        self.patch_size=patch_size
        self.DR=DR

        self.initializer=initializer
        self.kernel_causal=4
        self.activation=activation

        self.l2reg=l2reg
        self.outsize=100


        super(PATCHY_LAYER_CNNTOP_LAST, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        if input_mask is not None:
            return input_mask
        else:
            return None

    def PATCHY_LAYER_CNNTOP_LAST_initializer(self,shape, dtype=None):

        M=gen_filters_igloo_Nsteps_New(self.patch_size,self.nb_patches,self.vector_size,self.num_channels_input,return_sequences=False)
        M.astype(int)
#        print(" DTYPE!!!",M.dtype)

        return M


    def build(self, input_shape):



        self.batch_size=input_shape[0]
        self.vector_size=input_shape[1]
        self.num_channels_input=input_shape[2]



        self.mshapy=(int(self.nb_patches),int(self.patch_size*self.num_channels_input),2)

        ###SETTING THIS AS A NON TRAINABLE VARIABLE SO IT CAN BE SAVED ALONG THE WEIGHTS
        self.patches=self.add_weight(shape=(int(self.nb_patches),int(self.patch_size*self.num_channels_input),2),
                                    initializer=self.PATCHY_LAYER_CNNTOP_LAST_initializer,
                                    trainable=False,
                                    name="random_patches",dtype=np.int32)


        self.W_MULT = self.add_weight(shape=(1,int(self.nb_patches/1),self.patch_size*self.num_channels_input),
                                    initializer=self.initializer,
                                    trainable=True,
                                    regularizer=l2(self.l2reg),
                                    name="W_MULT")

        self.W_BIAS = self.add_weight(shape=(1,int(self.nb_patches/1)),
                                    initializer=self.initializer,
                                    trainable=True,
                                    regularizer=l2(self.l2reg),
                                    name="W_BIAS")



        super(PATCHY_LAYER_CNNTOP_LAST, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, y, mask=None):


        PATCH_tensor1=Patchy_nonzero_1D_lessD(y,self.patches,self.nb_patches,self.patch_size)

        W_MULT=tf.tile(self.W_MULT,[tf.shape(y)[0],1,1])

        MPI=tf.multiply(W_MULT,PATCH_tensor1)

        MPI=tf.reduce_sum(MPI, axis=-1)


        Bias=tf.tile(self.W_BIAS,[tf.shape(y)[0],1])

        MPI=tf.add(MPI,Bias)

        MPI=LeakyReLU(alpha=0.1) (MPI)

        MPI=Dropout(self.DR) (MPI)


        return MPI


    def compute_output_shape(self, input_shape):


        return input_shape[0],self.nb_patches


################################################################################
#### Training
################################################################################
class ADDITION_Evaluation(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()


        first,second = validation_data        ###  (34, 256, 256, 3) ,  (34, 256*256, 3)

        self.Q_TEST=first
        self.Q_LABELS=second

        self.timestamp=[]



    def on_batch_end(self,batch, logs={}):


        if batch==0:

            self.timestamp.append(0)

        else:
            self.timestamp.append(time.time()-self.end_time)

        self.end_time= time.time()


    def on_epoch_end(self, epoch, logs={}):

        print("AVG batch time:",np.mean(self.timestamp))

        y_pred = self.model.predict([self.Q_TEST], verbose=0)

        y_pred=np.squeeze(y_pred,axis=1)

        prod=(y_pred-self.Q_LABELS)*(y_pred-self.Q_LABELS)
        sumo=np.sum(prod)

        print("LABELS",self.Q_LABELS[:8])
        print("PREDS",y_pred[:8])



        print("Callback:", sumo)

        
#################################################################################
#### version for single output
#################################################################################

class PATCHY_LAYER_RETURNFULLSEQ(Layer):

    def __init__(self,patch_size,nb_patches,DR,initializer='glorot_normal',add_residual=False,stretch_factor=1,l2reg=0.000001,nb_sequences=-1,build_backbone=True,psy=0.15,**kwargs):

        ### add support for masking
        self.supports_masking = True

        self.patch_size=patch_size
        self.nb_patches=nb_patches
        self.DR=DR
        self.add_residual=add_residual

        self.initializer=initializer
        self.kernel_causal=4
        self.nb_sequences=nb_sequences
        self.build_backbone=build_backbone

        self.l2reg=l2reg
        self.stretch_factor=stretch_factor

        self.pc_patch_on_top=psy



        super(PATCHY_LAYER_RETURNFULLSEQ, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        if input_mask is not None:
            return input_mask
        else:
            return None


    def PATCHY_LAYER_CNNTOP_initializer(self,shape, dtype=None):

        M=gen_filters_igloo_Nsteps_New_returnreduce(self.patch_size,self.nb_patches,int(self.vector_size/1),self.num_channels_input,return_sequences=True,stretch_factor=self.stretch_factor,nb_sequences=self.nb_sequences,build_backbone=self.build_backbone)
        M.astype(int)
        return M


    def PATCHY_LAYER_CNNTOP_ontop_initializer(self,shape, dtype=None):

        M=gen_filters_fullseq_ontop(self.patch_size,max(int(self.pc_patch_on_top*self.nb_patches),self.patch_size-1),self.vector_size,self.num_channels_input,self.stretch_factor,nb_sequences=self.nb_sequences)
        M.astype(int)
        return M


    def build(self, input_shape):


        self.batch_size=input_shape[0]
        self.vector_size=input_shape[1]
        self.num_channels_input=input_shape[2]


        ###SETTING THIS AS A NON TRAINABLE VARIABLE SO IT CAN BE SAVED ALONG THE WEIGHTS
        self.patches=self.add_weight(shape=(self.vector_size,int(self.nb_patches),int(self.patch_size*self.num_channels_input),2),
                                    initializer=self.PATCHY_LAYER_CNNTOP_initializer,
                                    trainable=False,
                                    name="random_patches",dtype=np.int32)


        self.patches_ontop=self.add_weight(shape=(self.vector_size,max(int(self.pc_patch_on_top*self.nb_patches),self.patch_size-1),int(self.patch_size*self.num_channels_input),2),
                                    initializer=self.PATCHY_LAYER_CNNTOP_ontop_initializer,
                                    trainable=False,
                                    name="random_patches_ontop",dtype=np.int32)


        if self.nb_sequences <0 :


            self.W_MULT = self.add_weight(shape=(1,int(self.vector_size/self.stretch_factor),int(self.nb_patches/1),self.patch_size*self.num_channels_input),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_MULT")

            self.W_BIAS = self.add_weight(shape=(1,int(self.vector_size/self.stretch_factor),int(self.nb_patches/1)),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_BIAS")


            self.W_MULT_ontop = self.add_weight(shape=(1,self.vector_size,max(int(self.pc_patch_on_top*self.nb_patches),self.patch_size-1),self.patch_size*self.num_channels_input),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_MULT_ontop")

            self.W_BIAS_ontop = self.add_weight(shape=(1,self.vector_size,max(int(self.pc_patch_on_top*self.nb_patches),self.patch_size-1) ),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_BIAS_ontop")


        else:

            self.W_MULT = self.add_weight(shape=(1,int(self.nb_sequences/self.stretch_factor),int(self.nb_patches/1),self.patch_size*self.num_channels_input),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_MULT")

            self.W_BIAS = self.add_weight(shape=(1,int(self.nb_sequences/self.stretch_factor),int(self.nb_patches/1)),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_BIAS")


            self.W_MULT_ontop = self.add_weight(shape=(1,self.nb_sequences,max(int(self.pc_patch_on_top*self.nb_patches),self.patch_size-1),self.patch_size*self.num_channels_input),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_MULT_ontop")

            self.W_BIAS_ontop = self.add_weight(shape=(1,self.nb_sequences,max(int(self.pc_patch_on_top*self.nb_patches),self.patch_size-1) ),
                                        initializer=self.initializer,
                                        trainable=True,
                                        regularizer=l2(self.l2reg),
                                        name="W_BIAS_ontop")

        super(PATCHY_LAYER_RETURNFULLSEQ, self).build(input_shape)  # Be sure to call this somewhere!


    def call(self, y, mask=None):

        print("working with horizontal regular layer...")

        print("y.shape",y.shape)

        ###y.shape (?, 520, 1)

        PATCH_tensor1=Patchy_nonzero_1D(y,self.patches,self.nb_patches,self.patch_size)

        print("PATCH_tensor1 after",PATCH_tensor1) ###sshape=(?, 520, 200, 4)


        W_MULT=tf.tile(self.W_MULT,[tf.shape(y)[0],1,1,1])

        MPI=tf.multiply(W_MULT,PATCH_tensor1)

        print("MPI first::",MPI )

        MPI=tf.reduce_sum(MPI, axis=-1)     ##  shape=(?, 520, 733)


        Bias=tf.tile(self.W_BIAS,[tf.shape(y)[0],1,1])

        MPI=tf.add(MPI,Bias)        ##  shape=(?, 520, 733)

        MPI = LeakyReLU(alpha=0.1) (MPI)

        print("MPI before reansversal",MPI)     ###
        print("self.add_residual",self.add_residual)
        print("self.nb_sequences",self.nb_sequences)
        print("MPI after reshape",MPI)                  #### shape=(?, 25, 50)
        print("self.vector_size",self.vector_size)

        if self.nb_sequences<0:

            ###repeating the MPI sequence
            MPI=tf_repeat(MPI,[1,self.stretch_factor,1])

            print("MPI after repeating",MPI)

            ############### DOING THE top section of the tensors

            PATCH_tensor_ontop=Patchy_nonzero_1D(y,self.patches_ontop,self.nb_patches,self.patch_size)

            print("PATCH_tensor_ontop",PATCH_tensor_ontop) ###sshape=(?, 520, 200, 4)


            W_MULT_ontop=tf.tile(self.W_MULT_ontop,[tf.shape(y)[0],1,1,1])

            print("W_MULT_ontop",W_MULT_ontop)

            MPI_ontop=tf.multiply(W_MULT_ontop,PATCH_tensor_ontop)

            print("MPI_ontop",MPI_ontop)

            MPI_ontop=tf.reduce_sum(MPI_ontop, axis=-1)     ##  shape=(?, 520, 733)


            Bias_ontop=tf.tile(self.W_BIAS_ontop,[tf.shape(y)[0],1,1])

            MPI_ontop=tf.add(MPI_ontop,Bias_ontop)        ##  shape=(?, 520, 733)

            MPI_ontop = LeakyReLU(alpha=0.1) (MPI_ontop)

            print("MPI_ontop at end",MPI_ontop)

            ###concatenate the top and upper parts

            MPI=tf.concat([MPI,MPI_ontop],axis=-1)

        print("MPI after concat",MPI)

        if self.add_residual:

            if self.nb_sequences < 0:

                res=Conv1D(int(self.nb_patches + max(self.pc_patch_on_top*self.nb_patches, self.patch_size-1) ), 1, padding='valid') (y)
                res = LeakyReLU(alpha=0.1) (res)
                MPI=tf.add(res,MPI)
                MPI = LeakyReLU(alpha=0.1) (MPI)

            else:

                res=Conv1D(self.nb_patches , 1, padding='valid') (y)
                res = LeakyReLU(alpha=0.1) (res)
                res=res[:,-self.nb_sequences:,:]
                MPI=tf.add(res,MPI)
                MPI = LeakyReLU(alpha=0.1) (MPI)


        return MPI

    def compute_output_shape(self, input_shape):

        if self.nb_sequences < 0:

            return input_shape[0],self.vector_size,int(self.nb_patches + max(self.pc_patch_on_top*self.nb_patches, self.patch_size-1) )

        else:

            return input_shape[0],self.nb_sequences,self.nb_patches


#################################################################################################################################################
####
##################################################################################################################################################


def gen_filters_igloo_Nsteps_New(patch_size,nb_patches,vector_size,num_channels_input_reduced,return_sequences,nb_stacks=1,build_backbone=True,consecutive=False,nb_sequences=-1):
    ###nb_sequences=10 to extract last 10 elements

    OUTA=[]

    if nb_stacks==1:

        for step in range(vector_size):     ### 64


            if (step != vector_size-1) and (return_sequences==False):
    #            print("Skipping setp",step)
                continue

            if return_sequences==True and (nb_sequences !=-1):
                if step <vector_size-nb_sequences:
                    continue

            if step%10==0:
                print("step...",step)

            COLLECTA=[]

            if step <patch_size:

#                print("ok small patch")

                for kk in range(nb_patches):

                    randy_H=np.random.choice(range(step+1), patch_size, replace=True)

                    first=[]

                    for pp in randy_H:
                        for mm in range(num_channels_input_reduced):

                            first.append([pp,mm])

                    COLLECTA.append(first)
    #                print("COLLECTA shape",len(COLLECTA))

    #            print("len collecta",len(COLLECTA))
                OUTA.append(COLLECTA)


            else:

                ####first manufactur the arranged ones :

                if build_backbone:

                    maximum_its=int((step/(patch_size-1))+1)



                    if maximum_its>nb_patches:
                        print("nb_patches too small, recommende above:",maximum_its)
                        sys.exit()

                    for jj in range(maximum_its):
                        if iter==0:
                            randy_H=[step-pp for pp in range(patch_size)]
                        else:
                            randy_H=[max(step-(jj*(patch_size-1))-pp,0)  for pp in range(patch_size)]

                        first=[]

                        for pp in randy_H:
                            for mm in range(num_channels_input_reduced):

                                first.append([pp,mm])

                        COLLECTA.append(first)


                    ### doing rest of iterations as freestyle
                    rest_iters=max(nb_patches-maximum_its,0)

                    for itero in range(rest_iters):

                        if not consecutive:
                            randy_B=np.random.choice(range(step+1), patch_size, replace=False)
                        else:
                            uniq=np.random.choice(range(max(0,step+1-patch_size+1)), 1, replace=False)

                            randy_B= [ uniq[0]+pp  for pp in range(patch_size) ]

                        first=[]

                        for pp in randy_B:
                            for mm in range(num_channels_input_reduced):

                                first.append([pp,mm])


        #                print(first)
                        COLLECTA.append(first)

        #                print("COLLECTA shape",len(COLLECTA))


                    COLLECTA=np.stack(COLLECTA)
        #            print("COLLECTA.shape",COLLECTA.shape)
                    OUTA.append(COLLECTA)

                else:


                    for itero in range(nb_patches):
                        if not consecutive:
                            randy_B=np.random.choice(range(step+1), patch_size, replace=False)
                        else:
                            uniq=np.random.choice(range(max(0,step+1-patch_size+1)), 1, replace=False)

                            randy_B= [ uniq[0]+pp  for pp in range(patch_size) ]

                        first=[]

                        for pp in randy_B:
                            for mm in range(num_channels_input_reduced):

                                first.append([pp,mm])


        #                print(first)
                        COLLECTA.append(first)

        #                print("COLLECTA shape",len(COLLECTA))


                    COLLECTA=np.stack(COLLECTA)
        #            print("COLLECTA.shape",COLLECTA.shape)
                    OUTA.append(COLLECTA)



        OUTA=np.stack(OUTA)


        if return_sequences==False:
            OUTA=np.squeeze(OUTA,axis=0)

        return OUTA

    else:       ###if more than 1 stack

        for step in range(vector_size):     ### 64

            if (step != vector_size-1) and (return_sequences==False):
    #            print("Skipping setp",step)
                continue

            if return_sequences==True and (nb_sequences !=-1):
                if step <vector_size-nb_sequences:
                    continue

            if step%10==0:
                print("step...",step)

            COLLECTA=[]

            if step <patch_size:

                print("The part for multi stack in ONE SHOT has not been dev yet")
                sys.exit()

#                print("ok small patch")

                for kk in range(nb_patches):

                    randy_H=np.random.choice(range(step+1), patch_size, replace=True)

                    first=[]

                    for pp in randy_H:
                        for mm in range(num_channels_input_reduced):

                            first.append([pp,mm])

                    COLLECTA.append(first)
    #                print("COLLECTA shape",len(COLLECTA))

    #            print("len collecta",len(COLLECTA))
                OUTA.append(COLLECTA)


            else:

                ####first manufactur the arranged ones :

                if build_backbone:

                    maximum_its=int((step/(patch_size-1))+1)



                    if maximum_its>nb_patches:
                        print("nb_patches too small, recommende above:",maximum_its)
                        sys.exit()

                    for jj in range(maximum_its):
                        if iter==0:
                            randy_H=[step-pp for pp in range(patch_size)]
                        else:
                            randy_H=[max(step-(jj*(patch_size-1))-pp,0)  for pp in range(patch_size)]

                        first=[]

                        for pp in randy_H:
                            for mm in range(num_channels_input_reduced):

                                first.append([pp,mm])

                        COLLECTA.append(first)


                    ### doing rest of iterations as freestyle
                    rest_iters=max(nb_patches-maximum_its,0)

                    for itero in range(rest_iters):

                        if not consecutive:
                            randy_B=np.random.choice(range(step+1), patch_size, replace=False)
                        else:
                            uniq=np.random.choice(range(max(0,step+1-patch_size+1)), 1, replace=False)

                            randy_B= [ uniq[0]+pp  for pp in range(patch_size) ]

                        first=[]

                        for pp in randy_B:
                            for mm in range(num_channels_input_reduced):

                                first.append([pp,mm])


        #                print(first)
                        COLLECTA.append(first)

        #                print("COLLECTA shape",len(COLLECTA))


                    COLLECTA=np.stack(COLLECTA)
        #            print("COLLECTA.shape",COLLECTA.shape)
                    OUTA.append(COLLECTA)

                else:

                    for stack_id in range(nb_stacks):

                        for itero in range(nb_patches):


                            if not consecutive:

                                randy_B=np.random.choice(range(step+1), patch_size, replace=False)

                            else:

                                uniq=np.random.choice(range(max(0,step+1-patch_size+1)), 1, replace=False)

                                randy_B= [ uniq[0]+pp  for pp in range(patch_size) ]

                            first=[]

                            for pp in randy_B:
                                for mm in range(num_channels_input_reduced):

                                    first.append([stack_id,pp,mm])


            #                print(first)
                            COLLECTA.append(first)

            #                print("COLLECTA shape",len(COLLECTA))


                    COLLECTA=np.stack(COLLECTA)
        #            print("COLLECTA.shape",COLLECTA.shape)
                    OUTA.append(COLLECTA)



        OUTA=np.stack(OUTA)


        if return_sequences==False:
            OUTA=np.squeeze(OUTA,axis=0)

        return OUTA

#############################################################################################################
####
#############################################################################################################


def gen_filters_igloo_Nsteps_New_returnreduce(patch_size,nb_patches,vector_size,num_channels_input_reduced,return_sequences,stretch_factor=1,build_backbone=True,consecutive=False,nb_sequences=-1):
    ###nb_sequences=10 to extract last 10 elements

    OUTA=[]



    for step in range(vector_size):     ### 64


        if step%stretch_factor !=0 :
#                print("skipping step:",step)
            continue


        if return_sequences==True and (nb_sequences !=-1):
            if step <vector_size-nb_sequences:
#                print("skipping",step)
                continue

        if step%10==0:
            print("step for main layer...",step)

        COLLECTA=[]

        if step <patch_size:

#                print("ok small patch")

            for kk in range(nb_patches):

                randy_H=np.random.choice(range(step+1), patch_size, replace=True)

                first=[]

                for pp in randy_H:
                    for mm in range(num_channels_input_reduced):

                        first.append([pp,mm])

                COLLECTA.append(first)
#                print("COLLECTA shape",len(COLLECTA))

#            print("len collecta",len(COLLECTA))
            OUTA.append(COLLECTA)


        else:

            ####first manufactur the arranged ones :

            if build_backbone:

                maximum_its=int((step/(patch_size-1))+1)



                if maximum_its>nb_patches:
                    print("nb_patches too small, recommende above:",maximum_its)
                    sys.exit()

                for jj in range(maximum_its):
                    if iter==0:
                        randy_H=[step-pp for pp in range(patch_size)]
                    else:
                        randy_H=[max(step-(jj*(patch_size-1))-pp,0)  for pp in range(patch_size)]

                    first=[]

                    for pp in randy_H:
                        for mm in range(num_channels_input_reduced):

                            first.append([pp,mm])

                    COLLECTA.append(first)


                ### doing rest of iterations as freestyle
                rest_iters=max(nb_patches-maximum_its,0)

                for itero in range(rest_iters):

                    if not consecutive:
                        randy_B=np.random.choice(range(step+1), patch_size, replace=False)
                    else:
                        uniq=np.random.choice(range(max(0,step+1-patch_size+1)), 1, replace=False)

                        randy_B= [ uniq[0]+pp  for pp in range(patch_size) ]

                    first=[]

                    for pp in randy_B:
                        for mm in range(num_channels_input_reduced):

                            first.append([pp,mm])


    #                print(first)
                    COLLECTA.append(first)

    #                print("COLLECTA shape",len(COLLECTA))


                COLLECTA=np.stack(COLLECTA)
    #            print("COLLECTA.shape",COLLECTA.shape)
                OUTA.append(COLLECTA)

            else:


                for itero in range(nb_patches):
                    if not consecutive:
                        randy_B=np.random.choice(range(step+1), patch_size, replace=False)
                    else:
                        uniq=np.random.choice(range(max(0,step+1-patch_size+1)), 1, replace=False)

                        randy_B= [ uniq[0]+pp  for pp in range(patch_size) ]

                    first=[]

                    for pp in randy_B:
                        for mm in range(num_channels_input_reduced):

                            first.append([pp,mm])


    #                print(first)
                    COLLECTA.append(first)

    #                print("COLLECTA shape",len(COLLECTA))


                COLLECTA=np.stack(COLLECTA)
    #            print("COLLECTA.shape",COLLECTA.shape)
                OUTA.append(COLLECTA)



    OUTA=np.stack(OUTA)



    if return_sequences==False:
        OUTA=np.squeeze(OUTA,axis=0)

    return OUTA

######################################################################################################################
####  Generating extra filters for return_sequences=True
######################################################################################################################

def gen_filters_fullseq_ontop(patch_size,nb_patches,vector_size,num_channels_input_reduced,stretch_factor,return_sequences=True,nb_sequences=-1):

    print("generating additional patches for the top end of the return sequence...")

    OUTA=[]

    for step in range(vector_size):     ### 64

        if nb_sequences>0 and step<vector_size-nb_sequences:
            continue

        if step%10==0:
            print("step for top layer...",step)

        ###find distance to previous lighthouse
        lighthouses=[int(kk*stretch_factor) for kk in range(int(vector_size/stretch_factor))]

        COLLECTA=[]

        if step < patch_size:

            for kk in range(nb_patches):

                randy_H=np.random.choice(range(step+1), patch_size, replace=True)

                first=[]

                for pp in randy_H:
                    for mm in range(num_channels_input_reduced):

                        first.append([pp,mm])

                COLLECTA.append(first)

            OUTA.append(COLLECTA)


        else:

            ### find largest element of lighthouse that's smaller than step
#            print("----------------")
#            print("lighthouses",lighthouses)
#            print("step",step)
            lower_lim=0
            found=False
            for idx,elem in enumerate(lighthouses):
#                print("elem",elem)
                if elem>step:
#                    print("elem>step,...elem:",elem)
                    lower_lim=max(lighthouses[idx-1]-1,0)
                    found=True
                    break

            if found==False:
                lower_lim=lighthouses[-1]-1


#            print("lower_lim",lower_lim)

            for itero in range(nb_patches):

                if step-lower_lim>patch_size-1:
                    randy_B=np.random.choice(range(max(0,lower_lim),step), patch_size-1, replace=False)
                else:
                    randy_B=np.random.choice(range(max(0,lower_lim),step), patch_size-1, replace=True)

#                randy_B=np.random.choice(range(max(0,step-int(vector_size/stretch_factor)-1),step), patch_size-1, replace=False)

                ###forcing the inclusion of the current index

                randy_B = np.append(randy_B, np.array([step]))

                first=[]

                for pp in randy_B:

                    for mm in range(num_channels_input_reduced):

                        first.append([pp,mm])


                COLLECTA.append(first)


            COLLECTA=np.stack(COLLECTA)

            OUTA.append(COLLECTA)



    OUTA=np.stack(OUTA)

#    print("OUTA.shape for special case",OUTA.shape)



    return OUTA



######################################################################################
#####  callback function
#####################################################################################

class CATEGORICALEVAL(Callback):
    def __init__(self, validation_data=()):
        super(Callback, self).__init__()


        first,second = validation_data        ###  (34, 256, 256, 3) ,  (34, 256*256, 3)

        self.DATA=first
        self.LABELS=second


    def on_train_begin(self, logs={}):
        self.scores = []

    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict([self.DATA], verbose=0)


        print(self.LABELS[0],np.argmax(y_pred[0],axis=-1))
        print(self.LABELS[5],np.argmax(y_pred[5],axis=-1))


######################################################################################
##### PATCHY Layer
#####################################################################################
def Patchy_nonzero_1D(y,coords,nb_patches,patch_size):

  ### ----> comes in : y =(?,32, 10)
  ### ----> coords, comes in: 64,256,40,2

  y=tf.transpose(y,[1,2,0])

  print("y",y)

  M=tf.gather_nd(y,coords)

  print("M",M)

  reshaped_nonzero_values=tf.transpose(M,[3,0,1,2])

  print("reshaped_nonzero_values",reshaped_nonzero_values)



  return reshaped_nonzero_values

######################################################################################
##### PATCHY Layer
#####################################################################################
def Patchy_nonzero_1D_lessD(y,coords,nb_patches,patch_size):

  ### ----> comes in : y =(?,32, 10)
  ### ----> coords, comes in: 64,256,40,2

  y=tf.transpose(y,[1,2,0])

  print("coords.shape",coords.shape)

  print("y",y)

  M=tf.gather_nd(y,coords)

  print("M",M)


  reshaped_nonzero_values=tf.transpose(M,[2,0,1])

  print("reshaped_nonzero_values",reshaped_nonzero_values)



  return reshaped_nonzero_values



##############################################################################################################################
####  TF repeat
##############################################################################################################################

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)

    return repeated_tesnor


