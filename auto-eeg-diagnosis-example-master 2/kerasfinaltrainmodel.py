from igloo1d import IGLOO_RETURNFULLSEQ,IGLOO
import h5py
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import multiply, UpSampling1D,Add,Masking,Flatten,Concatenate,LeakyReLU,PReLU,Input,LSTM, core, Bidirectional, CuDNNLSTM, CuDNNGRU,Reshape,Lambda,Permute,TimeDistributed,RepeatVector,ConvLSTM2D,Conv3D,Dense,UpSampling3D,Embedding, SpatialDropout1D,GRU,Add,Activation,multiply
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
from keras.activations import tanh,softmax
from keras_generator_final import Mygenerator
from keras.models import load_model
import numpy as np
from eegtest import test_file
input_shape=(599,2736)
 
nb_patches=200 
nb_patches_FULL=60 
nb_patches_vertical=5 
 
patch_size=4 
mDR=0.45 
MAXPOOL_size=1 
 
CONV1D_dim=10 
nb_stacks=1 
nb_stacks_full=1 
 
igloo_l2reg=0.01 
 
C1D_K=54 
 
Conv1D_dim_full_seq=30 
 
stretch_factor=13       #13 
 
add_residual=True 
add_residual_vertical=True 
build_backbone=False 
 
padding="causal" 

learning_rate = 0.01 ##change

vocab_size = 1149 
lstm_layers = 3
dropout_rate = 0.22

h5f = h5py.File('embedding.h5', 'r')
balloony = h5f['embedding_dataset'][:]
balloony[np.isnan(balloony)]=0.1
balloony=np.nan_to_num(balloony)
h5f.close()

def sparse_cross_entropy(y_true, y_pred):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean

def sparse_cross_entropy2(y_true, y_pred):
    
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)

    return loss
 
def cross_entropy1(y_true, y_pred):
    
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                          logits=y_pred)

    loss_mean = tf.reduce_mean(loss)

    return loss_mean
def cross_entropy2(y_true, y_pred):
    
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,
                                                          logits=y_pred)

 
    return loss
#sparse_categorical_cross entropy-5, categorical_cross_entropy-6
 
def get_model(): 
 
    CONC=[] 
    IGLOO_V=[] 
 
    inin = Input(shape=input_shape, name='input')
    
    #inin=Lambda(lambda q: q[:,1:,:]) (inin) 

    a=Conv1D(40,2,padding="causal")(inin)
    b=Conv1D(40,4,padding="causal")(inin)
    c=Conv1D(40,8,padding="causal")(inin)

    x=Concatenate(axis=-1)([a,b,c])
    x=Activation("relu")(x)
    x= BatchNormalization(axis=-1)(x)

    a=Conv1D(40,2,padding="causal")(x)
    b=Conv1D(40,4, padding="causal")(x)
    c=Conv1D(40,8, padding="causal")(x)

    x=Concatenate(axis=-1)([a,b,c])
    x=Activation("relu")(x)
    x= BatchNormalization(axis=-1)(x)

    a=Conv1D(40,2,padding="causal")(x)
    b=Conv1D(40,4,padding="causal")(x)
    c=Conv1D(40,8, padding="causal")(x)

    x=Concatenate(axis=-1)([a,b,c])
    x=Activation("relu")(x)
    x= BatchNormalization(axis=-1)(x)
    
    x=Lambda(lambda q: q[:,1:,:]) (x)
 
    x=Conv1D(64, 1,strides=1,padding=padding)  (x) 
    x = BatchNormalization(axis=-1) (x) 
    x = Activation("relu") (x) 
    x = SpatialDropout1D(mDR) (x)

    IGLOO_V.append(IGLOO_RETURNFULLSEQ(x,nb_patches_FULL,Conv1D_dim_full_seq,patch_size=patch_size,padding_style=padding,stretch_factor=stretch_factor,l2reg=igloo_l2reg,
                                      add_residual=add_residual,nb_stacks=nb_stacks_full,build_backbone=build_backbone)) 
 

    CONC.append(IGLOO_V[0]) 
 
    for kk in range(5): 
 
        x=Conv1D(C1D_K, 1,strides=1,padding=padding)  (CONC[kk]) 
        x = BatchNormalization(axis=-1) (x) 
        x = Activation("relu") (x) 
        x = SpatialDropout1D(mDR) (x) 
 
        IGLOO_V.append(IGLOO_RETURNFULLSEQ(x,nb_patches_FULL,Conv1D_dim_full_seq,patch_size=patch_size,padding_style=padding,stretch_factor=stretch_factor,l2reg=igloo_l2reg,
                                           add_residual=add_residual,nb_stacks=nb_stacks_full,build_backbone=build_backbone)) 
 
 
        ###second residual connection 
        co=Add() ([IGLOO_V[kk+1],CONC[kk]]) 
        CONC.append(Activation("relu") (co)) 
 
 
    x=Conv1D(C1D_K, 1,strides=1,padding=padding)  (CONC[-1]) 
    x = BatchNormalization(axis=-1) (x) 
    x = Activation("relu") (x) 
    x = SpatialDropout1D(mDR) (x) 
 
    y=IGLOO(x,nb_patches,CONV1D_dim,patch_size=patch_size,return_sequences=False,l2reg=igloo_l2reg,padding_style=padding,nb_stacks=nb_stacks,DR=mDR,max_pooling_kernel=MAXPOOL_size) 
  
    
    y=Dense(64,activation='relu') (y) 
    y=Dropout(0.4) (y)
    output_1=Dense(1,activation='softmax') (y)

    word_input = Input(shape=(9,), name='decoder_input')
    
 
    embedded_word=Embedding(input_dim=1149, output_dim=500, name='word_embedding',input_length=9,trainable=True, weights=[balloony])(word_input) #trainable is false, weight=ballooney
   


    input_=embedded_word
    

    #input_ = BatchNormalization(axis=-1)(input_)
    gru_out=GRU(700, activation='tanh', recurrent_activation='sigmoid', 
    dropout=0.22,return_sequences=True, return_state=False,unroll=False,reset_after=True)(input_)
    
    input_=gru_out
    
    input_ = BatchNormalization(axis=-1)(input_)
    gru_out=GRU(700, activation='tanh', recurrent_activation='sigmoid', 
    dropout=0.22,return_sequences=True, return_state=False,unroll=False,reset_after=True)(input_)
    input_ = gru_out
    
    features=Permute((2,1))(x)
 
    part1=Dense(700)(features)
    gru_out=Permute((2,1))(gru_out)
    
    shape= K.int_shape(part1) #should features be part1? 
    
    part2=Dense(shape[1])(gru_out)
    part2=Permute((2,1))(part2)
    part3= Add()([part1,part2])
    
    score = Activation("tanh")(part3)
    part4= Dense(1)(score)
    
    attention_weights=Lambda(lambda x: softmax(x,axis=1))(part4)
    
    context_vector=multiply([attention_weights,features])
    context_vector=Lambda(lambda x: K.sum(x,axis=1))(context_vector)
    
    context_vector_mod=Dense(600)(context_vector)
    context_vector_mod = Lambda(lambda x: K.expand_dims(x, -1))(context_vector_mod)
    context_vector_mod=Permute((2,1))(context_vector_mod)
    
    gru_out_mod=Dense(600)(gru_out)

    
    input_=Concatenate(axis=1)([context_vector_mod, gru_out_mod])
    input_=Activation("tanh")(input_)


    input_ = BatchNormalization(axis=-1)(input_)
    gru_out=GRU(9, activation='tanh', recurrent_activation='sigmoid', dropout=0.22,return_sequences=True, return_state=False,unroll=False,reset_after=True)(input_)
    #gru_out = LSTM(units=9,
                  #return_sequences=True,
                  #dropout=0.22,
                  #recurrent_dropout=0.22)(input_)
    gru_out=Permute((2,1))(gru_out)
    #gru_out=GRU(700, activation='tanh', recurrent_activation='sigmoid', 
    #dropout=0.22,return_sequences=True, return_state=False,unroll=False,reset_after=True)(input_)
    #input_= gru_out


    #gru_out=Flatten()(input_)
    gru_out=Activation("tanh")(gru_out)
    sequence_output = TimeDistributed(Dense(units=vocab_size))(gru_out)
    
    
 
    opt = optimizers.Adam(lr=0.0005, clipnorm=1.0, decay=0.001) 
    model = Model(inputs=[inin,word_input],outputs=[output_1,sequence_output]) 
    

 
    model.compile(loss=['binary_crossentropy',cross_entropy2],optimizer=opt, metrics=['accuracy'],loss_weights=[100000,1]) 
 
    #return model
    model.fit_generator(Mygenerator(2),epochs=70)
    
    model.save('eegv2.h5')
    
    test_file()

    #del model
    
#cross_entropy1--> compiled but dimension error!, flatten doesn't work

 
 
model=get_model()
#print(model.summary())
