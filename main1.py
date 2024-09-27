# -*- coding: utf-8 -*-

import numpy as np
import time
import math  
import os
import random
from tensorflow.keras import applications
from tensorflow.keras.layers import Input
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.client import device_lib
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # CPU is used with -1. 0, 1 for GPU
print('GPU devise= ',tf.config.list_physical_devices('GPU'))

print(device_lib.list_local_devices())
# tf.config.threading.set_intra_op_parallelism_threads(2)
# tf.config.threading.set_inter_op_parallelism_threads(2)

# #config = tf.ConfigProto()
# tf.config.intra_op_parallelism_threads = 44
# tf.config.inter_op_parallelism_threads = 44
# #tf.session(config=config)
from tensorflow.compat.v1 import ConfigProto




import keras.backend as K
import keras
#import cv2
import numpy.matlib
from keras import callbacks

from keras.utils.vis_utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.utils import np_utils
# Creating our model
from keras.models import Model
from keras import layers

#from keras import backend as K
#import tensorflow.keras.backend as K
from keras.callbacks import LearningRateScheduler

Nclasses=8
Nclasses_coarse=2
Nclasses_1=3
Nclasses_2=5
Nclasses_3=2

VGG_manual=1
MultiTask=1
Training_ratio=0.6
onlyundeep=0

w1=1  #weight of deep features
w2=1  # weight of unlearning features

win_size=100 ; Nlook_a=3  ;Nlook_r=1 ; Ovrlp=21

NL1=32
NL2=32
Look_step= 3

Augment=0
TestAugment=0
NAug=1*Augment+1


ElsevieData=1
S1=1
TSX=0
TSX_S1=0
slcORspe=1#1 for spe and 2 for slc

NetType=3# 
Nepochs = 130 #130
Ntest_reduction = 1
TestStep = 100
BatchSize=32# 
fine_tuneing=1

training_loop=1
Hierarchical=1
save_results=0
model_save = 0


if S1==1:
    N_class1 =  370
    N_class2 =  325
    N_class3 =  350
    N_class4 =  223
    N_class5 =  372
    N_class6 =  216
    N_class7 =  342
    N_class8 =  352
    NsubsetTrain=round(370*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1))#150
    Ntrain_build=round(1*NsubsetTrain*Training_ratio)#60000#
    NsubsetTest=36
    Ntest_build=NsubsetTrain-Ntrain_build-1#NsubsetTrain-Ntrain_build
    
    NsubsetTrain_clutt=round(325*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1))#150
    
    Ntrain_clutt=round(1*NsubsetTrain_clutt*Training_ratio*NAug)#60000#
    NsubsetTest_clutt=300
    NTest_clutt=NsubsetTrain_clutt-Ntrain_clutt-1#NsubsetTrain_clutt- Ntrain_clutt
    #Ntest=NTest_clutt+Ntest_build#10000#
    
    NsubsetTrain_2=round(350*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_2=round(1*NsubsetTrain_2*Training_ratio*NAug)#60000#
    
    NsubsetTest_2=300
    NTest_2=NsubsetTrain_2-Ntrain_2-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_3=round(223*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_3=round(1*NsubsetTrain_3*Training_ratio*NAug)#60000#
    NsubsetTest_3=300
    NTest_3=NsubsetTrain_3-Ntrain_3-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_4=round(372*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_4=round(1*NsubsetTrain_4*Training_ratio*NAug)#60000#
    NsubsetTest_4=300
    NTest_4=NsubsetTrain_4-Ntrain_4-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_5=round(216*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_5=round(1*NsubsetTrain_5*Training_ratio*NAug)#60000#
    NsubsetTest_5=300
    NTest_5=NsubsetTrain_5-Ntrain_5-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_6=round(342*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_6=round(1*NsubsetTrain_6*Training_ratio*NAug)#60000#
    NsubsetTest_6=300
    NTest_6=NsubsetTrain_6-Ntrain_6-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_7=round(352*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_7=round(1*NsubsetTrain_7*Training_ratio*NAug)#60000#
    NsubsetTest_7=300
    Augment=0
    NAug=1*Augment+1    
    
    
elif TSX==1:
    N_class1 =  370
    N_class2 =  325
    N_class3 =  350 
    N_class4 =  305
    N_class5 =  341
    N_class6 =  390
    N_class7 =  225
    N_class8 =  352
elif TSX_S1==1:
    N_class1 =  370
    N_class2 =  325
    N_class3 =  350
    N_class4 =  305
    N_class5 =  341
    N_class6 =  390
    N_class7 =  225
    N_class8 =  352
else:
    N_class1 =  370
    N_class2 =  325
    N_class3 =  350
    N_class4 =  223
    N_class5 =  372
    N_class6 =  216
    N_class7 =  342
    N_class8 =  352                                                                      


if TSX_S1==1:
    NsubsetTrain=round(370*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1))#150
    Ntrain_build=round(1*NsubsetTrain*Training_ratio)#60000#
    NsubsetTest=36
    Ntest_build=NsubsetTrain-Ntrain_build-1#NsubsetTrain-Ntrain_build
    
    NsubsetTrain_clutt=round(325*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1))#150
    
    Ntrain_clutt=round(1*NsubsetTrain_clutt*Training_ratio*NAug)#60000#
    NsubsetTest_clutt=300
    NTest_clutt=NsubsetTrain_clutt-Ntrain_clutt-1#NsubsetTrain_clutt- Ntrain_clutt
    #Ntest=NTest_clutt+Ntest_build#10000#
    
    NsubsetTrain_2=round(350*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_2=round(1*NsubsetTrain_2*Training_ratio*NAug)#60000#
    
    NsubsetTest_2=300
    NTest_2=NsubsetTrain_2-Ntrain_2-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_3=round(305*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_3=round(1*NsubsetTrain_3*Training_ratio*NAug)#60000#
    NsubsetTest_3=300
    NTest_3=NsubsetTrain_3-Ntrain_3-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_4=round(341*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_4=round(1*NsubsetTrain_4*Training_ratio*NAug)#60000#
    NsubsetTest_4=300
    NTest_4=NsubsetTrain_4-Ntrain_4-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_5=round(390*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_5=round(1*NsubsetTrain_5*Training_ratio*NAug)#60000#
    NsubsetTest_5=300
    NTest_5=NsubsetTrain_5-Ntrain_5-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_6=round(225*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_6=round(1*NsubsetTrain_6*Training_ratio*NAug)#60000#
    NsubsetTest_6=300
    NTest_6=NsubsetTrain_6-Ntrain_6-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_7=round(352*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_7=round(1*NsubsetTrain_7*Training_ratio*NAug)#60000#
    NsubsetTest_7=300
    Augment=0
    NAug=1*Augment+1


Ntest_build=NsubsetTrain-Ntrain_build-1
Ntest_clutt=NsubsetTrain_clutt-Ntrain_clutt-1
Ntest_2=NsubsetTrain_2-Ntrain_2-1
Ntest_3=NsubsetTrain_3-Ntrain_3-1
Ntest_4=NsubsetTrain_4-Ntrain_4-1
Ntest_5=NsubsetTrain_5-Ntrain_5-1
Ntest_6=NsubsetTrain_6-Ntrain_6-1
Ntest_7=NsubsetTrain_7-Ntrain_7-1




h=42#28
w=42#12828
ww=42*Augment+w*(1-Augment)#28#
hh=42*Augment+h*(1-Augment)#28#    
DetectionResults_Show=0

Iteration_num=2
DR=np.zeros((1,Iteration_num))
FAR=np.zeros((1,Iteration_num))
F1=np.zeros((1,Iteration_num))

NetworkAcc=np.zeros((1,Iteration_num))
NetworkF1=np.zeros((1,Iteration_num))
NetworkPrc=np.zeros((1,Iteration_num))
NetworkRec=np.zeros((1,Iteration_num))

NetworkAcc_coarse=np.zeros((1,Iteration_num))
NetworkF1_coarse=np.zeros((1,Iteration_num))
NetworkPrc_coarse=np.zeros((1,Iteration_num))
NetworkRec_coarse=np.zeros((1,Iteration_num))

NetworkAcc_Hierarchical=np.zeros((1,Iteration_num))
NetworkF1_Hierarchical=np.zeros((1,Iteration_num))
NetworkPrc_Hierarchical=np.zeros((1,Iteration_num))
NetworkRec_Hierarchical=np.zeros((1,Iteration_num))

for i_iter in range(Iteration_num):

    if ElsevieData==1:
      if S1==1:
        from ImportFile import ImportData
        #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
        [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()
      elif TSX==1:
        from ImportFile_TSX import ImportData
        #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
        [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()
      elif TSX_S1==1:
        from ImportFile_TSX_S1 import ImportData
        #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
        [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()

      else:
        from ImportFile import ImportData
        #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
        [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()
    
       
    

    

    if Nclasses==2:
        Ntrain=Ntrain_clutt+Ntrain_build#10000#
        Ntest=Ntest_clutt+Ntest_build 
    elif Nclasses==4:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3#10000#
    elif Nclasses==5:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4#10000#
    elif Nclasses==6:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4+Ntrain_5#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4+Ntest_5#10000#        
 
    elif Nclasses==7:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4+Ntrain_5+Ntrain_6#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4+Ntest_5+Ntest_6#10000#
        
    else:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4+Ntrain_5+Ntrain_6+Ntrain_7#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4+Ntest_5+Ntest_6+Ntest_7#10000#
    N_val= np.floor(np.divide(Ntrain*0.15,BatchSize))* BatchSize#1984#9984#100000
    N_val=N_val.astype('uint16')

    Ntrain1 = Ntrain_clutt+Ntrain_build+Ntrain_2
    N_val1= np.floor(np.divide(Ntrain1*0.15,BatchSize))* BatchSize#1984#9984#100000
    N_val1=N_val1.astype('uint16')       
    if ElsevieData ==1:
        #train_images=train_images.reshape(28,28,Ntrain)
        h=32*slcORspe#28
        w=32*slcORspe#12828

        h=32*slcORspe#28
        w=32*slcORspe#12828         
        ww=32*slcORspe*Augment+w*(1-Augment)#28#
        hh=32*slcORspe*Augment+h*(1-Augment)#28#         
        ww1=32*slcORspe
        hh1=32*slcORspe
        dim = (ww1, ww1)
        dim1 = (ww1, ww1)
        train_images1=np.zeros((Ntrain,ww1,ww1),dtype=np.float16)
        train_images2=np.zeros((Ntrain,ww1,ww1),dtype=np.float16)
        
        test_images1=np.zeros((Ntest,ww1,ww1),dtype=np.float16)
        test_images2=np.zeros((Ntest,ww1,ww1),dtype=np.float16)
        i_max=0
        i_nmax=[]
        for n in range(Ntrain):
          img= train_images[n,:,:]
          img1= train_images[n,:,:]
          
          #img = cv2.resize(img, dim1, interpolation = cv2.INTER_AREA)
          #img1 = cv2.resize(train_images[n,:,:], dim, interpolation = cv2.INTER_AREA)
          
          train_images1[n,:,:]=abs(img1)
          train_images2[n,:,:]=(img-np.mean(img))/np.var(img)  
          train_images2[n,:,:]=img/np.max(img)
    
          #train_images1=train_images2
        in_nan=np.ones((1,Ntest))
        in_nan0=np.ones((1,Ntest))  
        in_nan0_m=np.ones((1,Ntest))  
    
        for n in range(Ntest):
          img= test_images[n,:,:]
          img1= test_images[n,:,:]
          
          in_nan0[0,n]=np.var(img)
          innn0=in_nan0[0,n] 
          in_nan0_m[0,n]=np.mean(img)
          innn0_m=in_nan0_m[0,n]
          
          #img = cv2.resize(img, dim1, interpolation = cv2.INTER_AREA)
          #img1 = cv2.resize(test_images[n,:,:], dim, interpolation = cv2.INTER_AREA)
    
          test_images1[n,:,:]=abs(img1)
          test_images2[n,:,:]=(img-np.mean(img))/np.var(img)
          test_images2[n,:,:]=img/np.max(img)  
          #test_images1=test_images2
          in_nan[0,n]=np.var(img)
          innn=in_nan[0,n]
          if innn==0:
              NNN=n
          
        #test_images = test_images1
       
        #train_images=train_images1
        X_train=train_images2
        X_test=test_images2
    

    # CNN Network
    
    
    
    
    
    plt.close('all')
    # Load data
    
    #Y_train = np_utils.to_categorical(train_labels,dtype="uint8")
    #Y_test = np_utils.to_categorical(test_labels,dtype="uint8")
    
    Y_train = np_utils.to_categorical(train_labels)
    Y_test = np_utils.to_categorical(test_labels)
    
    Y_train_coarse = np_utils.to_categorical(train_labels_H)
    Y_test_coarse = np_utils.to_categorical(test_labels_H) 
    
    Y_train_1 = np_utils.to_categorical(train_labels[0:Ntrain_build+Ntrain_clutt+Ntrain_2])
    Y_test_1 = np_utils.to_categorical(test_labels[0:Ntest_build+Ntest_clutt+Ntest_2])
    
    Y_train_2 = np_utils.to_categorical(train_labels[Ntrain_build+Ntrain_clutt+Ntrain_2:]-Nclasses_2+2)
    Y_test_2 = np_utils.to_categorical(test_labels[Ntest_build+Ntest_clutt+Ntest_2:]-Nclasses_2+2)
    
    Y_train_3 = np_utils.to_categorical(train_labels_H)
    Y_test_3 = np_utils.to_categorical(test_labels_H) 
    
    NTest_1= Y_test_1.shape[0]
    NTest_2= Y_test_2.shape[0]

    #Y_train = np.asarray(train_labels).astype('float32').reshape((-1,1))
    #Y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))
    
    #Y_train = train_labels
    #Y_test = test_labels
    if NetType==1:
       Y_train = Y_train.reshape(Ntrain, 1, 1,Nclasses)
       Y_test = Y_test.reshape(Ntest, 1, 1,Nclasses)
    #Y_train=np.roll(Y_train,146,0)
    #Y_test=np.roll(Y_test,146,0)
    
    
    
    # Data attributes
    print("train_images dimentions: ", train_images.ndim)
    print("train_images shape: ", train_images.shape)
    print("train_images type: ", train_images.dtype)
                             

    
    if onlyundeep==0:
        #Y_train = train_labels.reshape(699,2)
        #Y_test = test_labels.reshape(99,2)
        X_train = X_train.reshape(Ntrain, ww1, hh1,1)
        X_test = X_test.reshape(Ntest, ww1, hh1,1)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        
        X_train_1 = X_train[0:Ntrain_build+Ntrain_clutt+Ntrain_2,:,:].reshape(Ntrain_build+Ntrain_clutt+Ntrain_2, ww1, hh1,1)
        X_test_1 = X_test[0:Ntest_build+Ntest_clutt+Ntest_2,:,:].reshape(Ntest_build+Ntest_clutt+Ntest_2, ww1, hh1,1)
        X_train_1 = X_train_1.astype('float32')
        X_test_1 = X_test_1.astype('float32')    
        
        X_train_2 = X_train[Ntrain_build+Ntrain_clutt+Ntrain_2:Ntrain,:,:].reshape(Ntrain_3+Ntrain_4+Ntrain_5+Ntrain_6+Ntrain_7, ww1, hh1,1)
        X_test_2 = X_test[Ntest_build+Ntest_clutt+Ntest_2:Ntest,:,:].reshape(Ntest_3+Ntest_4+Ntest_5+Ntest_6+Ntest_7, ww1, hh1,1)
        X_train_2 = X_train_2.astype('float32')
        X_test_2 = X_test_2.astype('float32')    
        
        '''
        X_train = train_images
        X_test = test_images
        '''
        
        
        #_train /= 255
        #_test /= 255
        
        from keras.utils import np_utils
        #CUDA_VISIBLE_DEVICES=0
        import tensorflow as tf
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall        
        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))        
        def scheduler(epoch):
            if NetType==1:
              lr_init=0.004# 0.001 for VGG
            else:
              lr_init=0.0001
            lr=lr_init*pow(.5,math.floor(epoch/50))
            print('lr= ',lr)
            return lr
        def scheduler1(epoch):
            if NetType==1:
              lr_init=0.04# 0.001 for VGG
            else:
              lr_init=0.0001
            lr=lr_init*pow(.5,math.floor(epoch/50))
            print('lr= ',lr)
            return lr
        def scheduler_1(epoch):
            if NetType==1:
              lr_init=0.04# 0.001 for VGG
            else:
              lr_init=0.01
            lr=lr_init*pow(.5,math.floor(epoch/10))
            print('lr= ',lr)
            return lr        

        myInput = layers.Input(shape=(32*slcORspe,32*slcORspe,3))
            
        #myInput = layers.Input(shape=(32,32))
        #conv1 = layers.Conv2D(4, 3, activation='relu', padding='same', strides=2)(myInput)
        #conv2 = layers.Conv2D(4, 3, activation='relu', padding='same', strides=2)(conv1)

        if NetType==3 :
            config = ConfigProto()
            #config.gpu_options.per_process_gpu_memory_fraction = 0.333
            config.gpu_options.per_process_gpu_memory_fraction=0.830
            config.gpu_options.allow_growth = True            
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
             tf.config.experimental.set_memory_growth(physical_devices[0], True)
             tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])

            
        
            
   
            
  
            
            if VGG_manual==1:
               
                ################
                myInput_ = layers.Input(shape=(32*slcORspe,32*slcORspe,3))

                conv1_ = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(myInput_)
                conv1_ = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(conv1_)
                
                pool1_ = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_)
                #conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
                conv2_ = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool1_)
                conv2_ = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(conv2_)
                
                pool2_ = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_)
                pool2_ = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool2_)#added in 1403/4/27
                pool2_ = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(pool2_)# added in 1403/4/27
                

                #conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv2)
                #conv4 = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv3)
                flat_ = layers.Flatten()(pool2_)
                
                #Dense1_ = layers.Dense(4096, activation='relu')(flat_)# commented in 2 shahrivar 1403
                Dense2_ = layers.Dense(4096, activation='relu')(flat_)
                #Dense2_=layers.Dropout(409640(Dense1_)
                
                out_layer_vgg_ = layers.Dense(Nclasses, activation='softmax')(Dense2_)
                VGG_Manual_Natural= Model(inputs=myInput_, outputs=out_layer_vgg_)#myInput
                
                #####################################################
                
                myInput_1 = layers.Input(shape=(32*slcORspe,32*slcORspe,3))

                conv1_1 = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(myInput_1)
                pool1_1 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_1)
                
                conv1_1 = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(conv1_1)
                
                #pool1_1 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_1)
                #conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
                conv2_1 = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool1_1)
                pool2_1 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_1)
                
                conv2_1 = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool2_1)
                
                #pool2_1 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_1)

                
                #conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv2)
                #conv4 = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv3)
                #flat_1 = layers.Flatten()(pool2_1)
                flat_1 = layers.Flatten()(conv2_1)
                
                Dense1_1 = layers.Dense(4096, activation='relu')(flat_1)
                #Dense2_1= layers.Dropout(0.5)(Dense1_1)
                Dense2_1 = layers.Dense(4096, activation='relu')(Dense1_1)

                #Dense2_=layers.Dropout(409640(Dense1_)
                
                out_layer_vgg_1 = layers.Dense(Nclasses, activation='softmax')(Dense2_1)
                VGG_Manual_ManMade= Model(inputs=myInput_1, outputs=out_layer_vgg_1)#myInput 
                out_layer_vgg_1_new = layers.Dense(Nclasses_3, activation='softmax')(Dense2_1)
#################################
                myInput_1_ = layers.Input(shape=(32*slcORspe,32*slcORspe,3))

                conv1_1_ = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(myInput_1_)
                pool1_1_= layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_1_)
                
                conv1_1_ = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(pool1_1_)
                
                #pool1_1 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_1)
                #conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
                conv2_1_ = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(conv1_1_)
                pool2_1_ = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_1_)
                
                conv2_1_ = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool2_1_)
                
                #pool2_1 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_1)

                
                #conv3 = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv2)
                #conv4 = layers.Conv2D(128, 3, activation='relu', padding='same', strides=2)(conv3)
                #flat_1 = layers.Flatten()(pool2_1)
                flat_1_ = layers.Flatten()(conv2_1_)
                
                Dense1_1_ = layers.Dense(4096, activation='relu')(flat_1_)
                #Dense2_1= layers.Dropout(0.5)(Dense1_1)
                Dense2_1_ = layers.Dense(4096, activation='relu')(Dense1_1_)
                out_layer_vgg_1_ = layers.Dense(Nclasses_3, activation='softmax')(Dense2_1_)

                VGG_Manual_ManMade_= Model(inputs=myInput_1_, outputs=out_layer_vgg_1_)#myInput 

                myModel=VGG_Manual_ManMade
                baseModel1 = applications.VGG16(weights='model_vgg16.h5', include_top=False,
            	input_tensor=Input(shape=(32*slcORspe, 32*slcORspe, 3)))

                myModel_Hierarchical = myModel
                ###################################### 

###########################################################################

                ###################
                myInput_9 = layers.Input(shape=(16,16,128))
                flat_9 = layers.Flatten()(myInput_9)
                
                Dense2_9 = layers.Dense(4096, activation='relu')(flat_9)
                
                out_layer_vgg_44 = layers.Dense(Nclasses_3, activation='softmax')(Dense2_9)
                VGG_Manual_ManNature_new_1 = Model(inputs=myInput_9, outputs=out_layer_vgg_44)#myInput
                ###################
                myInput_10 = layers.Input(shape=(16,16,128))
                flat_10 = layers.Flatten()(myInput_10)
                
                Dense1_10 = layers.Dense(4096, activation='relu')(flat_10)
                Dense2_10 = layers.Dense(4096, activation='relu')(Dense1_10)
                
                out_layer_vgg_55 = layers.Dense(Nclasses_1, activation='softmax')(Dense2_10)
                VGG_Manual_Natural_new_1 = Model(inputs=myInput_10, outputs=out_layer_vgg_55)#myInput                 
###########################################################################                 

            else:
                baseModel1 = applications.VGG16(weights='model_vgg16.h5', include_top=False,
            	input_tensor=Input(shape=(32*slcORspe, 32*slcORspe, 3)))
                #baseModel1.save('model_vgg16.h5')
                headModel1 = baseModel1.output
                flat1 = layers.Flatten()(headModel1)            
                FC1 = layers.Dense(256, activation='relu')(flat1)#8192
                DR1=layers.Dropout(0.5)(FC1)     
                out_layer1 = layers.Dense(Nclasses, activation='softmax')(DR1)
                myModel_Hierarchical = Model(inputs=baseModel1.input, outputs=out_layer1)#myInput
                
                baseModel = applications.VGG16(weights='model_vgg16.h5', include_top=False,
            	input_tensor=Input(shape=(32*slcORspe, 32*slcORspe, 3)))
                #baseModel.save('model_vgg16.h5')
                headModel = baseModel.output
                flat = layers.Flatten()(headModel)         
                FC = layers.Dense(256, activation='relu')(flat)#8192
                DR=layers.Dropout(0.5)(FC)
                out_layer = layers.Dense(Nclasses, activation='softmax')(DR)
                myModel = Model(inputs=baseModel.input, outputs=out_layer)#myInput
        
                baseModel2 = applications.VGG16(weights='model_vgg16.h5', include_top=False,
            	input_tensor=Input(shape=(32*slcORspe, 32*slcORspe, 3)))
                #baseModel2.save('model_vgg16.h5')
                headModel2 = baseModel2.output
                flat2 = layers.Flatten()(headModel2)            
                FC2 = layers.Dense(128,activation='relu')(flat2)# 512 1024
                DR2=layers.Dropout(0.5)(FC2)
                out_layer_coarse = layers.Dense(Nclasses_coarse, activation='softmax')(DR2)
                myModel_coarse = Model(inputs=baseModel2.input, outputs=out_layer_coarse)#myInput
    
                baseModel3 = applications.VGG16(weights='model_vgg16.h5', include_top=False,
            	input_tensor=Input(shape=(32*slcORspe, 32*slcORspe, 3)))
                #baseModel3.save('model_vgg16.h5')
                headModel3 = baseModel3.output
                flat3 = layers.Flatten()(headModel3)            
                FC3 = layers.Dense(512, activation='relu')(flat3)#8192
                DR3=layers.Dropout(0.5)(FC3)
                out_layer_1 = layers.Dense(Nclasses_1, activation='softmax')(DR3)
                myModel_1 = Model(inputs=baseModel3.input, outputs=out_layer_1)#myInput  


                
                baseModel4 = applications.VGG16(weights='model_vgg16.h5', include_top=False,
            	input_tensor=Input(shape=(32*slcORspe, 32*slcORspe, 3)))
                #baseModel4.save('model_vgg16.h5')
                headModel4 = baseModel4.output
                flat4 = layers.Flatten()(headModel4)            
                FC4 = layers.Dense(4096, activation='relu')(flat4)#8192
                DR4=layers.Dropout(0.5)(FC4)
                out_layer_2 = layers.Dense(Nclasses_2, activation='softmax')(DR4)
                myModel_2 = Model(inputs=baseModel4.input, outputs=out_layer_2)#myInput                
        '''
        ################### 
        '''
        

        
        plot_model(myModel,to_file='model.png',show_shapes=True,show_layer_names=True)
        plot_model(myModel_Hierarchical,to_file='model2.png',show_shapes=True,show_layer_names=True)
        plot_model(myModel_coarse,to_file='model1.png',show_shapes=True,show_layer_names=True)
        
        myModel.summary()
        VGG_Manual_Natural.summary()
        VGG_Manual_ManNature_new_1.summary()


            #myModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        myModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        
        #==================================================
        # Train our model
        #network_history = myModel.fit(X_train, Y_train, batch_size=128, epochs=2, validation_split=0.2)
        #lrold=round(myModel.optimizer.lr.numpy(), 5)
        lr0=LearningRateScheduler(scheduler)
        model_checkpoint = callbacks.ModelCheckpoint('/model.{epoch}.h5')
        
        checkpoint_filepath = 'D/codes/checkpoint'
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        
        
        logger = callbacks.CSVLogger('training.log')
        
        tensorboard = callbacks.TensorBoard(log_dir='./tensorboard')
        callbacks_list=[lr0]
        
        #callbacks = [model_checkpoint, logger, tensorboard]
        #network_history = myModel.fit(X_train, Y_train, batch_size = 128, epochs=Nepochs,callbacks = callbacks_list,shuffle=True,verbose=1, validation_split=0.2)
        
        #network_history = myModel.fit(X_train, Y_train, batch_size = 128, epochs=Nepochs,shuffle=True,verbose=1, validation_split=0.2)
        it=0
        start = time.time()
            
        X_train_vgg=np.repeat(np.asarray(X_train)[:,:],3,axis=3)
        X_test_vgg=np.repeat(np.asarray(X_test)[:,:],3,axis=3)
 
        X_train_vgg_1=np.repeat(np.asarray(X_train_1)[:,:],3,axis=3)
        X_test_vgg_1=np.repeat(np.asarray(X_test_1)[:,:],3,axis=3)        
        
        X_train_vgg_2=np.repeat(np.asarray(X_train_2)[:,:],3,axis=3)
        X_test_vgg_2=np.repeat(np.asarray(X_test_2)[:,:],3,axis=3)             
               

            #final_conv_layer = myModel.layers[21]
         
        
        if fine_tuneing==1:

           if Hierarchical==1:
                fine_tuneing=0


                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                
                # Prepare the training dataset.
                batch_size = BatchSize
                #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
                myInput = layers.Input(shape=(batch_size,32,32,3))
                #inputs = keras.Input(shape=(784,), name="digits")
                inputs = myInput   
                
                Ntrain_rem =  np.floor(np.divide(Ntrain,BatchSize))              
                Ntrain=Ntrain_rem*BatchSize
                Ntrain= Ntrain.astype('int64')

                Ntrain_rem1 =  np.floor(np.divide(Ntrain1,BatchSize))              
                Ntrain1=Ntrain_rem1*BatchSize
                Ntrain1= Ntrain1.astype('int64')

                X_train_vgg = X_train_vgg[0:Ntrain,:,:,:]
                x_train=X_train_vgg
                y_train=Y_train[0:Ntrain,:]
                
                x_train_coarse=X_train_vgg[0:Ntrain,:,:,:]#X_train_vgg_1
                y_train_coarse=Y_train[0:Ntrain,:]#Y_train_coarse
                
                x_train_coarse1=X_train_vgg_1#for ManMade
                y_train_coarse1=Y_train_1#Y_train_coarse
                
                
            
                x_train_coarse2=X_train_vgg_1#X_train_vgg_1
                y_train_coarse2=Y_train_1#Y_train_coarse

                y_train_coarse3 = Y_train_3[0:Ntrain,:]

                
                val_ind_coarse=np.linspace(0,x_train_coarse.shape[0]-1,x_train_coarse.shape[0])
                random.shuffle(val_ind_coarse)
                val_ind_coarse=val_ind_coarse.astype('int32')

                val_ind_coarse1=np.linspace(0,x_train_coarse1.shape[0]-1,x_train_coarse1.shape[0])
                random.shuffle(val_ind_coarse1)
                val_ind_coarse1=val_ind_coarse1.astype('int32')
                
                x_val_coarse = x_train_coarse[val_ind_coarse[0:N_val],:,:,:]
                y_val_coarse = y_train_coarse[val_ind_coarse[0:N_val]]

                x_val_coarse1 = x_train_coarse1[val_ind_coarse1[0:N_val1],:,:,:]
                y_val_coarse1= y_train_coarse1[val_ind_coarse1[0:N_val1]]
                                
                y_train_coarse = y_train_coarse[val_ind_coarse[N_val:]]                 
                x_train_coarse = x_train_coarse[val_ind_coarse[N_val:],:,:,:]

                y_train_coarse1 = y_train_coarse1[val_ind_coarse1[N_val1:]]                 
                x_train_coarse1 = x_train_coarse1[val_ind_coarse1[N_val1:],:,:,:]

                y_val_coarse3 = y_train_coarse3[val_ind_coarse[0:N_val]]             
                y_train_coarse3 = y_train_coarse3[val_ind_coarse[N_val:]]                 

                train_dataset_coarse = tf.data.Dataset.from_tensor_slices((x_train_coarse, y_train_coarse, y_train_coarse3))
                #train_dataset_coarse = tf.data.Dataset.from_tensor_slices((x_train_coarse, y_train_coarse))

                train_dataset_coarse = train_dataset_coarse.batch(batch_size)

                train_dataset_coarse1 = tf.data.Dataset.from_tensor_slices((x_train_coarse1, y_train_coarse1, y_train_coarse1))
                #train_dataset_coarse = tf.data.Dataset.from_tensor_slices((x_train_coarse, y_train_coarse))

                train_dataset_coarse1 = train_dataset_coarse1.batch(batch_size)


                val_dataset_coarse = tf.data.Dataset.from_tensor_slices((x_val_coarse, y_val_coarse,y_val_coarse3))
                val_dataset_coarse = val_dataset_coarse.batch(batch_size)

                val_dataset_coarse1 = tf.data.Dataset.from_tensor_slices((x_val_coarse1, y_val_coarse1,y_val_coarse1))
                val_dataset_coarse1 = val_dataset_coarse1.batch(batch_size)

                #model = keras.Model(inputs=inputs, outputs=outputs)
                if VGG_manual==1:
                        
                    model_coarse= VGG_Manual_ManMade_#VGG_Manual_Natural#VGG_Manual_coarse#myModel_coarse
                    model= VGG_Manual_Natural#VGG_Manual_ManMade

                    model_coarse3 = VGG_Manual_ManNature_new_1#VGG_Manual_ManNature#VGG_Manual_ManNature_new#VGG_Manual_ManMade_#VGG_Manual_ManNature# myModel_coarse_new##VGG_Manual_4#VGG_Manual_ManNature
                    model_coarse1=VGG_Manual_Natural_new_1#myModel_coarse_1_new#VGG_Manual_Natural                
                    # Instantiate an optimizer to train the model.
                    if MultiTask==0:
                      model_coarse3 = VGG_Manual_ManMade_
                else:
                    model=myModel#VGG_Manual
                    model_coarse3 = myModel_coarse#VGG_Manual_4#VGG_Manual_ManNature
                    # Instantiate an optimizer to train the model.  
                schedules_=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_steps=10000 ,decay_rate=0.95,staircase=False)
                    
                optimizer  = tf.keras.optimizers.SGD(learning_rate=schedules_, momentum=0.9)
                optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=.9)
                optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-3)
                optimizer3 = tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=.9)


                # Instantiate a loss function.
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                
                # Prepare the metrics.
                train_acc_metric = keras.metrics.CategoricalAccuracy()
                val_acc_metric = keras.metrics.CategoricalAccuracy()
                train_acc_metric_coarse = keras.metrics.CategoricalAccuracy()
                train_acc_metric_coarse3 = keras.metrics.CategoricalAccuracy()
                train_acc_metric_coarse1 = keras.metrics.CategoricalAccuracy()
                
                train_acc_metric_Hierarchical = keras.metrics.CategoricalAccuracy()
                
                val_acc_metric_coarse = keras.metrics.CategoricalAccuracy()
                val_acc_metric_coarse3 = keras.metrics.CategoricalAccuracy()
                val_acc_metric_coarse2 = keras.metrics.CategoricalAccuracy()
                val_acc_metric_coarse1 = keras.metrics.CategoricalAccuracy()
                
                train_acc_coarse=np.zeros((Nepochs,1))
                train_loss_coarse=np.zeros((Nepochs,1)) 
                
                val_acc_metric_Hierarchical = keras.metrics.CategoricalAccuracy()
                train_acc_Hierarchical=np.zeros((Nepochs,1))
                train_loss_Hierarchical=np.zeros((Nepochs,1))                
                import time
                val_acc_coarse=np.zeros((Nepochs,1))
                val_loss_coarse=np.zeros((Nepochs,1))
                #logits_Hierarchical=np.zeros(())
                
                epochs = Nepochs
                Train_Acc=np.zeros((Nepochs,1))
                Val_Acc=np.zeros((Nepochs ,1))
                Train_Acc_coarse=np.zeros ((Nepochs,1))
                Val_Acc_coarse=np.zeros((Nepochs ,1)) 
                Train_Acc_coarse3 =np.zeros((Nepochs,1))
                Train_Acc_coarse1 =np.zeros((Nepochs,1))
                
                Val_Acc_coarse3=np.zeros((Nepochs ,1)) 
                Val_Acc_coarse1=np.zeros((Nepochs ,1)) 
                
                
                Train_Acc_Hierarchical=np.zeros((Nepochs,1))
                Val_Acc_Hierarchical=np.zeros((Nepochs ,1)) 
                
                #model_coarse1=load_model('model_coarse1_training0.01_lookstep_3_new.h5')
                #model_coarse2=load_model('model_coarse2_training0.01_lookstep_3.h5')
                #model_coarse3=load_model('model_coarse3.h5')
                w_t=np.ones((32,Nclasses))
                for epoch in range(epochs):
                    print("\nStart of epoch %d" % (epoch,))
                    start_time = time.time()
                    A22=([0.9, 0.5, 0, 0,0,0,0,0,0,0,0,0,0,0])
                    A11=[0.1, 0.5, 1, 1,1,1,1,1,1,1,1,1,1,1]                    
                    A22= np.array(A22)
                    A11= np.array(A11)
                    
                    ind_A1=math.floor(epoch/20)
                    #ind_A1=ind_A1.astype('int32')
                    A1=A11[ind_A1,]
                    A2=A22[ind_A1,]
                    
                    print("A1=" ,A1)
                    print("A2=" ,A2)

                                                     
                    w_3 = np.ones((32,8))
                    LI  = np.zeros((32,8))
                    val_w_3 = np.ones((32,8))
                    val_LI  = np.ones((32,8))  
                    loss_value_coarse12=np.zeros((32,1))

                    

                    
                    #for step, (x_batch_train, y_batch_train_coarse, y_train_coarse3) in enumerate(train_dataset_coarse):
                    for step, (x_batch_train, y_batch_train_coarse, y_train_coarse3) in enumerate(train_dataset_coarse):

                        xbtch=x_batch_train.numpy()
                        x_batch_train1=np.array(xbtch)    
                        x_batch_train3= x_batch_train1#x_train_coarse[step:step+32,:,:,:]
                        y_batch_train_coarse3 = y_train_coarse3
                        if MultiTask ==0:

                        
                            with tf.GradientTape() as tape3:
                                logits_coarse3 = model_coarse3(x_batch_train3, training=True)
                                loss_value_coarse3  = loss_fn(y_batch_train_coarse3, logits_coarse3)
                            
                            grads3 = tape3.gradient(loss_value_coarse3, model_coarse3.trainable_weights)
                            optimizer3.apply_gradients(zip(grads3, model_coarse3.trainable_weights))                
                            train_acc_metric_coarse3.update_state(y_batch_train_coarse3, logits_coarse3)                        
                        if MultiTask==0:
                            with tf.GradientTape() as tape:
                              logits_coarse_ = model(x_batch_train1, training=True)
                              loss_value_coarse  = loss_fn(y_batch_train_coarse, logits_coarse_)

                            grads = tape.gradient(loss_value_coarse, model.trainable_weights)
                            optimizer.apply_gradients(zip(grads, model.trainable_weights))                
                            train_acc_metric_coarse.update_state(y_batch_train_coarse, logits_coarse_)


                        elif MultiTask ==1:
                            get_output = K.function([model.layers[0].input], 
                            [model.layers[4].output]) #model.layers[4].output for 
                            x_batch_train3_ = get_output([x_batch_train3])
                            #x_batch_train3_=x_batch_train3_.numpy()
                            #x_batch_train3_=np.array(x_batch_train3_)
                            with tf.GradientTape() as tape3:
                                logits_coarse3 = model_coarse3(x_batch_train3_, training=True)
                                loss_value_coarse3  = loss_fn(y_batch_train_coarse3, logits_coarse3)
                            
                            grads3 = tape3.gradient(loss_value_coarse3, model_coarse3.trainable_weights)
                            optimizer3.apply_gradients(zip(grads3, model_coarse3.trainable_weights))                
                            train_acc_metric_coarse3.update_state(y_batch_train_coarse3, logits_coarse3)                                
                            
                            #final_conv_layer = model.layers[7]
                            logits_coarse3 = model_coarse3(x_batch_train3_, training=False) 

                            #loss_value_coarse3  = loss_fn(y_batch_train_coarse3, logits_coarse3)

                            
                            val_CoarseClass =  np.argmax(logits_coarse3, axis=1)
                            val_logits_coarse3=np.divide(logits_coarse3,np.repeat((np.max(logits_coarse3,axis=1)).reshape(logits_coarse3.shape[0],1),Nclasses_3,axis=1))
                        
                            
                            val_Ind_ManMade = (val_CoarseClass>0)
                            val_Ind_Nature  = (val_CoarseClass<1)
                            
                            
                            val_w_a = np.repeat(np.reshape(val_logits_coarse3[:,0],(32,1)), Nclasses_1, axis=1)
                            val_w_b = np.repeat(np.reshape(val_logits_coarse3[:,1],(32,1)), Nclasses_2, axis=1)
                            val_w_3[:,0:3] = val_w_a
                            
                            
                            val_w_3[:,3:]  = val_w_b
                            with tf.GradientTape() as tape:
                              logits_coarse_ = model(x_batch_train1, training=True)
                              loss_value_coarse  = loss_fn(y_batch_train_coarse, logits_coarse_)

                              loss_value_coarse  = loss_value_coarse*A1+ loss_value_coarse3*A2#+ loss_value_coarse12*.33
                              logits_coarse__=logits_coarse_*val_w_3   

                            grads = tape.gradient(loss_value_coarse, model.trainable_weights)
                            optimizer.apply_gradients(zip(grads, model.trainable_weights))  

                            train_acc_metric_coarse.update_state(y_batch_train_coarse, logits_coarse_)

                        else:
                            logits_coarse_ = model(x_batch_train1, training=True)
                            loss_value_coarse  = loss_fn(y_batch_train_coarse, logits_coarse_)                            
                            grads = tape.gradient(loss_value_coarse, model.trainable_weights)
                            optimizer.apply_gradients(zip(grads, model.trainable_weights))                
                            train_acc_metric_coarse.update_state(y_batch_train_coarse, logits_coarse__)
                        
                        if step % 400 == 0:

                           
                            print(
                                "Training acc coarse3 (for one batch) at step %d: %.4f"
                                % (step, float(train_acc_metric_coarse3.result()))
                            )                            
                            ''

                            print(
                                "Training acc coarse (for one batch) at step %d: %.4f"
                                % (step, float(train_acc_metric_coarse.result()))
                            )                            

                    train_acc = train_acc_metric.result()
                    Train_Acc[epoch]=train_acc

                    train_acc_coarse = train_acc_metric_coarse.result()
                    Train_Acc_coarse[epoch]=train_acc_coarse
                    
                    train_acc_coarse3 = train_acc_metric_coarse3.result()
                    Train_Acc_coarse3[epoch]=train_acc_coarse3                   
                    
                    train_acc_Hierarchical = train_acc_metric_Hierarchical.result()
                    Train_Acc_Hierarchical[epoch]=train_acc_Hierarchical
                
                    # Reset training metrics at the end of each epoch

                    train_acc_metric.reset_states()
                    train_acc_metric_coarse3.reset_states()
                    
                    train_acc_metric_coarse.reset_states()
                    train_acc_metric_Hierarchical.reset_states()
                
                    # Run a validation loo at the end of each epoch.
                    #for step, (x_batch_train, y_batch_train_coarse, y_train_coarse3) in enumerate(train_dataset_coarse):


                    val_acc = val_acc_metric.result()
                    val_acc_metric.reset_states()                                        
                    val_acc_coarse = val_acc_metric_coarse.result()
                    val_acc_metric_coarse.reset_states()
                    
                    val_acc_coarse1 = val_acc_metric_coarse1.result()
                    val_acc_metric_coarse1.reset_states()                    

                    print("Validation acc coarse1: %.4f" % (float(val_acc_coarse1),))
                    
                    
                    
                    print("Time taken coarse: %.2fs" % (time.time() - start_time))                     
                    

                    Val_Acc_coarse1[epoch] = val_acc_coarse1
                    
########################
                    for step_val, (x_batch_val,  y_batch_val_coarse, y_batch_val_coarse3) in enumerate(val_dataset_coarse):
                        #val_logits = model(x_batch_val, training=False)
                        x_batch_val_coarse3 = x_val_coarse[step_val:step_val+32,:,:,:]
                        #y_batch_val_coarse3 = y_val_coarse3[step_val:step_val+32,:]
                        #y_batch_val_coarse3 = y_batch_val_coarse
                        val_logits1_ = model(x_batch_val, training=False)
                        loss_value_coarse_val  = loss_fn(y_batch_val_coarse, val_logits1_)
                        

                        #val_logits2=np.zeros((val_logits2.shape))
                        #val_logits2 = model_Hierarchical(x_batch_val, training=False)
                        
                        #val_logits_coarse1 = model_coarse1(x_batch_val, training=False)
                        #val_logits_coarse2 = model_coarse2(x_batch_val, training=False)
                        final_conv_layer = model.layers[7]
                        
                        #val_logits_coarse1 = model_coarse1(final_conv_layer.output, training=False)
                        #val_logits_coarse2 = model_coarse2(final_conv_layer.output, training=False)
                        if MultiTask==1:
                           x_batch_val = get_output([x_batch_val])
                        val_logits_coarse3 = model_coarse3(x_batch_val, training=False)
                        loss_value_coarse3_val  = loss_fn(y_batch_val_coarse3, val_logits_coarse3)
                        
                        if MultiTask==1:
                        
                            val_CoarseClass =  np.argmax(val_logits_coarse3, axis=1)
                            
                            #logits_coarse = np.array(logits_coarse)
                            
                            val_Ind_ManMade = (val_CoarseClass>0)
                            val_Ind_Nature  = (val_CoarseClass<1)
                            
                            val_w_a = np.repeat(np.reshape(val_logits_coarse3[:,0],(32,1)), Nclasses_1, axis=1)
                            val_w_b = np.repeat(np.reshape(val_logits_coarse3[:,1],(32,1)), Nclasses_2, axis=1)
                            val_w_3[:,0:3] = val_w_a
                            val_w_3[:,3:]  = val_w_b
                            val_logits1=val_logits1_*val_w_3   
                            #logits_coarse = np.multiply(logits_coarse,w_3)

                        # Update val metrics
                        #val_acc_metric.update_state(y_batch_val, val_logits)
                        val_acc_metric_coarse.update_state(y_batch_val_coarse, val_logits1_)
                        val_acc_metric_coarse3.update_state(y_batch_val_coarse3, val_logits_coarse3)
                          
                    val_acc = val_acc_metric.result()
                    val_acc_metric.reset_states()                                        
                    val_acc_coarse = val_acc_metric_coarse.result()
                    val_acc_metric_coarse.reset_states()
                    
                    val_acc_coarse3 = val_acc_metric_coarse3.result()
                    val_acc_metric_coarse3.reset_states()                    
                    val_acc_Hierarchical = val_acc_metric_Hierarchical.result()
                    val_acc_metric_Hierarchical.reset_states()
                    print("Validation acc coarse: %.4f" % (float(val_acc_coarse),))
                    print("Validation acc coarse3: %.4f" % (float(val_acc_coarse3),))
                    
                    
                    
                    print("Time taken coarse: %.2fs" % (time.time() - start_time))                     
                    

                    Val_Acc[epoch]=val_acc
                    Val_Acc_coarse[epoch]=val_acc_coarse
                    Val_Acc_coarse3[epoch] = val_acc_coarse3
                    
                    Val_Acc_Hierarchical[epoch]=val_acc_Hierarchical
                if model_save==1:
                   if TSX_S1==1: 
                        model.save('model_TSXS1_trained_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')
                        model_coarse3.save('model_coarse3_TSXS1_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')

                   else:  
                        model.save('model_S1_trained_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')
                        model_coarse3.save('model_coarse3_S1_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')
                   
                       
                  
           else:         
                #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
                # Instantiate a loss function.
                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        end = time.time()
        print('training_time= ',end - start)

        
        

        #test_loss2, test_acc2 = myModel_1.evaluate(X_test_vgg, Y_test)
        #test_loss_coarse, test_acc_coarse = myModel_2.evaluate(X_test_vgg, Y_test_coarse)
        #test_loss_Hierarchical, test_acc_Hierarchical = model_Hierarchical.evaluate(X_test_vgg, Y_test)
        #test_loss_coarse, test_coarse = model_coarse.evaluate(X_test_vgg, Y_test)
        
        #myModel=model
        if NetType==3: 
            ss=np.linspace(0,Ntest-1,round(Ntest/10)-1)
            ss=ss.astype('int32')
            #test_loss, test_acc = myModel.evaluate(X_test_vgg[ss,:,:,:], Y_test[ss])
            #test_loss_coarse, test_acc_coarse = model_coarse.evaluate(X_test_vgg[ss,:,:,:], Y_test_coarse[ss])        



        test_labels_p=np.zeros((Ntest,Nclasses))
        start = time.time()
        test_labels_p=np.zeros((round(Ntest/Ntest_reduction),Nclasses))
        test_labels_p_coarse=np.zeros((round(Ntest/Ntest_reduction),Nclasses_3))
        
        test_labels_p_coarse1=np.zeros((round(Ntest/Ntest_reduction),Nclasses_1))
        test_labels_p_coarse2=np.zeros((round(Ntest/Ntest_reduction),Nclasses_2))

        test_LI=np.ones((round(Ntest/Ntest_reduction),Nclasses))
        #test_LI=np.zeros((round(Ntest/Ntest_reduction),Nclasses))
        
        test_labels_p_Hierarchical=np.zeros((Ntest_reduction,Nclasses))
        test_w_3=np.zeros((TestStep,Nclasses))

        if NetType==3:    
           
         for ideep in range(0,round(Ntest/Ntest_reduction)-TestStep,TestStep):#NTest_2
           test_labels_p[ideep:ideep+TestStep] = model(X_test_vgg[ideep:ideep+TestStep,:,:,:]) 
           
           if MultiTask==1:
               test_labels_p_coarse[ideep:ideep+TestStep]=model_coarse3(get_output(X_test_vgg[ideep:ideep+TestStep,:,:,:]))  
               #test_labels_p_coarse1[ideep:ideep+TestStep]=model_coarse1(X_test_vgg[ideep:ideep+TestStep,:,:,:])  
               #test_labels_p_coarse2[ideep:ideep+TestStep]=model_coarse2(X_test_vgg[ideep:ideep+TestStep,:,:,:])                

               test_CoarseClass =  np.argmax(test_labels_p_coarse[ideep:ideep+TestStep], axis=1)
               #test_CoarseClass =  np.argmax(test_labels_p_coarse[ideep], axis=1)
                
                #logits_coarse = np.array(logits_coarse)
               
               test_Ind_ManMade = (test_CoarseClass>0)
               test_Ind_Nature  = (test_CoarseClass<1)
                
                
               test_w_a = np.repeat(np.reshape(test_labels_p_coarse[ideep:ideep+TestStep,0],(TestStep,1)), Nclasses_1, axis=1)
               test_w_b = np.repeat(np.reshape(test_labels_p_coarse[ideep:ideep+TestStep,1],(TestStep,1)), Nclasses_2, axis=1)
               test_w_3[:,0:3] = test_w_a
               test_w_3[:,3:]  = test_w_b
               test_labels_p[ideep:ideep+TestStep]=model(X_test_vgg[ideep:ideep+TestStep,:,:,:])*test_w_3
               #test_labels_p[ideep:ideep+TestStep]=test_w_3


           else:
               test_labels_p_coarse[ideep:ideep+TestStep]=model_coarse3(([X_test_vgg[ideep:ideep+TestStep,:,:,:]]))  

         if MultiTask==1:

             test_CoarseClass =  np.argmax(test_labels_p_coarse, axis=1)
             test_Ind_ManMade = (test_CoarseClass>0)
             test_Ind_Nature  = (test_CoarseClass<1)
             test_LI [test_Ind_ManMade,3:8] = test_labels_p_coarse2[test_Ind_ManMade][:]
             test_LI [test_Ind_Nature,0:3] = test_labels_p_coarse1[test_Ind_Nature][:]
    
             test_labels_p = test_labels_p *test_LI

         #test_labels_p=model(X_test_vgg[0:round(Ntest/4),:,:,:])      
            
        else:
           test_labels_p = myModel.predict(X_test)
 
        end = time.time()

        print('testing_time= ',end - start)
  


