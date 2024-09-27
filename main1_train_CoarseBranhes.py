# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
from keras.models import Model
from keras import layers
import keras

#from keras import backend as K
#import tensorflow.keras.backend as K
from keras.callbacks import LearningRateScheduler
import numpy as np
import time
import math  
import os
import random
from keras.utils import np_utils
#CUDA_VISIBLE_DEVICES=0
import tensorflow as tf
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.client import device_lib


from tensorflow.compat.v1 import ConfigProto

import keras.backend as K
#import cv2
import numpy.matlib
from keras import callbacks

from keras.utils.vis_utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # CPU is used with -1. 0, 1 for GPU
print('GPU devise= ',tf.config.list_physical_devices('GPU'))

print(device_lib.list_local_devices())

Nclasses=8
Nclasses_coarse=2
Nclasses_1=3
Nclasses_2=5
Nclasses_3=2
VGG_manual=1
save_midel=0
Training_ratio=0.01
OnlyNatural_training=1
OnlyManMade_training=0

N_val=200
featurefusion=1
featurefusionType=2
onlyundeep=0

num_Deepfeatures=256
w1=1  #weight of deep features
w2=1  # weight of unlearning features


Multilooking=0
win_size=100 ; Nlook_a=3  ;Nlook_r=1 ; Ovrlp=21

NL1=32
NL2=32
Look_step=3

Augment=0
TestAugment=0
NAug=1*Augment+1


ElsevieData=1
S1=1
TSX=0
TSX_S1=0
load_from_file = 1
slcORspe=1#1 for spe and 2 for slc
train_Natural = 0# 1 for Natural training and else for ManMade training


NetType=3# 0 for CNN, 1 for A-Conv, 3 for VGG


Nepochs=50

BatchSize=32# 18 for AConv, 32 for VGG
fine_tuneing=1
ClassifierType=1 # 1 for SVM, 2 for RF and 3 for Knn

training_loop=1
Hierarchical=1

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



SecClassify=1



MNISTDATA=0
Skimagedata=0
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



save_results=0
h=42#28
w=42#12828
ww=42*Augment+w*(1-Augment)#28#
hh=42*Augment+h*(1-Augment)#28#    
DetectionResults_Show=0
featureMapPlot=0
iplot_feature=10
plot_Convergence=0 
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

NetworkAcc_tree1=np.zeros((1,Iteration_num))
NetworkF1_tree1=np.zeros((1,Iteration_num))
NetworkPrc_tree1=np.zeros((1,Iteration_num))
NetworkRec_tree1=np.zeros((1,Iteration_num))

NetworkAcc_tree2=np.zeros((1,Iteration_num))
NetworkF1_tree2=np.zeros((1,Iteration_num))
NetworkPrc_tree2=np.zeros((1,Iteration_num))
NetworkRec_tree2=np.zeros((1,Iteration_num))
for i_iter in range(Iteration_num):
    #i_iter_sum=i_iter
    #Adeli=1
    
    


    if ElsevieData==1:
      if S1==1:
        if load_from_file==1:
           train_images = np.load('train_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1'+ '.npy')
           train_labels = np.load('train_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1'+ '.npy') 
           train_labels_H = np.load('train_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1'+ '.npy')      
           
           #values_train = np.load('values_train_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')

           test_images = np.load('test_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1'+ '.npy')
           test_labels = np.load('test_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1'+ '.npy')
           #values_test = np.load('values_test_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           test_labels_H = np.load('test_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1'+ '.npy')      

        else:          
           from ImportFile import ImportData
        #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
           [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()
      elif TSX==1:
        from ImportFile_TSX import ImportData
        #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
        [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()
      elif TSX_S1==1:
        if load_from_file==1:
           train_images = np.load('train_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           train_labels = np.load('train_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy') 
           train_labels_H = np.load('train_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')      
           
           #values_train = np.load('values_train_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')

           test_images = np.load('test_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           test_labels = np.load('test_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           #values_test = np.load('values_test_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           test_labels_H = np.load('test_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')      

        else:
               from ImportFile import ImportData
               #[train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels]=ImportData()
               [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H]=ImportData()
                          
      else:
        if load_from_file==1:
           train_images = np.load('train_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           train_labels = np.load('train_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy') 
           train_labels_H = np.load('train_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')      
           
           #values_train = np.load('values_train_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')

           test_images = np.load('test_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           test_labels = np.load('test_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           #values_test = np.load('values_test_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')
           test_labels_H = np.load('test_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+ '.npy')      

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
    elif Nclasses==6:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4+Ntrain_5#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4+Ntest_5#10000#        
    elif Nclasses==7:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4+Ntrain_5+Ntrain_6#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4+Ntest_5+Ntest_6#10000#
        
    else:
        Ntrain=Ntrain_clutt+Ntrain_build+Ntrain_2+Ntrain_3+Ntrain_4+Ntrain_5+Ntrain_6+Ntrain_7#10000#
        Ntest=Ntest_clutt+Ntest_build+Ntest_2+Ntest_3+Ntest_4+Ntest_5+Ntest_6+Ntest_7#10000#


       
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
        
    plt.close('all')
    # Load data
    
    #Y_train = np_utils.to_categorical(train_labels,dtype="uint8")
    #Y_test = np_utils.to_categorical(test_labels,dtype="uint8")
    
    Y_train = np_utils.to_categorical(train_labels)
    Y_test = np_utils.to_categorical(test_labels)
    
    Y_train_3 = np_utils.to_categorical(train_labels_H)
    Y_test_3 = np_utils.to_categorical(test_labels_H) 
    
    Y_train_1 = np_utils.to_categorical(train_labels[0:Ntrain_build+Ntrain_clutt+Ntrain_2])
    Y_test_1 = np_utils.to_categorical(test_labels[0:Ntest_build+Ntest_clutt+Ntest_2])
    
    Y_train_2 = np_utils.to_categorical(train_labels[Ntrain_build+Ntrain_clutt+Ntrain_2:]-Nclasses_2+2)
    Y_test_2 = np_utils.to_categorical(test_labels[Ntest_build+Ntest_clutt+Ntest_2:]-Nclasses_2+2)
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
        if OnlyNatural_training==1:
           Ntrain = Ntrain_clutt+Ntrain_build+Ntrain_2
        elif OnlyManMade_training ==1:
           Ntrain = Ntrain_3+Ntrain_4+Ntrain_5+Ntrain_6+Ntrain_7
        N_val= np.floor(np.divide(Ntrain*0.15,BatchSize))* BatchSize#1984#9984#100000
        N_val=N_val.astype('uint16')        
        '''
            X_train = train_images
            X_test = test_images
        '''
        
        
        #_train /= 255
        #_test /= 255
        

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
        #Y_train = np_utils.to_categorical(train_labels)
        #Y_test = np_utils.to_categorical(test_labels)
        #Y_train = train_labels
        #Y_test = test_labels

        #==================================================
        # Creating our model

        myInput = layers.Input(shape=(32*slcORspe,32*slcORspe,3))

        #myInput = layers.Input(shape=(32,32))
        #conv1 = layers.Conv2D(4, 3, activation='relu', padding='same', strides=2)(myInput)
        #conv2 = layers.Conv2D(4, 3, activation='relu', padding='same', strides=2)(conv1)
        '''
        ###################### CNN
        #conv1 = layers.Conv2D(32, 5, activation='relu', padding='same', strides=2)(myInput)
        '''
        '''
        ################### A-CONVNet
        '''
        
        
        '''
        ################ CAE HL-CNN
        
        '''
        
        '''
        ###################### VGG
        '''
        if NetType==3 :
            config = ConfigProto()
            #config.gpu_options.per_process_gpu_memory_fraction = 0.333
            config.gpu_options.per_process_gpu_memory_fraction=0.830
            config.gpu_options.allow_growth = True            
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if len(physical_devices) > 0:
             tf.config.experimental.set_memory_growth(physical_devices[0], True)
             tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
            from tensorflow.keras import applications
            from tensorflow.keras.layers import Input
            
        
            
   
            
  
            
            if VGG_manual==1:
                
                ################
                myInput_9 = layers.Input(shape=(32*slcORspe,32*slcORspe,3))

                conv1_9 = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(myInput_9)
                conv1_9 = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(conv1_9)
                
                pool1_9 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_9)
                #conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
                conv2_9 = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool1_9)
                
                conv2_9 = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(conv2_9)
                pool2_9 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_9)

                flat_9 = layers.Flatten()(pool2_9)
                
                Dense1_9 = layers.Dense(4096, activation='relu')(flat_9)
                Dense2_9 = layers.Dense(4096, activation='relu')(Dense1_9)
                #Dense2_=layers.Dropout(409640(Dense1_)
                
                out_layer_vgg_9 = layers.Dense(Nclasses_1, activation='softmax')(Dense2_9)
                VGG_Manual_Natural_new_1= Model(inputs=myInput_9, outputs=out_layer_vgg_9)#myInput

                if TSX_S1==1: 
                        model_trained=load_model('model_TSXS1_trained_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')
                    

                else:  
                        model_trained=load_model('model_S1_trained_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')


                   

                VGG_Manual_Natural_new_1.layers[0].set_weights(model_trained.layers[0].get_weights())
                VGG_Manual_Natural_new_1.layers[1].set_weights(model_trained.layers[1].get_weights())
                VGG_Manual_Natural_new_1.layers[2].set_weights(model_trained.layers[2].get_weights())
                VGG_Manual_Natural_new_1.layers[3].set_weights(model_trained.layers[3].get_weights())
                VGG_Manual_Natural_new_1.layers[4].set_weights(model_trained.layers[4].get_weights())                
                
                VGG_Manual_Natural_new_1.layers[0].trainable = False
                VGG_Manual_Natural_new_1.layers[1].trainable = False
                VGG_Manual_Natural_new_1.layers[2].trainable = False
                VGG_Manual_Natural_new_1.layers[3].trainable = False
                VGG_Manual_Natural_new_1.layers[4].trainable = False

                ################
                myInput_10 = layers.Input(shape=(32*slcORspe,32*slcORspe,3))

                conv1_10 = layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(myInput_10)
                conv1_10= layers.Conv2D(64, 3, activation='relu', strides=1, padding="same")(conv1_10)
                
                pool1_10 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv1_10)
                #conv2 = layers.Conv2D(64, 3, activation='relu', padding='same', strides=2)(conv1)
                conv2_10 = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(pool1_10)
                
                conv2_10 = layers.Conv2D(128, 3, activation='relu', strides=1, padding="same")(conv2_10)
                pool2_10 = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None)(conv2_10)

                flat_10 = layers.Flatten()(pool2_10)
                
                Dense1_10 = layers.Dense(4096, activation='relu')(flat_10)
                Dense2_10 = layers.Dense(4096, activation='relu')(Dense1_10)
                #Dense2_=layers.Dropout(409640(Dense1_)
                
                out_layer_vgg_10 = layers.Dense(Nclasses_2, activation='softmax')(Dense2_10)
                VGG_Manual_ManMade_new_1= Model(inputs=myInput_10, outputs=out_layer_vgg_10)#myInput

                if TSX_S1==1: 
                        model_trained=load_model('model_TSXS1_trained_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')
                    

                else:  
                        model_trained=load_model('model_S1_trained_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')

                VGG_Manual_ManMade_new_1.layers[0].set_weights(model_trained.layers[0].get_weights())
                VGG_Manual_ManMade_new_1.layers[1].set_weights(model_trained.layers[1].get_weights())
                VGG_Manual_ManMade_new_1.layers[2].set_weights(model_trained.layers[2].get_weights())
                VGG_Manual_ManMade_new_1.layers[3].set_weights(model_trained.layers[3].get_weights())
                VGG_Manual_ManMade_new_1.layers[4].set_weights(model_trained.layers[4].get_weights())                
                
                VGG_Manual_ManMade_new_1.layers[0].trainable = False
                VGG_Manual_ManMade_new_1.layers[1].trainable = False
                VGG_Manual_ManMade_new_1.layers[2].trainable = False
                VGG_Manual_ManMade_new_1.layers[3].trainable = False
                VGG_Manual_ManMade_new_1.layers[4].trainable = False

                #####################################################  
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
                FC2 = layers.Dense(256, activation='relu')(flat2)#8192
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
        
        if NetType==3:
 
           hg=1
        else:
          myModel = Model(myInput, out_layer)#myInput
        
        
        if NetType==1 :
            i_featuremap=3

            print('NN is A-ConvNet')
        else:
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

           if training_loop==0:
               
              myModel.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
            
           elif Hierarchical==1:
                fine_tuneing=0


                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                
                # Prepare the training dataset.
                batch_size = BatchSize
                #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
                myInput = layers.Input(shape=(batch_size,32,32,3))
                #inputs = keras.Input(shape=(784,), name="digits")
                inputs = myInput                
                


                x_train_coarse1=X_train_vgg_2#for ManMade
                y_train_coarse1=Y_train_2#Y_train_coarse
                
                x_train_coarse2=X_train_vgg_1#X_train_vgg_1
                y_train_coarse2=Y_train_1#Y_train_coarse

                x_train_coarse3=X_train_vgg#X_train_vgg_1
                y_train_coarse3=Y_train_3#Y_train_coarse
                
                val_ind_coarse1=np.linspace(0,x_train_coarse1.shape[0]-1,x_train_coarse1.shape[0])
                random.shuffle(val_ind_coarse1)
                val_ind_coarse1=val_ind_coarse1.astype('int32')
                
                x_val_coarse1 = x_train_coarse1[val_ind_coarse1[0:N_val],:,:,:]
                y_val_coarse1 = y_train_coarse1[val_ind_coarse1[0:N_val]]
                
                val_ind_coarse2=np.linspace(0,x_train_coarse2.shape[0]-1,x_train_coarse2.shape[0])
                random.shuffle(val_ind_coarse2)
                val_ind_coarse2=val_ind_coarse2.astype('int32')
                
                x_val_coarse2 = x_train_coarse2[val_ind_coarse2[0:N_val],:,:,:]
                y_val_coarse2 = y_train_coarse2[val_ind_coarse2[0:N_val]]

                val_ind_coarse3=np.linspace(0,x_train_coarse3.shape[0]-1,x_train_coarse3.shape[0])
                random.shuffle(val_ind_coarse3)
                val_ind_coarse3=val_ind_coarse3.astype('int32')
                
                x_val_coarse3 = x_train_coarse3[val_ind_coarse3[0:N_val],:,:,:]
                y_val_coarse3 = y_train_coarse3[val_ind_coarse3[0:N_val]]                
                
                y_train_coarse1 = y_train_coarse1[val_ind_coarse1[N_val:]]                 
                x_train_coarse1 = x_train_coarse1[val_ind_coarse1[N_val:],:,:,:]

                y_train_coarse2 = y_train_coarse2[val_ind_coarse2[N_val:]]                 
                x_train_coarse2 = x_train_coarse2[val_ind_coarse2[N_val:],:,:,:]

                y_train_coarse3 = y_train_coarse3[val_ind_coarse3[N_val:]]                 
                x_train_coarse3 = x_train_coarse3[val_ind_coarse3[N_val:],:,:,:]
                


                
                # Prepare the training dataset.

                train_dataset_coarse1 = tf.data.Dataset.from_tensor_slices((x_train_coarse1, y_train_coarse1))
                train_dataset_coarse1 = train_dataset_coarse1.shuffle(buffer_size=1024).batch(batch_size)

                train_dataset_coarse2 = tf.data.Dataset.from_tensor_slices((x_train_coarse2, y_train_coarse2))
                train_dataset_coarse2 = train_dataset_coarse2.shuffle(buffer_size=1024).batch(batch_size)
                
                train_dataset_coarse3 = tf.data.Dataset.from_tensor_slices((x_train_coarse3, y_train_coarse3))
                train_dataset_coarse3 = train_dataset_coarse3.shuffle(buffer_size=1024).batch(batch_size)

                val_dataset_coarse1 = tf.data.Dataset.from_tensor_slices((x_val_coarse1, y_val_coarse1))
                val_dataset_coarse1 = val_dataset_coarse1.batch(batch_size)
                val_dataset_coarse2 = tf.data.Dataset.from_tensor_slices((x_val_coarse2, y_val_coarse2))
                val_dataset_coarse2 = val_dataset_coarse2.batch(batch_size)


                model_coarse1=VGG_Manual_ManMade_new_1#VGG_Manual_ManMade #VGG_Manual_Natural#VGG_Manual_coarse#myModel_coarse
                model_coarse2= VGG_Manual_Natural_new_1# VGG_Manual_Nature_new_1# VGG_Manual_Natural#VGG_Manual## #
                

                optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9)
                optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=.9)
                optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=.9)
                optimizer3 = tf.keras.optimizers.SGD(learning_rate=1e-3,momentum=.9)


                # Instantiate a loss function.
                loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                
                # Prepare the metrics.

                train_acc_metric_coarse1 = keras.metrics.CategoricalAccuracy()
                train_acc_metric_coarse2 = keras.metrics.CategoricalAccuracy()
                train_acc_metric_coarse3 = keras.metrics.CategoricalAccuracy()
                
                
                val_acc_metric_coarse1 = keras.metrics.CategoricalAccuracy()
                val_acc_metric_coarse2 = keras.metrics.CategoricalAccuracy()
                val_acc_metric_coarse3 = keras.metrics.CategoricalAccuracy()
                
                train_acc_coarse1=np.zeros((Nepochs,1))
                train_loss_coarse1=np.zeros((Nepochs,1)) 
                train_acc_coarse2=np.zeros((Nepochs,1))
                train_loss_coarse2=np.zeros((Nepochs,1)) 
                train_acc_coarse3=np.zeros((Nepochs,1))
                train_loss_coarse3=np.zeros((Nepochs,1)) 
                                
               
                import time
                val_acc_coarse1=np.zeros((Nepochs,1))
                val_loss_coarse1=np.zeros((Nepochs,1))
                val_acc_coarse2=np.zeros((Nepochs,1))
                val_loss_coarse2=np.zeros((Nepochs,1))                
                val_acc_coarse3=np.zeros((Nepochs,1))
                val_loss_coarse3=np.zeros((Nepochs,1))                
                #logits_Hierarchical=np.zeros(())
                
                epochs = Nepochs

                Train_Acc_coarse1=np.zeros((Nepochs,1))
                Val_Acc_coarse1=np.zeros((Nepochs ,1)) 
                
                Train_Acc_coarse2=np.zeros((Nepochs,1))
                Val_Acc_coarse2=np.zeros((Nepochs ,1)) 
                Train_Acc_coarse3=np.zeros((Nepochs,1))
                Val_Acc_coarse3=np.zeros((Nepochs ,1)) 
                
             
                
                for epoch in range(epochs):
                    print("\nStart of epoch %d" % (epoch,))
                    start_time = time.time()

                    
                    if train_Natural==1:
                        for step, (x_batch_train, y_batch_train_coarse1) in enumerate(train_dataset_coarse2):
                            xbtch=x_batch_train.numpy()
                            x_batch_train1=np.array(xbtch)                        
                            with tf.GradientTape() as tape1:
                                logits_coarse = model_coarse2(x_batch_train, training=True)
                                loss_value_coarse = loss_fn(y_batch_train_coarse1, logits_coarse)
    
                                
                            grads1 = tape1.gradient(loss_value_coarse, model_coarse2.trainable_weights)
                            optimizer1.apply_gradients(zip(grads1, model_coarse2.trainable_weights))                
                            train_acc_metric_coarse1.update_state(y_batch_train_coarse1, logits_coarse)
                            if step % 200 == 0:
                                print(
                                    "Training acc Hierarchical (for one batch) at step %d: %.4f"
                                    % (step, float(train_acc_metric_coarse1.result()))
                                )
                                print(
                                    "Training acc coarse (for one batch) at step %d: %.4f"
                                    % (step, float(train_acc_metric_coarse1.result()))
                                )                           
    
                                print(
                                    "Training acc (for one batch) at step %d: %.4f"
                                    % (step, float(train_acc_metric_coarse1.result()))
                                )                            
                                print("Seen so far: %d samples" % ((step + 1) * batch_size))
            
                    else:
                        for step, (x_batch_train, y_batch_train_coarse1) in enumerate(train_dataset_coarse1):
                            xbtch=x_batch_train.numpy()
                            x_batch_train1=np.array(xbtch)                        
                            with tf.GradientTape() as tape1:
                                logits_coarse = model_coarse1(x_batch_train, training=True)
                                loss_value_coarse = loss_fn(y_batch_train_coarse1, logits_coarse)
    
                                
                            grads1 = tape1.gradient(loss_value_coarse, model_coarse1.trainable_weights)
                            optimizer1.apply_gradients(zip(grads1, model_coarse1.trainable_weights))                
                            train_acc_metric_coarse1.update_state(y_batch_train_coarse1, logits_coarse)
                            if step % 200 == 0:
                                print(
                                    "Training acc Hierarchical (for one batch) at step %d: %.4f"
                                    % (step, float(train_acc_metric_coarse1.result()))
                                )
                                print(
                                    "Training acc coarse (for one batch) at step %d: %.4f"
                                    % (step, float(train_acc_metric_coarse1.result()))
                                )                           
    
                                print(
                                    "Training acc (for one batch) at step %d: %.4f"
                                    % (step, float(train_acc_metric_coarse1.result()))
                                )                            
                                print("Seen so far: %d samples" % ((step + 1) * batch_size))
                        
                    train_acc_coarse1 = train_acc_metric_coarse1.result()
                    Train_Acc_coarse1[epoch]=train_acc_coarse1

                    train_acc_coarse2 = train_acc_metric_coarse2.result()
                    Train_Acc_coarse2[epoch]=train_acc_coarse2
                    
                    train_acc_coarse3 = train_acc_metric_coarse3.result()
                    Train_Acc_coarse3[epoch]=train_acc_coarse3    
                    train_acc_metric_coarse1.reset_states()
                    train_acc_metric_coarse2.reset_states()
                    train_acc_metric_coarse3.reset_states()

                    if train_Natural==1:
                        for x_batch_val,  y_batch_val_coarse in val_dataset_coarse2:
                            #val_logits = model(x_batch_val, training=False)
                            
                            val_logits1 = model_coarse2(x_batch_val, training=False)
                            val_acc_metric_coarse1.update_state(y_batch_val_coarse, val_logits1)
                    else:
                        for x_batch_val,  y_batch_val_coarse in val_dataset_coarse1:
                            val_logits1 = model_coarse1(x_batch_val, training=False)
                            val_acc_metric_coarse1.update_state(y_batch_val_coarse, val_logits1)
                    val_acc_coarse1 = val_acc_metric_coarse1.result()
                    val_acc_metric_coarse1.reset_states()

                    val_acc_coarse2 = val_acc_metric_coarse2.result()
                    val_acc_metric_coarse2.reset_states()
                    
                    val_acc_coarse3 = val_acc_metric_coarse3.result()
                    val_acc_metric_coarse3.reset_states()
                        

                    print("Validation acccoarse1: %.4f" % (float(val_acc_coarse1),))
                    print("Validation acc coarse2: %.4f" % (float(val_acc_coarse2),))
                    print("Validation acc coarse3: %.4f" % (float(val_acc_coarse3),))
                    
                    
                    print("Time taken coarse: %.2fs" % (time.time() - start_time))                     
                    


                    

                    Val_Acc_coarse1[epoch]=val_acc_coarse1
                    Val_Acc_coarse2[epoch]=val_acc_coarse2
                    Val_Acc_coarse3[epoch]=val_acc_coarse3
                if save_midel==1:
                 if train_Natural==1:
                   model_coarse2.save('model_coarse1_S1_Natural_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')   
                 else:
                   model_coarse1.save('model_coarse2_S1_ManMade_training'+str(Training_ratio)+'lookstep'+str(Look_step)+'.h5')   
                     
           else:

                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                

        end = time.time()
        print('training_time= ',end - start)



