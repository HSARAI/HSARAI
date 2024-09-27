# -*- coding: utf-8 -*-

import random
import glob
#import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randint

def ImportData():
    #from numpy.random import seed
    Augment=0
    NAug=1*Augment+1
    NL1=32
    NL2=32
    Look_step= 3
    Hw=32
    h=Hw
    w=Hw
    ww=Hw*Augment+w*(1-Augment)
    hh=Hw*Augment+h*(1-Augment)
       
    
    
        
    Nclasses=8
    Training_ratio=0.01
        
    NsubsetTrain=round(370*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1))#150
    Ntrain_build=round(1*NsubsetTrain*Training_ratio)#60000#
    Ntest_build=NsubsetTrain-Ntrain_build-1#NsubsetTrain-Ntrain_build
    
    NsubsetTrain_clutt=round(325*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1))#150
    
    Ntrain_clutt=round(1*NsubsetTrain_clutt*Training_ratio*NAug)#60000#
    NTest_clutt=NsubsetTrain_clutt-Ntrain_clutt-1#NsubsetTrain_clutt- Ntrain_clutt
    #Ntest=NTest_clutt+Ntest_build#10000#
    
    NsubsetTrain_2=round(350*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_2=round(1*NsubsetTrain_2*Training_ratio*NAug)#60000#

    NTest_2=NsubsetTrain_2-Ntrain_2-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_3=round(223*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_3=round(1*NsubsetTrain_3*Training_ratio*NAug)#60000#
    NTest_3=NsubsetTrain_3-Ntrain_3-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_4=round(372*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_4=round(1*NsubsetTrain_4*Training_ratio*NAug)#60000#
    NTest_4=NsubsetTrain_4-Ntrain_4-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_5=round(216*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_5=round(1*NsubsetTrain_5*Training_ratio*NAug)#60000#
    NTest_5=NsubsetTrain_5-Ntrain_5-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_6=round(342*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_6=round(1*NsubsetTrain_6*Training_ratio*NAug)#60000#
    NTest_6=NsubsetTrain_6-Ntrain_6-1#NsubsetTrain_clutt- Ntrain_clutt
    
    NsubsetTrain_7=round(352*(np.floor(NL1/Look_step)+1)*(np.floor(NL2/Look_step)+1)) #150
    Ntrain_7=round(1*NsubsetTrain_7*Training_ratio*NAug)#60000#
    NTest_7=NsubsetTrain_7-Ntrain_7-1#NsubsetTrain_clutt- Ntrain_clutt
    
    
    label11=0;label2=1;label3=2;label4=3;label5=4;label6=5;label7=6;label8=7;

    iii_plt=3000
    iaug_plt=1
    
    
    '''
    ##########################################################################
    1-
    '''
    values_train=[]
    values_test=[]
    
    if  Nclasses+1>1 :  
        images_path = "/agriculture/"
    

    
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+glob.glob(images_path + "*.jpg") 
            
        images.sort()
        images_train_1=images
        # = []
        Y = []
        X = []

            
        iii=0
        iiif=0
        for img in images:
            iii=iii+1

            image = np.load(img)

            image1 = image#[:,:,1]#-127
            #image1 = cv2.resize(image1, (ww,hh), interpolation = cv2.INTER_AREA)
           
            
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
             
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label11)
                        Y.append(label1)
        X1=X
        input_array1=np.array(X1,dtype=complex) 
        input_array=np.array(X,dtype=complex) 
        
        input_array1=np.array(X1,dtype=np.float16)
        input_array=np.array(X,dtype=np.float16)
        
        label_array=np.array(Y)
        train_images=np.zeros((Ntrain_build*NAug,ww,hh),dtype=complex)
        test_images=np.zeros(((Ntest_build)*NAug,ww,hh),dtype=complex)

        train_images=np.zeros((Ntrain_build*NAug,ww,hh),dtype=np.float16)
        test_images=np.zeros(((Ntest_build)*NAug,ww,hh),dtype=np.float16)    
        train_labels=np.zeros((Ntrain_build*NAug,))
        train_labels = train_labels.astype('int32')
        
        test_labels=np.zeros(((Ntest_build)*NAug,))
        test_labels = test_labels.astype('int32')
    
    
    
        #seed(1)
        values = randint(1, NsubsetTrain,NsubsetTrain)
        values1=values[0:Ntrain_build]
        #values_train.append(values1)
        
        values1=np.linspace(0,NsubsetTrain-1,NsubsetTrain)
        random.shuffle(values1)
        values1=values1.astype('int32')
    
        #values2=values[NsubsetTrain+1:Nsubset]
        values_train.append(values1[0:Ntrain_build])
    
        values1_aug=np.zeros((Ntrain_build*NAug,))
        for ir in range(Ntrain_build):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
        for ir in range(Ntest_build):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            test_images[(ir)*NAug:(ir)*NAug+NAug]=input_array[values1[ir+Ntrain_build-1]*NAug:values1[ir+Ntrain_build-1]*NAug+NAug]
    
            test_labels[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_build-1]*NAug:values1[ir+Ntrain_build-1]*NAug+NAug]
    
        train_labels_H=np.zeros((train_labels.shape[0]))
        test_labels_H=np.zeros((test_labels.shape[0]))    
    '''
    #################################################################################
    2-train
    
    '''
    if  Nclasses+1>2 :  
        
        images_path = "/forest/"
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []        
        X = []
        iii=0
        for img in images:
            iii=iii+1
            image = np.load(img)
            image1 = image#[:,:,1]#-127
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label2)
                        Y.append(label1) 
        X1=X
        input_array=np.array(X,dtype=complex) 
        
        input_array=np.array(X,dtype=np.float16) 
        label_array=np.array(Y) 
        train_images1=np.zeros((Ntrain_clutt*NAug,ww,hh),dtype=complex)
        test_images1=np.zeros(((NTest_clutt)*NAug,ww,hh),dtype=complex)
        train_images1=np.zeros((Ntrain_clutt*NAug,ww,hh),dtype=np.float16)
        test_images1=np.zeros(((NTest_clutt)*NAug,ww,hh),dtype=np.float16)
    
        train_labels1=np.zeros((Ntrain_clutt*NAug,))
        train_labels1 = train_labels1.astype('int32')
        test_labels1=np.zeros(((NTest_clutt)*NAug,))
        test_labels1 = test_labels1.astype('int32')
        
        values = randint(1, NsubsetTrain_clutt,NsubsetTrain_clutt)
        values1=values[0:Ntrain_clutt]
        
        values1=np.linspace(0,NsubsetTrain_clutt-1,NsubsetTrain_clutt)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_clutt])
        for ir in range(Ntrain_clutt):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images1[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels1[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_clutt):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_clutt]
    
            test_images1[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_clutt]*NAug:values1[ir+Ntrain_clutt]*NAug+NAug]
    
            test_labels1[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_clutt]*NAug:values1[ir+Ntrain_clutt]*NAug+NAug]
        train_labels1_H=np.zeros((train_labels1.shape[0]))
        test_labels1_H=np.zeros((test_labels1.shape[0]))     
      
    '''
    #################################################################################
    3-train
    
    '''
    if  Nclasses+1>3 :  
        
        images_path = "/water/"
    
        #images_path = "I:/NN/dataset/Adeli/Clutteregypt/"
    
        #images_path = "C:/NN/train/train/ZSU_23_4/"
        #images_path = "C:/Users/jooya/Desktop/Clutter-egypt/"
    
        
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []
        
        X = []
        #width = 158
        #height = 158
        iii=0
        for img in images:
            iii=iii+1
            image = np.load(img)
            #image = cv2.resize(image, (width, height))
            #image = image.astype('int32')
            #cv2.imwrite(os.path.join(images_path , 'Residental'+str(iii)+'.jpg'),abs(image))
    
            image1 = image#[:,:,1]#-127
            #image1 = cv2.resize(image1, (ww,hh), interpolation = cv2.INTER_AREA)
           
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
            
            #            image= image / np.max(image)
                        #image= image 
             
                      #image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label3)
                        Y.append(label1)
        X1=X
        input_array=np.array(X,dtype=complex) 
        
        input_array=np.array(X,dtype=np.float16) 
        label_array=np.array(Y) 
        train_images2=np.zeros((Ntrain_2*NAug,ww,hh),dtype=complex)
        test_images2=np.zeros(((NTest_2)*NAug,ww,hh),dtype=complex)
        train_images2=np.zeros((Ntrain_2*NAug,ww,hh),dtype=np.float16)
        test_images2=np.zeros(((NTest_2)*NAug,ww,hh),dtype=np.float16)
        
        train_labels2=np.zeros((Ntrain_2*NAug,))
        train_labels2 = train_labels2.astype('int32')
        #test_images2=np.zeros(((NTest_2)*NAug,ww,hh),dtype=complex)
        test_labels2=np.zeros(((NTest_2)*NAug,))
        test_labels2 = test_labels2.astype('int32')
        
        values = randint(1, NsubsetTrain_2,NsubsetTrain_2)
        values1=values[0:Ntrain_2]
        
        values1=np.linspace(0,NsubsetTrain_2-1,NsubsetTrain_2)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_2])
        #values2=values[NsubsetTrain+1:Nsubset]
        #values1_aug=np.zeros((NTrain_clutt*NAug,))
        
        for ir in range(Ntrain_2):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images2[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels2[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_2):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_2]
    
            test_images2[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_2]*NAug:values1[ir+Ntrain_2]*NAug+NAug]
    
            test_labels2[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_2]*NAug:values1[ir+Ntrain_2]*NAug+NAug]
    
      
        train_labels2_H=np.zeros((train_labels2.shape[0]))
        test_labels2_H=np.zeros((test_labels2.shape[0]))     
      
    '''
    #################################################################################
    4-train
    
    '''
    if  Nclasses+1>4 :  
        
        # Loading train images
        images_path = "/industrialbuilding/"
    
        #images_path = "I:/NN/dataset/Adeli/Clutteregypt/"
    
        #images_path = "C:/NN/train/train/ZSU_23_4/"
        #images_path = "C:/Users/jooya/Desktop/Clutter-egypt/"
    
        
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []        
        X = []
        #width = 158
        #height = 158
        iii=0
        for img in images:
            iii=iii+1
            image = np.load(img)

            image1 = image#[:,:,1]#-127
           
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label4)
                        Y.append(label1)
        X1=X
        input_array=np.array(X,dtype=complex)
        input_array=np.array(X,dtype=np.float16)         
        label_array=np.array(Y) 
        train_images3=np.zeros((Ntrain_3*NAug,ww,hh),dtype=complex)
        test_images3=np.zeros(((NTest_3)*NAug,ww,hh),dtype=complex)
 
        train_images3=np.zeros((Ntrain_3*NAug,ww,hh),dtype=np.float16)
        test_images3=np.zeros(((NTest_3)*NAug,ww,hh),dtype=np.float16)
        train_labels3=np.zeros((Ntrain_3*NAug,))
        train_labels3 = train_labels3.astype('int32')
        #test_images3=np.zeros(((NTest_3)*NAug,ww,hh),dtype=complex)
        test_labels3=np.zeros(((NTest_3)*NAug,))
        test_labels3 = test_labels3.astype('int32')
        
        values = randint(1, NsubsetTrain_3,NsubsetTrain_3)
        values1=values[0:Ntrain_3]
        
        values1=np.linspace(0,NsubsetTrain_3-1,NsubsetTrain_3)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_3])
        values_test=values_train
    
        #values2=values[NsubsetTrain+1:Nsubset]
        #values1_aug=np.zeros((NTrain_clutt*NAug,))
        
        for ir in range(Ntrain_3):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images3[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels3[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_3):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_3]
    
            test_images3[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_3]*NAug:values1[ir+Ntrain_3]*NAug+NAug]
    
            test_labels3[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_3]*NAug:values1[ir+Ntrain_3]*NAug+NAug]
        train_labels3_H=np.ones((train_labels3.shape[0]))
        test_labels3_H=np.ones((test_labels3.shape[0]))       
    '''
    #################################################################################
    5-
    
    '''
    if  Nclasses+1>5 :  
        
        # Loading train images
        images_path = "/skyscraper/"        
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []
        
        X = []
        #width = 158
        #height = 158
        iii=0
        for img in images:
            iii=iii+1
            image = np.load(img)

            image1 = image#[:,:,1]#-127

            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label5)
                        Y.append(label1)
        X1=X
        input_array=np.array(X,dtype=complex) 
        input_array=np.array(X,dtype=np.float16) 
        
        label_array=np.array(Y) 
        train_images4=np.zeros((Ntrain_4*NAug,ww,hh),dtype=complex)
        test_images4=np.zeros(((NTest_4)*NAug,ww,hh),dtype=complex)
        train_images4=np.zeros((Ntrain_4*NAug,ww,hh),dtype=np.float16)
        test_images4=np.zeros(((NTest_4)*NAug,ww,hh),dtype=np.float16)    
        train_labels4=np.zeros((Ntrain_4*NAug,))
        train_labels4 = train_labels4.astype('int32')
        #test_images4=np.zeros(((NTest_4)*NAug,ww,hh),dtype=complex)
        test_labels4=np.zeros(((NTest_4)*NAug,))
        test_labels4 = test_labels4.astype('int32')
        
        values = randint(1, NsubsetTrain_4,NsubsetTrain_4)
        values1=values[0:Ntrain_4]
        
        values1=np.linspace(0,NsubsetTrain_4-1,NsubsetTrain_4)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_4])
        values_test=values_train
        
        for ir in range(Ntrain_4):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images4[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels4[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_4):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_4]
    
            test_images4[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_4]*NAug:values1[ir+Ntrain_4]*NAug+NAug]
    
            test_labels4[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_4]*NAug:values1[ir+Ntrain_4]*NAug+NAug]
        train_labels4_H=np.ones((train_labels4.shape[0]))
        test_labels4_H=np.ones((test_labels4.shape[0]))     
    '''
    #################################################################################
    6-
    
    '''
    if  Nclasses+1>6 :  
        
        # Loading train images
        images_path = "/container/"
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []
        
        X = []

        iii=0
        for img in images:
            iii=iii+1
            image = np.load(img)

            image1 = image#[:,:,1]#-127
           
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
            
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label6)
                        Y.append(label1)
        X1=X
        input_array=np.array(X,dtype=np.float16) 
        label_array=np.array(Y) 
        train_images5=np.zeros((Ntrain_5*NAug,ww,hh),dtype=np.float16)
        test_images5=np.zeros(((NTest_5)*NAug,ww,hh),dtype=np.float16)
    
        train_labels5=np.zeros((Ntrain_5*NAug,))
        train_labels5 = train_labels5.astype('int32')
        test_images5=np.zeros(((NTest_5)*NAug,ww,hh),dtype=np.float16)
        test_labels5=np.zeros(((NTest_5)*NAug,))
        test_labels5 = test_labels5.astype('int32')
        
        values = randint(1, NsubsetTrain_5,NsubsetTrain_5)
        values1=values[0:Ntrain_5]
        
        values1=np.linspace(0,NsubsetTrain_5-1,NsubsetTrain_5)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_5])
        values_test=values_train
    
        #values2=values[NsubsetTrain+1:Nsubset]
        #values1_aug=np.zeros((NTrain_clutt*NAug,))
        
        for ir in range(Ntrain_5):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images5[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels5[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_5):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_5]
    
            test_images5[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_5]*NAug:values1[ir+Ntrain_5]*NAug+NAug]
    
            test_labels5[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_5]*NAug:values1[ir+Ntrain_5]*NAug+NAug]
        train_labels5_H=np.ones((train_labels5.shape[0]))
        test_labels5_H=np.ones((test_labels5.shape[0]))       
    '''
    #################################################################################
    7-
    
    '''
    if  Nclasses+1>7 :  
        
        # Loading train images
        images_path = "/storagetank/"
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []
        
        X = []
        #width = 158
        #height = 158
        iii=0
        for img in images:
            iii=iii+1
            image = np.load(img)
            #image = cv2.resize(image, (width, height))
            #image = image.astype('int32')
            #cv2.imwrite(os.path.join(images_path , 'Residental'+str(iii)+'.jpg'),abs(image))
    
            image1 = image#[:,:,1]#-127
            #image1 = cv2.resize(image1, (ww,hh), interpolation = cv2.INTER_AREA)
           
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label7)
                        Y.append(label1)
        X1=X
        input_array=np.array(X,dtype=np.float16) 
        label_array=np.array(Y) 
        train_images6=np.zeros((Ntrain_6*NAug,ww,hh),dtype=np.float16)
        test_images6=np.zeros(((NTest_6)*NAug,ww,hh),dtype=np.float16)
    
        train_labels6=np.zeros((Ntrain_6*NAug,))
        train_labels6 = train_labels6.astype('int32')
        test_images6=np.zeros(((NTest_6)*NAug,ww,hh),dtype=np.float16)
        test_labels6=np.zeros(((NTest_6)*NAug,))
        test_labels6 = test_labels6.astype('int32')
        
        values = randint(1, NsubsetTrain_6,NsubsetTrain_6)
        values1=values[0:Ntrain_6]
        
        values1=np.linspace(0,NsubsetTrain_6-1,NsubsetTrain_6)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_6])
        values_test=values_train
        for ir in range(Ntrain_6):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images6[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels6[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_6):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_6]
    
            test_images6[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_6]*NAug:values1[ir+Ntrain_6]*NAug+NAug]
    
            test_labels6[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_6]*NAug:values1[ir+Ntrain_6]*NAug+NAug]
        train_labels6_H=np.ones((train_labels6.shape[0]))
        test_labels6_H=np.ones((test_labels6.shape[0]))    
    '''
    #################################################################################
    8-
    
    '''
    if  Nclasses+1>8 :  
        
        # Loading train images
        images_path = "/residential/"
        images = glob.glob(images_path + "*.npy") + glob.glob(images_path + "*.jpeg")+ glob.glob(images_path + "*.jpg")
        images.sort()
        images.sort()
        images_train_2=images
        
        Y = []
        
        X = []
        #width = 158
        #height = 158
        iii=0
        for img in images:                                                                                                  
            iii=iii+1
            image = np.load(img)
            #image =                      j  [uopou[uo[ /lo[u o[o[[uo[uou;o[u[utj6;tyg[; 76ytg[ 6uthg 0o439etrd xkcv2.resize(image, (width, height))
            #image = image.astype('int32')
            #cv2.imwrite(os.path.join(images_path , 'Residental'+str(iii)+'.jpg'),abs(image))
    
            image1 = image#[:,:,1]#-127
            #image1 = cv2.resize(image1, (ww,hh), interpolation = cv2.INTER_AREA)
           
            center = image1.shape 
            for iaug1 in range(0,NL1,Look_step):
                    for iaug in range(0,NL2,Look_step):
                        iiif=iiif+1
                        #seed(2)
                        x = center[1]/2 - w/2 +randint(0, w-ww+1,1)*Augment
                        #seed(1)
                        y = center[0]/2 - h/2 +randint(0, h-hh+1,1)*Augment
                        #image = image1[int(y):int(y+hh), int(x):int(x+ww)]
                        image = image1[:,:, iaug1,iaug]
            
            #            image= image / np.max(image)
                        #image= image 
             
                      #image = cv2.normalize(image, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                        if iii==iii_plt :
                           if iaug==iaug_plt:
                             plt.figure()
                             plt.imshow(image)
                             plt.gray()
                             plt.title('x = '+str(x)+'y = '+str(y))
                             plt.show()
            
                        X.append(image)
                        label1=np.array(label8)
                        Y.append(label1)
        X1=X
        input_array=np.array(X,dtype=np.float16) 
        label_array=np.array(Y) 
        train_images7=np.zeros((Ntrain_7*NAug,ww,hh),dtype=np.float16)
        test_images7=np.zeros(((NTest_7)*NAug,ww,hh),dtype=np.float16)
    
        train_labels7=np.zeros((Ntrain_7*NAug,))
        train_labels7 = train_labels7.astype('int32')
        test_images7=np.zeros(((NTest_7)*NAug,ww,hh),dtype=np.float16)
        test_labels7=np.zeros(((NTest_7)*NAug,))
        test_labels7 = test_labels7.astype('int32')
        
        values = randint(1, NsubsetTrain_7,NsubsetTrain_7)
        values1=values[0:Ntrain_7]
        
        values1=np.linspace(0,NsubsetTrain_7-1,NsubsetTrain_7)
        random.shuffle(values1)
        values1=values1.astype('int32')
        
        values_train.append(values1[0:Ntrain_7])
        values_test=values_train
        #values2=values[NsubsetTrain+1:Nsubset]
        #values1_aug=np.zeros((NTrain_clutt*NAug,))
        
        for ir in range(Ntrain_7):
            #values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir]
            train_images7[ir*NAug:ir*NAug+NAug]=input_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
            train_labels7[ir*NAug:ir*NAug+NAug]=label_array[values1[ir]*NAug:values1[ir]*NAug+NAug]
    
        for ir in range(NTest_7):
    
            values1_aug[ir*NAug:ir*NAug+NAug]=np.zeros((NAug,))+values1[ir+Ntrain_7]
    
            test_images7[ir*NAug:ir*NAug+NAug]=input_array[values1[ir+Ntrain_7]*NAug:values1[ir+Ntrain_7]*NAug+NAug]
    
            test_labels7[ir*NAug:ir*NAug+NAug]=label_array[values1[ir+Ntrain_7]*NAug:values1[ir+Ntrain_7]*NAug+NAug]
     
        train_labels7_H=np.ones((train_labels7.shape[0]))
        test_labels7_H=np.ones((test_labels7.shape[0]))        
    


     
            

    if  Nclasses==2 : 
        train_images=np.concatenate((train_images, train_images1), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1), axis=0)
        test_images=np.concatenate((test_images, test_images1), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1), axis=0)
        train_labels_H=np.concatenate((train_labels_H, train_labels1_H), axis=0)
        test_labels_H=np.concatenate((test_labels_H, test_labels1_H), axis=0)         
    
    if  Nclasses==3 : 
        train_images=np.concatenate((train_images, train_images1,train_images2), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1,train_labels2), axis=0)
        test_images=np.concatenate((test_images, test_images1,test_images2), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1,test_labels2), axis=0)
        
    if  Nclasses==4 : 
        train_images=np.concatenate((train_images, train_images1,train_images2,train_images3), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1,train_labels2,train_labels3), axis=0)
        test_images=np.concatenate((test_images, test_images1,test_images2,test_images3), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1,test_labels2,test_labels3), axis=0)    
        train_labels_H=np.concatenate((train_labels_H, train_labels1_H,train_labels2_H,train_labels3_H), axis=0)

        test_labels_H=np.concatenate((test_labels_H, test_labels1_H,test_labels2_H,test_labels3_H), axis=0)    
        
    if  Nclasses==5 : 
        train_images=np.concatenate((train_images, train_images1,train_images2,train_images3, train_images4), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1,train_labels2,train_labels3,train_labels4), axis=0)
        test_images=np.concatenate((test_images, test_images1,test_images2,test_images3,test_images4), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1,test_labels2,test_labels3,test_labels4), axis=0)    
    if  Nclasses==6 : 
        train_images=np.concatenate((train_images, train_images1,train_images2,train_images3, train_images4, train_images5), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1,train_labels2,train_labels3,train_labels4,train_labels5), axis=0)
        test_images=np.concatenate((test_images, test_images1,test_images2,test_images3,test_images4,test_images5), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1,test_labels2,test_labels3,test_labels4,test_labels5), axis=0)    
        train_labels_H=np.concatenate((train_labels_H, train_labels1_H,train_labels2_H,train_labels3_H,train_labels4_H,train_labels5_H), axis=0)

        test_labels_H=np.concatenate((test_labels_H, test_labels1_H,test_labels2_H,test_labels3_H,test_labels4_H,test_labels5_H), axis=0)    


    if  Nclasses==7 : 
        train_images=np.concatenate((train_images, train_images1,train_images2,train_images3, train_images4, train_images5, train_images6), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1,train_labels2,train_labels3,train_labels4,train_labels5,train_labels6), axis=0)
        test_images=np.concatenate((test_images, test_images1,test_images2,test_images3,test_images4,test_images5,test_images6), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1,test_labels2,test_labels3,test_labels4,test_labels5,test_labels6), axis=0)    
        train_labels_H=np.concatenate((train_labels_H, train_labels1_H,train_labels2_H,train_labels3_H,train_labels4_H,train_labels5_H,train_labels6_H), axis=0)

        test_labels_H=np.concatenate((test_labels_H, test_labels1_H,test_labels2_H,test_labels3_H,test_labels4_H,test_labels5_H,test_labels6_H), axis=0)    

    if  Nclasses==8 : 
        train_images=np.concatenate((train_images, train_images1,train_images2,train_images3, train_images4, train_images5, train_images6, train_images7), axis=0)
        train_labels=np.concatenate((train_labels, train_labels1,train_labels2,train_labels3,train_labels4,train_labels5,train_labels6, train_labels7), axis=0)
        test_images=np.concatenate((test_images, test_images1,test_images2,test_images3,test_images4,test_images5,test_images6,test_images7), axis=0)
        test_labels=np.concatenate((test_labels, test_labels1,test_labels2,test_labels3,test_labels4,test_labels5,test_labels6,test_labels7), axis=0)    
        train_labels_H=np.concatenate((train_labels_H, train_labels1_H,train_labels2_H,train_labels3_H,train_labels4_H,train_labels5_H,train_labels6_H, train_labels7_H), axis=0)
        test_labels_H=np.concatenate((test_labels_H, test_labels1_H,test_labels2_H,test_labels3_H,test_labels4_H,test_labels5_H,test_labels6_H,test_labels7_H), axis=0)    
        np.save('test_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1',test_images)
        np.save('test_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1',test_labels)
        #np.save('values_test_training'+str(Training_ratio) +'_lookstep'+str(Look_step),values_test)
        np.save('test_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1',test_labels_H)
        
        np.save('train_images_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1',train_images)
        np.save('train_labels_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1',train_labels)
        np.save('train_labels_H_training'+str(Training_ratio) +'_lookstep'+str(Look_step)+'_S1',train_labels_H)

    return train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7,train_labels_H, test_labels_H





if __name__ == '__main__':

 [train_images ,values_train,images_train_1,images_train_2,train_labels,values_test,test_images,test_labels,Ntrain_build, Ntrain_clutt,Ntrain_2,Ntrain_3,Ntrain_4,Ntrain_5,Ntrain_6,Ntrain_7, train_labels_H, test_labels_H]=ImportData()
 