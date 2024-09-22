# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:11:04 2022

@author: zbheg

"""
import numpy as np
import time
import matplotlib.pyplot as plt

def Fisher_feature(train_images,test_images,num_components, Ntrain, Ntest, ww1,ww2 ):
   from sklearn.mixture import GaussianMixture
   X_train_fish=train_images.reshape(Ntrain*ww1*ww2,1)
   gm = GaussianMixture(n_components=num_components,covariance_type='diag', random_state=0).fit(X_train_fish)
   f_w=gm.weights_
   f_mu=gm.means_
   f_sigma=gm.covariances_
   u_ktn=np.zeros((Ntrain,ww1*ww2,num_components))
   X_nt=train_images.reshape(Ntrain,ww1*ww2)
   X_nt1=train_images.reshape(Ntrain,ww1*ww2,1)
   Xntk2=np.zeros((Ntrain,ww1*ww2,num_components))
   wu=np.zeros((Ntrain,ww1*ww2))
   gama_knt=np.zeros((Ntrain,ww1*ww2,num_components))
   
   feature_w=np.zeros((Ntrain,num_components))
   feature_mu=np.zeros((Ntrain,num_components))
   feature_sigma=np.zeros((Ntrain,num_components))
   f_sigmak=np.zeros((Ntrain,num_components))
   for k in range(num_components):
       argu=np.power(X_nt-f_mu[k],2)
       u_ktn[:,:,k]=np.exp(-argu/(2*np.power(f_sigma[k],1)))/np.power(2*np.pi*np.power(f_sigma[k],.5),.5)#  
       wu += f_w[k]*u_ktn[:,:,k]
   #u_ktn=gm.predict_proba(X_train_fish)
   for k in range(num_components):
       gama_knt[:,:,k]=np.divide(f_w[k]*u_ktn[:,:,k],wu)
   for k in range(num_components):
       f_ak=(np.sum(gama_knt,1)-f_w[k])/(ww1*ww2*np.power(f_w[k],.5)) 
       Gmu=(np.sum(gama_knt*f_mu[k],1))
       Gxnt=np.repeat(np.asarray(X_nt1),num_components,axis=2)
       f_muk=(np.sum(np.multiply(gama_knt,Gxnt),1)-Gmu)/(ww1*ww2*np.power(f_w[k],.5)*f_sigma[k]) 
       #aa=np.matmul((Gxnt-f_mu[k]).T,(Gxnt-f_mu[k]))/np.power(f_sigma[k])-1
       Xntk2[:,:,k]= np.power(X_nt-f_mu[k],2)-f_sigma[k]
       f_sigmak[:,k]=np.sum(np.multiply(gama_knt[:,:,k],Xntk2[:,:,k]),1)/(ww1*ww2*np.power(2*f_w[k],.5)*f_sigma[k])
   fisher_features_train= np.concatenate((f_ak,f_muk,f_sigmak),axis=1)   
   plt.figure(1001)
                                
   plt.imshow(fisher_features_train, cmap='gray')
   plt.show  
   
   # X_train_fish=test_images.reshape(Ntest*ww1*ww1,1)
   # gm = GaussianMixture(n_components=num_components,covariance_type='diag', random_state=0).fit(X_train_fish)
   # f_w=gm.weights_
   # f_mu=gm.means_
   # f_sigma=gm.covariances_
   start=time.time()
   u_ktn=np.zeros((Ntest,ww1*ww2,num_components))
   X_nt=test_images.reshape(Ntest,ww1*ww2)
   X_nt1=test_images.reshape(Ntest,ww1*ww2,1)
   Xntk2=np.zeros((Ntest,ww1*ww2,num_components))
   wu=np.zeros((Ntest,ww1*ww2))
   gama_knt=np.zeros((Ntest,ww1*ww2,num_components))
   
   feature_w=np.zeros((Ntest,num_components))
   feature_mu=np.zeros((Ntest,num_components))
   feature_sigma=np.zeros((Ntest,num_components))
   f_sigmak=np.zeros((Ntest,num_components))
   for k in range(num_components):
       argu=np.power(X_nt-f_mu[k],2)
       u_ktn[:,:,k]=np.exp(-argu/(2*np.power(f_sigma[k],1)))/np.power(2*np.pi*np.power(f_sigma[k],.5),.5)#  
       wu += f_w[k]*u_ktn[:,:,k]
   #u_ktn=gm.predict_proba(X_train_fish)
   for k in range(num_components):
       gama_knt[:,:,k]=np.divide(f_w[k]*u_ktn[:,:,k],wu)
   for k in range(num_components):
       f_ak=(np.sum(gama_knt,1)-f_w[k])/(ww1*ww2*np.power(f_w[k],.5)) 
       #f_ak=f_ak/np.max(f_ak)
       Gmu=(np.sum(gama_knt*f_mu[k],1))
       Gxnt=np.repeat(np.asarray(X_nt1),num_components,axis=2)
       f_muk=(np.sum(np.multiply(gama_knt,Gxnt),1)-Gmu)/(ww1*ww2*np.power(f_w[k],.5)*f_sigma[k]) 
       #f_muk=f_muk/np.max(f_muk)

       #aa=np.matmul((Gxnt-f_mu[k]).T,(Gxnt-f_mu[k]))/np.power(f_sigma[k])-1
       Xntk2[:,:,k]= np.power(X_nt-f_mu[k],2)-f_sigma[k]
       f_sigmak[:,k]=np.sum(np.multiply(gama_knt[:,:,k],Xntk2[:,:,k]),1)/(ww1*ww2*np.power(2*f_w[k],.5)*f_sigma[k])
       #f_sigmak=f_sigmak/np.max(f_sigmak)
   fisher_features_test= np.concatenate((f_ak,f_muk,f_sigmak),axis=1)   
   end=time.time()
   fisher_test_time=end-start
   '''
   plt.figure(1002)
                                
   plt.imshow(fisher_features_test, cmap='gray')
   plt.show 
   '''
   X_pred=gm.predict(X_train_fish,)
   num_unlearning_features_train_4=fisher_features_train.shape[1]
   unlearning_features_test_4=fisher_features_test
   unlearning_features_train_4=fisher_features_train
   return num_unlearning_features_train_4, unlearning_features_test_4, unlearning_features_train_4, fisher_features_test, fisher_features_train