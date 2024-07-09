import pandas as pd
import numpy as np
from config import get_config
from utils import prepare_dirs_and_logger,save_config
from tboard import file2number
from toydata import get_toy_data

import tensorflow as tf
import math
import numpy as np
import os
import sys
import glob2
from itertools import product
from tqdm import trange
import time
import copy

from ArrayDict import ArrayDict
from config import get_config
from nonlinearities import name2nonlinearity
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

#import sympy
#from sympy import symbols
#from sympy.logic.boolalg import And,Or
#from sympy import Max,Min,srepr
#from sympy.utilities.lambdify import lambdify


from nonlinearities import relu#np version
from vis_utils import (split_posneg , get_path, get_np_network,
                        get_neuron_values, splitL, load_weights,
                        resample_grid,vec_get_neuron_values,
                        get_del_weights  )
from calc_maps import get_net_states
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

'''
This document is designed to compare the path classifier wbar and maxmargin
from svm


'''



'''
Needs:
    load data
    when does train error -> 0

(for just last time point at first):
    sklearn libsvm (either linear or callable kernel)
        C=1e-5
        .fit()

Optional:
    calc gradient of input for svm classifier
        (eval e1,e2 at x0)

'''

def get_state_fn(weights):
    def get_states(X):
        #Layers*(wpad,xpad,trn_iter,layersize)#(in general)
        PLayers=vec_get_neuron_values(X,weights)#d*(len(X)xnl)#in this code
        states=[(p>=0).astype('int') for p in PLayers     ][:-1]#skip output
        return states
    return get_states


def get_gram_fn(Bweights,get_states):

    def get_gram(X1,X2):
        states1=get_states(X1)
        states2=get_states(X2)

        X1s=[X1]+[B*S for B,S in zip(Bweights[:-1],states1)]#d+1
        X2s=[X2]+[B*S for B,S in zip(Bweights[:-1],states2)]

        layers_kernels=[]#paths starting at a layer
        for lyr, (x1,x2) in enumerate(zip(X1s,X2s)):
            #lyr 0,..,d
            sta1=[x1]+states1[lyr:]
            sta2=[x2]+states2[lyr:]
            l_kers=[np.dot(s1,s2.T) for s1,s2 in zip(sta1,sta2)]
            layers_kernels.append(l_kers)
        l_grams=[np.multiply.reduce(kers) for kers in layers_kernels]
        gram=sum(l_grams)
        return gram
    return get_gram

from vis_utils import paint_data, paint_binary_contours
#def paint_data(ax,npX,npY):
#    Xpos=npX[np.where(npY>0)[0]]
#    Xneg=npX[np.where(npY<0)[0]]
#    ax.xaxis.set_ticks_position('none') # tick markers
#    ax.yaxis.set_ticks_position('none')
#    ax.set_xticks([])
#    ax.set_yticks([])
#    ax.scatter(Xpos[:,0],Xpos[:,1],marker='+',s=600,c='b',linewidth='3')
#    ax.scatter(Xneg[:,0],Xneg[:,1],marker='_',s=600,c='r',linewidth='3')

#def paint_contours(ax,eval_fn,cbar=False):
#    lvls=np.arange(-1,1.,0.2)
#    cmap = plt.cm.get_cmap("Purples")
#    ctf=ax.contourf(gX0,gX1,eval_fn,
#    #                   levels=[-1.01,0,1.01],
#                      levels=lvls,
#                      cmap=plt.get_cmap('PuOr'),
#                      alpha=0.5,
#                     )
#    if cbar:
#        plt.colorbar(ctf,ax=ax)


def do_outer(A1,A2):
    N=len(A1)
    assert(len(A1)==len(A2))
    A1=A1.reshape(A1.shape+tuple(np.ones_like(A2.shape[1:])))
    A2=A2.reshape((N,)+tuple(np.ones_like(A1.shape[1:]))+A2.shape[1:])
    return A1*A2






if __name__=='__main__':
    plt.close('all')
    ##Maybe just use the paths corresp to largest wbar


    from vis_utils import Pub_Model_Dirs
    ###Index by [arch#-1][data#-1]

    ##Config##
    Arch=3
    Data=3
    log_dir=Pub_Model_Dirs[Arch-1][Data-1]
    descrip,id_str=log_dir.split('_')[-1],str(file2number(log_dir))
    print 'using log_dir:',log_dir,' descrip:',descrip, 'id str:',id_str



    record_dir=os.path.join(log_dir,'records')
    pactensor_dir=os.path.join(log_dir,'pac-tensor')
    if not os.path.exists(pactensor_dir):
        os.makedirs(pactensor_dir)

    all_step=np.load(get_path('step','wwatch',log_dir))
    all_weights=load_weights(log_dir)#d*2*Txn1xn2
    del_weights=get_del_weights(all_weights)
    arch=[b.shape[-1] for w,b in del_weights[:-1]]#net architecture
    iter_slice=np.arange(len(all_step))

    #dt=10#every 100
    #iter_slice=iter_slice[::dt]
    #print 'dt=',dt
    iter_slice=iter_slice[-1:]#just last entry
    #W_weights,b_weights=zip(*weights)

    step=all_step[iter_slice]
    step_weights=[[w[iter_slice],b[iter_slice]] for w,b in all_weights]
    step_dweights=[[w[iter_slice],b[iter_slice]] for w,b in del_weights]#d*2*dTxn1xn2
    time_weights=[[w[-1],b[-1]] for w,b in del_weights]#d*2*n1xn2


    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)
    m_samples,xdim=npX.shape
    gridX=np.load(get_path('gridX','hmwatch',log_dir))
    #res=200
    res=700
    gX=resample_grid(gridX,res)
    print 'grid res',res
    #gX=gridX #my poor computer

    gX0,gX1=gX[:,:,0],gX[:,:,1]
    X,Y=npX,npY.ravel()


#Your kernel must take as arguments two matrices of shape (n_samples_1, n_features),
#(n_samples_2, n_features) and return a kernel matrix of shape (n_samples_1, n_samples_2).


    trn_iter=-1
    weights=[[w[trn_iter],b[trn_iter]] for w,b in step_dweights]#d*2*wtshape
    Wweights,Bweights=zip(*weights)


    #############Begin SVM###################

    get_states=get_state_fn(weights)
    get_gram=get_gram_fn(Bweights,get_states)
    X1,X2=X,X
    gram=get_gram(X1,X2)

    #Doesnt help
    #scaler = StandardScaler( with_std=False )
    #scaler = StandardScaler()
    #scaler = MinMaxScaler([-1.,1.],copy=False)
    #gram=scaler.fit_transform(gram)



    #Debug Gram#
    #This code worked..#
#
#    get_gram=get_gram_fn(Bweights,get_states)
#
#    X1,X2=X,X
#    #Debug Gram#
#    states1=get_states(X1)
#    states2=get_states(X2)
#
#    #Consider centering
#
#    X1s=[X1]+states1
#    X2s=[X2]+states2
#    Dots=[np.dot(x1,x2.T) for x1,x2 in zip(X1s,X2s)]
#    l_grams=[np.multiply.reduce(Dots[lyr:]) for lyr in range(len(Dots))]
#
#    gram=sum(l_grams)
#
#
#
#    ####phi(x,w)####
#    Xs=[X]+get_states(X)
#    Tendims=[xdim]+arch #2,4,6,8

    ##forget what i was doing here
    #all_paths=[]
    #all_tensor=[Xs[-1]]
    #for xs in Xs[-2::-1]:#start2nd from end, count backwards
    #    ten=do_outer(xs,all_tensor[0])
    #    all_tensor=[ten]+all_tensor


    #stophere


#    #Try centering #seemed to help
#
#    #In kernel space#
#    #k=np.sum(gram)/(len(gram)**2.)
#    #row_ave=np.mean(gram,axis=1,keepdims=True)
#    #col_ave=np.mean(gram,axis=0,keepdims=True)
#    #gram-=row_ave+col_ave-k  #helps
#
#    #treat as linear features
#    #gram-=gram.mean(axis=0,keepdims=True)#ave over samples#helps
#    #gram/=gram.std(axis=0,keepdims=True)#Seems to hurt
#
#    #gram/=gram.std(axis=0,keepdims=True)
#    #gram/=gram.std(axis=1,keepdims=True)
#
#    #plt.figure()
#    #plt.imshow(gram)
#    #plt.colorbar()
#    #plt.title('gram')
#
#    #X1s=[X1]+[B*S for B,S in zip(Bweights[:-1],states1)]#d+1
#    #X2s=[X2]+[B*S for B,S in zip(Bweights[:-1],states2)]
#    #layers_kernels=[]#paths starting at a layer
#    #for lyr, (x1,x2) in enumerate(zip(X1s,X2s)):
#    #    #lyr 0,..,d
#    #    sta1=[x1]+states1[lyr:]
#    #    sta2=[x2]+states2[lyr:]
#    #    l_kers=[np.dot(s1,s2.T) for s1,s2 in zip(sta1,sta2)]
#    #    layers_kernels.append(l_kers)
#    #l_grams=[np.multiply.reduce(kers) for kers in layers_kernels]
#    #gram=sum(l_grams)



    ##Get phi(X,w)##
    #states=get_states(X)


    #Kers=[np.dot(s1,s2.transpose()) for s1,s2 in zip(states1,states2)]
    #gram=np.multiply.reduce(Kers)


    #stophere
    net_X=vec_get_neuron_values(X,weights)[-1]
    net_acc= np.mean(np.sign(net_X)==npY)


    #########Fit Model#########
    models=[]

    def net_model():#dummy
        pass
    net_model.name='network'
    net_model.acc=net_acc
    models.append(net_model)


    #ajil said 1e-5
    #1e-5, maxiter 500k, shrinking=False worked for D3A2
    clf_precom=svm.SVC(kernel='precomputed',
               #C=1e-2,
               C=1e-5,
               #C=1e5,
               tol=1e-5,
               shrinking=False,
               #cache_size=300,#MB
               max_iter=500000,
              )
    clf_precom.name='svc-precomputed'
    models.append(clf_precom)


    clf_linear=svm.SVC(kernel='linear',
               #C=1e-5,
               tol=1e-5,
               max_iter=500000,
               #class_weight='balanced',
               shrinking=False,
                      )
    clf_linear.name='LinearKernel'
    models.append(clf_linear)


    lib_lin=LinearSVC(loss='hinge',
                  C=1e-5,
                  tol=1e-8,
#                  dual=True,
#                  max_iter=20000,
                  #penalty='l1',
                 )
    lib_lin.name='LinearSVC'
    models.append(lib_lin)

    logistic=LogisticRegression(
            #C=1e5,#works
            C=1e2,
            solver='lbfgs',
            max_iter=10000,#1k default
    )
    logistic.name='logistic'
    models.append(logistic)

    sgd=SGDClassifier(
        loss='log',
        penalty='l2',
        #max_iter=10000,
        #tol=1e-5,
        alpha=0.1,
        warm_start=True,
    )
    sgd.name='sgd'
    models.append(sgd)

    #models=[clf_precom,clf_linear,lib_lin,logistic,sgd]

    print 'fitting models:'
    for clf in models[1:]:#skip NN
        clf.fit(gram,Y)
        print '..fit model',clf.name
        clf.acc=np.round(clf.score(gram,Y),4)
        clf.correct=(clf.predict(gram)==Y)
        clf.wh_wrong=np.where(~ clf.correct)[0]



    #stophere

    #clf_precom.fit(gram,Y)
    #clf_precom.acc=clf_precom.score(gram,Y)
    #clf_linear.fit(gram,Y)
    #clf_linear.acc=clf_linear.score(gram,Y)
    #lib_lin.fit(gram,Y)
    #lib_lin.acc= lib_lin.score(gram,Y)
    #clf_linear.correct=(clf_linear.predict(gram)==Y)
    #clf_linear.wh_wrong=np.where(~ clf_linear.correct)[0]


    ###How Well Did it Do?###
    print 'NN score',net_acc
    print 'SVC score',clf_precom.acc#, 'net-uncer', net_X[clf_precom.wh_wrong].std()
    print 'SVC kernel=linear score',clf_linear.acc#, 'net-uncer',net_X[clf_linear.wh_wrong].std()
    print 'LinearSVC score',lib_lin.acc#, 'net-uncer',net_X[lib_lin.wh_wrong].std()
    print 'Logistic score',logistic.acc#, 'net-uncer',net_X[logistic.wh_wrong].std()
    print 'SGD score',sgd.acc#, 'net-uncer',net_X[sgd.wh_wrong].std()


    print 'calculating gram grid..'
    fl_gX=gX.reshape([-1,xdim])
    gram_flgrid=get_gram(fl_gX,X)
    print '..done'

    net_fn=vec_get_neuron_values(gX,weights)[-1].reshape(gX.shape[:xdim])
    net_pred=np.sign(net_fn)
    net_fn/=np.max(np.abs(net_fn))

    net_model.grid_fn=net_fn
    net_model.grid_pred=net_pred



    #net_file=os.path.join(pactensor_dir,'net_gridX'

    for clf in models[1:]:
        clf_fn=np.reshape(clf.decision_function(gram_flgrid),gX.shape[:xdim])
        clf_pred=np.sign(clf_fn)
        clf_fn/=np.max(np.abs(clf_fn))
        clf.grid_pred=clf_pred
        clf.grid_fn=clf_fn


    for clf in models:
#        fig,axes=plt.subplots(1,2)
#        ax,ax2=axes
#        paint_data(ax,X,Y)
#        paint_data(ax2,X,Y)
#        paint_binary_contours(ax,gX,clf.grid_fn,cbar=True)
#        paint_binary_contours(ax2,gX,clf.grid_pred,cbar=True)

#        ax.set_title(clf.name)
#        plt.savefig(pactensor_dir +'/contourf_'+'res_'+str(res)+'_'+clf.name+'.pdf')

        fig,ax = plt.subplots()
        paint_data(ax,X,Y)
        paint_binary_contours(ax,gX,clf.grid_pred,thresh=0.)
        plt.savefig(pactensor_dir+'/contourf_'+'res_'+str(res)+'_'+clf.name+'.pdf')

        if clf in [net_model, clf_precom]:#best
            plt.savefig('./figures/Fig1/'+id_str+'_dataclassif_'+descrip+'res_'+str(res)+'_'+clf.name+'.pdf')
            plt.savefig('./figures/Fig1/'+id_str+'_dataclassif_'+descrip+'res_'+str(res)+'_'+clf.name+'.png')


    print '..done saving'
    #plt.show()

    stophere

    #print 'Model Fit'

    #correct=clf_precom.predict(gram)==Y
    #if not correct.all():
    #    print 'Warn not all learned'
    #    wh_bad=np.where(correct==False)[0]
    #alpha=clf_precom.dual_coef_
    #svm_dec=clf_precom.decision_function(gram)
    #recon=np.sum(gram[:,clf_precom.support_]*alpha,axis=-1)+clf_precom.intercept_#Good2know
    #Good. dec==recon True

    #clf_precom.predict(n_samples_test x n_samples_train)(when kernel='precomputed')

#    fl_gX=gX.reshape([-1,xdim])
#    gram_flgrid=get_gram(fl_gX,X)
#    #svm_pred=np.reshape(clf_precom.predict(gram_flgrid),gX.shape[:xdim])
#    svm_fn=np.reshape(clf_precom.decision_function(gram_flgrid),gX.shape[:xdim])
#    net_fn=vec_get_neuron_values(gX,weights)[-1].reshape(gX.shape[:xdim])
#    svm_pred=np.sign(svm_fn)
#    net_pred=np.sign(net_fn)
#    svm_fn/=np.max(np.abs(svm_fn))
#    net_fn/=np.max(np.abs(net_fn))
#
#    #net_dec=vec_get_neuron_values(X,weights)[-1]
#    #net_pred=np.sign(net_dec)
#
#
#    plt.figure()
#    plt.imshow(gram)
#    plt.colorbar()
#    plt.title('gram')
#
#
#    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(6,6))
#
#    cmap = plt.cm.get_cmap("Purples")
#    my_cmap = cmap(np.arange(cmap.N))
#    my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
#    plt.tight_layout()
#    plt.xticks([]) # labels 
#    plt.yticks([])
#    for ax in [ax1,ax2]:
#        ax.xaxis.set_ticks_position('none') # tick markers
#        ax.yaxis.set_ticks_position('none')
#        ax.scatter(Xpos[:,0],Xpos[:,1],marker='+',s=600,c='b',linewidth='3')
#        ax.scatter(Xneg[:,0],Xneg[:,1],marker='_',s=600,c='r',linewidth='3')

    #ctf1=ax1.contourf(gX0,gX1,(net_dec>0).astype('float'),levels=[-0.01,0,1.01],colors=['w',cmap(190)])
    #ctf2=ax2.contourf(gX0,gX1,(svm_dec>0).astype('float'),levels=[-0.01,0,1.01],colors=['w',cmap(190)])
    #ctf1=ax1.contourf(gX0,gX1,net_dec,levels=[-0.01,0,1.01],colors=['w',cmap(190)])
    #ctf2=ax2.contourf(gX0,gX1,svm_dec,levels=[-0.01,0,1.01],colors=['w',cmap(190)])

    #im1=ax1.imshow(net_fn.T,vmin=-1,vmax=+1)
    #im2=ax2.imshow(svm_fn.T,vmin=-1,vmax=+1)


#    im1=ax1.imshow(net_fn[::-1],vmin=-1,vmax=+1)
#    im2=ax2.imshow(svm_fn[::-1],vmin=-1,vmax=+1)
#
#    fig.subplots_adjust(right=0.8)
#    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#    fig.colorbar(im1, cax=cbar_ax)
#
#
#    plt.show()




















    #Sig,Centers,Cnts,Idx0=get_net_states(npX,weights,return_bdry_idx=True)

    #dic={'L'+str(1+i):PLayers[i] for i in range(len(PLayers))}
    #AD=ArrayDict({'L'+str(1+i):(PLayers[i]>=0).astype('int') for i in range(len(PLayers))})

#    if pred_is_a_state:
#        states=[(p>=0).astype('int') for p in PLayers     ]
#    else:
#        states=[(p>=0).astype('int') for p in PLayers[:-1]]

    #paths=np.concatenate(states,axis=-1)


#    fl_paths=paths.reshape([-1,paths.shape[-1]])
#    fl_X=npX.reshape([-1,npX.shape[-1]])#xdim


#    fl_paths_hidden=fl_paths[:,:-1]
#    fl_paths_pred  =fl_paths[:,-1:]


