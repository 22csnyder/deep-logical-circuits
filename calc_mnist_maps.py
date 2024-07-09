import pandas as pd
import numpy as np
from config import get_config
from utils import prepare_dirs_and_logger,save_config
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

import sympy
from sympy import symbols
from sympy.logic.boolalg import And,Or
from sympy import Max,Min,srepr
from sympy.utilities.lambdify import lambdify

'''
The bool expansion part of this file in __main__ is generation v3
the new point is to decouple mu and tau.

This is a file to calculate the different linear maps that arise
through the different neuron activations

It works with numpy arrays
'''

from vis_utils import (split_posneg , get_path, get_np_network,
                        get_neuron_values, splitL, load_weights,
                        resample_grid,vec_get_neuron_values,
                        get_del_weights  )
from tboard import file2number
from nonlinearities import relu#np version


def get_net_states(X,weights,return_bdry_idx=False):
    '''
    Returns the neuron states for all neurons except the final output,
    unless pred_is_a_state is True, in which case a binary value is returned
    for final layer

    X is usually a grid in the input space
    weights is a list of pairs of indexed weights,biases

    '''
    PLayers=vec_get_neuron_values(X,weights)#Layers*(wpad,xpad,trn_iter,layersize)
    dic={'L'+str(1+i):PLayers[i] for i in range(len(PLayers))}
    AD=ArrayDict({'L'+str(1+i):(PLayers[i]>=0).astype('int') for i in range(len(PLayers))})

#    if pred_is_a_state:
#        states=[(p>=0).astype('int') for p in PLayers     ]
#    else:
#        states=[(p>=0).astype('int') for p in PLayers[:-1]]

    states=[(p>=0).astype('int') for p in PLayers     ]
    paths=np.concatenate(states,axis=-1)


    #I guess requires weights to not have any indexing
    #e.g. weights evolving over time
    #Could fix later or just loop over time
    assert(X.shape[:-1]==paths.shape[:-1])


    fl_paths=paths.reshape([-1,paths.shape[-1]])
    fl_X=X.reshape([-1,X.shape[-1]])#xdim


    fl_paths_hidden=fl_paths[:,:-1]
    fl_paths_pred  =fl_paths[:,-1:]

    Sig,Inv,Cnts=np.unique(fl_paths_hidden,axis=0,
                           #return_index=True,
                           return_inverse=True,
                           return_counts=True)

    IdxPlus =Inv[np.where(fl_paths_pred==1)[0]]
    IdxMinus=Inv[np.where(fl_paths_pred==0)[0]]
    Idx0=np.intersect1d(IdxPlus,IdxMinus)


    Centers=np.zeros([len(Sig),X.shape[-1]])
    #for k in range(len(Sig)):#region
    for k in np.unique(Inv): #equivalent
        Xk=fl_X[np.where(Inv==k)[0]]#the corresp x's
        Centers[k]=np.mean(Xk,axis=0)




    #Have Sig,Centers,Cnts
    if return_bdry_idx:
#        Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]
        return Sig,Centers,Cnts,Idx0
    else:
        return Sig,Centers,Cnts


def rescale_weights(weights,activations):
    '''
    Takes wts (weights) and neuron activations
    (thus the last dim must agree
    and zeros the last dims determined by multiplying
    the two arguments after reshaping

    Converts network activations and weights into a list of affine maps
    that compose to give you the network function
    corresponding to the activation pattern


    sigl only corresponds to network states. don't pass in array([1]) for final
    network value
    '''
    wts=weights
    sigl=activations

    assert(len(wts)==len(sigl)+1)

    w0,b0=wts[0]
    wpad=len(w0.shape)-2#last two dim for matrix
    sh_widx=w0.shape[:wpad]
    sig_pad=len(sigl[0].shape)-1

    #Reshaping
    #sig_shapes=[(1,)*wpad+s.shape for s in sigl]
    #rs_X=X.reshape(Xshape)

    def wrs(W):#also works on biases
        new_shape=sh_widx + (1,)*sig_pad + W.shape[wpad:]
        return W.reshape(new_shape)
    rs_weights=[[wrs(W),wrs(b)] for W,b in wts]

    #pretty sure not needed
    #def srs(sig):
    #    new_shape=(1,)*wpad+sig.shape
    #    return sig.reshape(new_shape)
    ##rs_sigl=[srs(sig) for sig in sigl]

    new_weights=[]
    for l in range(len(sigl)):
        w,b=rs_weights[l]#skips last
        sig=sigl[l]
        s1=sig.reshape((1,)*wpad+sig.shape[:-1]+(1,)+sig.shape[-1:])
        s2=sig.reshape((1,)*(wpad+0)+sig.shape)
        new_weights.append([w*s1,b*s2])
    new_weights.append(wts[-1])#Last one unchanged

    return new_weights


def compose_affine(list_affine):
    #func section
    '''
    Takes list of network weights (with abitrary leading indicies)
    and composes them as if they were a composition of affine functions
    without nonlinearities between
    '''
    wts=list_affine
    w0=wts[0][0]
    #xdim=w0.shape[-2]

    #A,b
    #init
    #A=np.eye(xdim)
    #b=np.zeros

    #x*Linear+Bias
    Linear,Bias=wts[0]

    #W,b=wts[1]#test
    for W,b in wts[1:]:
        rsLinear=np.expand_dims(Linear,axis=-1)
        rsBias=np.expand_dims(Bias,axis=-1)
        rsW=np.expand_dims(W,axis=-3)
        Linear=np.sum(rsLinear*rsW, axis=-2)
        Bias=np.sum(rsBias*W,axis=-2)+b
        #Bias=np.sum(rsBias*rsW,axis=-2)+b

    #print 'Affine map is \n'+\
    #    'Linear',Linear,'\n'+\
    #    'Bias',Bias
    return (Linear,Bias)


def AndOr2MinMax(Op):
    if Op is Or:
        return Max#sympy.Max
    elif Op is And:
        return Min#sympy.Min
    else:
        raise ValueError('Either simpy.And or simpy.Or must be passed')


def swap_bool(Op):
    if Op is Or:
        return And
    elif Op is And:
        return Or
    else:
        raise ValueError('Either simpy.And or simpy.Or must be passed')

def calc_Sig(log_dir,res=5000):#takes long time. do while sleep.
    print 'using log_dir:',log_dir
    record_dir=os.path.join(log_dir,'records')
    all_weights=load_weights(log_dir)
    all_step=np.load(get_path('step','wwatch',log_dir))
    dt=10
    weights=[[w[::dt],b[::dt]] for w,b in all_weights]
    arch=[b.shape[-1] for w,b in weights[:-1]]#net architecture
    del_weights=get_del_weights(weights)
    gridX=np.load(get_path('gridX','hmwatch',log_dir))
    HighResX=resample_grid(gridX,5000)
    #Does everything at last time point
    trn_iter=-1
    time_weights=[[w[trn_iter],b[trn_iter]] for w,b in del_weights]#d*2*wtshape

    sve='/res'+str(res)
    pth_Sig =record_dir+sve+'_Sig.npy'
    pth_Sig0=record_dir+sve+'_Sig0.npy'
    #pth_Sig =record_dir+'/res5000_Sig.npy'
    #pth_Sig0=record_dir+'/res5000_Sig0.npy'
    if os.path.exists(pth_Sig):
        print 'Already has Sig data'
        Sig=np.load(pth_Sig)
        Sig0=np.load(pth_Sig0)
        print 'finished with Sig data'
    else:
        print 'Getting high res Sig data'
        Sig,Centers,Cnts,Idx0=get_net_states(HighResX,time_weights,return_bdry_idx=True)
        Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]
        #np.save(record_dir+'/res5000_Sig.npy',Sig)
        #np.save(record_dir+'/res5000_Sig0.npy',Sig0)
        np.save(pth_Sig,Sig)
        np.save(pth_Sig0,Sig0)
        print 'finished with Sig data'

if __name__=='__main__':
    plt.close('all')

    #log_dir='./logs/Model_0513_092517_MnistPCA3m20k_A2Iter5k'

    #log_dir='./logs/Model_0513_120107_MnistPCA3A2_labs01.67'
        #worked but logical ops were kind of trivial

    #log_dir='./logs/debugValidation_0513_160116_MnistPCA3A2_labsAll'
    #like.88val
    # (a1 | a28 | a3 | a43 | a8) & (a10 | a11 | a16 | a23 | a32) & (a14 | a19 |
    # a22 | a33 | a38) & (a2 | a30 | a31 | a34 | a7) & (a12 | a13 | a18 | a24 |
    # a39 | a40 | a5 | a9)& (a0 | a15 | a17 | a20 | a21 | a25 | a26 | a27 | a29
    # | a35 | a36 | a37 | a4 | a41 | a42 | a6)

    #log_dir='./logs/Model_0513_161151_Mnist20kA2PCA4_LabsAll'
        #93%val acc
        #128 atoms. booltree saved in records/*.png
    #log_dir='./logs/Model_0513_181123_Mnist20kA2PCA4_Labs012.567'
        #acc 99/95 |Sig0|=169

    #log_dir='./logs/Model_0513_162218_Mnist55kA2PCA4_LabsAll'#bigboy
    #    #93%valacc. |Sig0|=75
    #    #decisiontree wasnt that interesting somehow

    #log_dir='./logs/Model_0514_010526_Mnist20kA3PCA4_LabsAll'
    #   |Sig0|=139

    #log_dir='./logs/Model_0514_011608_Mnist5kA3PCA4_LabsAll'

    #log_dir='./logs/Model_0514_012756_Mnist5kA3PCA4_LabsAll'
    #97/89

    #log_dir='./logs/Model_0514_013017_Mnist5kA3PCA4_LabsAll_50kiter'
    #95/84#Not done yet

    #log_dir='./logs/Model_0514_014452_Mnist5kA3PCA4_LabsAll_50kiter'
    #|Sig0|=71   .92/.83


    #Same experimental settings for the next two!
    #log_dir='./logs/Model_0514_151155_Mnist1kA3PCA4_LabsAll_50kiter'
    #1./.781 |Sig0|=55 |Sig|=2432

    #log_dir='./logs/Model_0514_162548_Mnist1kA3PCA4_LabsAll_50kiter'
    #0.91/0.91 |Sig0|=3 |Sig|=191


    #log_dir='./logs/Model_0514_185024_Mnist20kA3PCA6_LabsAll_50kiter'#.93/.91#MEM ERR
    #log_dir='./logs/Model_0514_191120_Mnist20kA3PCA6_LabsAll_50kiter'#.91/.87

    #Try A1 instead for mem reasons
    #log_dir='./logs/Model_0514_201828_Mnist20kA1PCA6_LabsAll_50kiter'#.98/.91
    #|Sig| 44/182

    #log_dir='./logs/Model_0514_210136_Mnist50kA1PCA6_LabsAll_50kiter'#.96/.93
    #65/386

    #log_dir='./logs/Model_0514_220539_Mnist50kA1PCA6_LabsAll_50kiter';m=50000
    #.985/.94    14/147

    log_dir='./logs/Model_0514_230032_Mnist50kA1PCA6_LabsAll_50kiter';m=50000
    #.980/.94


    #Setup
    print 'using log_dir:',log_dir
    record_dir=os.path.join(log_dir,'records')
    id_str=str(file2number(log_dir))
    all_weights=load_weights(log_dir)
    all_step=np.load(get_path('step','wwatch',log_dir))
    arch=[b.shape[-1] for w,b in all_weights[:-1]]#net architecture
    print 'arch ',arch


    dt=10
    weights=[[w[::dt],b[::dt]] for w,b in all_weights]
    del_weights=get_del_weights(weights)
    #Does everything at last time point
    all_time_weights=[[w[-1],b[-1]] for w,b in del_weights]#d*2*wtshape
    step=all_step[::dt]
    print 'number of affine maps ',len(all_weights)

    ###This part pca specific
    time_weights=all_time_weights[1:]
    projW,projB=all_time_weights[0]
    weights=all_weights[1:]
    pca_dim,arch=arch[0],arch[1:]

    #split off first layer from all_weights
    #enforce first layer linear

    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)


    #res=50.  #standard for pca4
    res=10   #pca6
    print 'Using resolution ',res

    HL1=np.matmul(npX,projW)+projB
    upper= np.ceil(HL1.max(axis=0))
    lower=np.floor(HL1.min(axis=0))
    sz=upper-lower

    delta=np.min(sz)/float(res)
    #delta=sz/float(res)

    xi=[np.arange(lower[i],upper[i],delta)  for i in range(len(sz))]
    print 'forming GridX..'
    ResX=np.stack(np.meshgrid(*xi),axis=-1)#LxLx2

    print '..formed GridX'
    #stophere

    sve='/res'+str(res)
    pth_Sig =record_dir+sve+'_Sig.npy'
    pth_Sig0=record_dir+sve+'_Sig0.npy'
    #pth_Sig =record_dir+'/res5000_Sig.npy'
    #pth_Sig0=record_dir+'/res5000_Sig0.npy'
    if os.path.exists(pth_Sig):
        Sig=np.load(pth_Sig)
        Sig0=np.load(pth_Sig0)
        print 'finished with Sig data'
    else:
        print 'Getting high res Sig data'
        Sig,Centers,Cnts,Idx0=get_net_states(ResX,time_weights,return_bdry_idx=True)
        Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]
        np.save(pth_Sig,Sig)
        np.save(pth_Sig0,Sig0)
        #np.save(record_dir+'/res5000_Sig.npy',Sig)
        #np.save(record_dir+'/res5000_Sig0.npy',Sig0)
        print 'finished with Sig data'


    #future, could aggregate across time before this part
    sigl=np.split(Sig,#split up into layers again
                  indices_or_sections=np.cumsum(arch)[:-1],
                  axis=-1)
    sigl0=np.split(Sig0,#split up into layers again
                  indices_or_sections=np.cumsum(arch)[:-1],
                  axis=-1)

    print '|Sig|=',len(Sig)
    print '|Sig0|=',len(Sig0)
    #stophere


###Build Boolean###

    #using sigl for DEBUG
    #n_sig=len(sigl[0])
    #depth=len(sigl)

    ###arguments
    #sigl (sigl or sigl0)
    #set of weights
    ###returns
    #terminal_leafs, Bool_Tree


    #config
    #0,1,1 is the config I settled on. The rest of it was messing around.
    #WhichSig='All'
    WhichSig=0

    OrAndPivot=1
        #1 only start last layer with init_op, alternate
        #2 always start each layer with init_op
    LeafSigMethod=1
        #1 sets new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
        #2 LeafSig includes all list_unq_sig[hidden_layer]

    #Either is a fine choice
    init_op=Or
    #init_op=And

    config_str='WS'+str(WhichSig)+'_OAP'+str(OrAndPivot)+'_LSM'+str(LeafSigMethod)
    config_str+='_io'+str(init_op)


    if WhichSig=='All':
        list_unq_sig=sigl
    elif WhichSig==0:
        list_unq_sig=sigl0
    else:
        raise ValueError('WhichSig is ',WhichSig)

    #init
    n_sig=len(list_unq_sig[0])
    depth=len(list_unq_sig)
    Bool_Tree=symbols('0')
    MinMax_Tree=symbols('0')
    leaf=Bool_Tree.atoms().pop()#atoms() returns a set
    leaf_stack={leaf:{
                      'last_operator'  :init_op,#default start
                      #'idx_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'id_mu_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'id_tu_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'hidden_layer'   :depth,#0,...,depth-1(first is dummy)
                      'affine'         :time_weights[-1],
                     }
               }
    debug_stack=leaf_stack.copy()#shallow
    terminal_leafs={}#so ctrl-f finds it


    print 'Entering While Loop'
    LOOP=-1
    while leaf_stack:#until empty
        LOOP+=1#DEBUG
        #start loop
        a_leaf=leaf_stack.keys()[0]
        leaf_info=leaf_stack.pop(a_leaf)#while nonempty
        leaf_parent_operator=leaf_info['last_operator']#sympy Or/And
        #idx_remain_sig=leaf_info['idx_remain_sig']
        id_mu_remain_sig=leaf_info['id_mu_remain_sig']
        id_tu_remain_sig=leaf_info['id_tu_remain_sig']
        ##Increment## hidden_layer -= 1
        hidden_layer=leaf_info['hidden_layer']-1   #hidden layers iterate 0,..,depth-1
        gamma,beta=leaf_info['affine']#current linear map
        #bool_remain_sig=leaf_info['bool_remain_sig']
        #print 'leaf:',a_leaf
        #print 'G',gamma,'b',beta,'\n'
        #print 'DEBUG','IDX_REMAIN_SIG',idx_remain_sig
        #leaf_sig=list_unq_sig[hidden_layer][idx_remain_sig]#becomes mu_,tu_leaf_sig
        mu_leaf_sig=list_unq_sig[hidden_layer][id_mu_remain_sig]
        tu_leaf_sig=list_unq_sig[hidden_layer][id_tu_remain_sig]
            #leaf_sig=list_unq_sig[hidden_layer]#simply use all
        #leaf_weights=time_weights[hidden_layer:]#includes weights feeding into layer
        leaf_weights=[time_weights[hidden_layer],[gamma,beta]]#includes weights feeding into layer

        gamma_pos=(gamma>0).astype('int').ravel()#strict ineq justified
        gamma_neg=(gamma< 0).astype('int').ravel()
        pos_sig,pos_inv=np.unique(gamma_pos*mu_leaf_sig,axis=0,return_inverse=True)
        neg_sig,neg_inv=np.unique(gamma_neg*tu_leaf_sig,axis=0,return_inverse=True)
        mu_Sig,tu_Sig=pos_sig,neg_sig#db
        #pos_sig,pos_inv=np.unique(gamma_pos*leaf_sig,axis=0,return_inverse=True)
        #neg_sig,neg_inv=np.unique(gamma_neg*leaf_sig,axis=0,return_inverse=True)
        #unq_sig,unq_inv=np.unique(leaf_sig,axis=0,return_inverse=True)

        #split into pos,neg pieces
        #posrelu_gamma+negrelu_gamma=gamma
        posrelu_gamma,posrelu_beta=relu(gamma).ravel(),relu(beta)
        negrelu_gamma,negrelu_beta=-relu(-gamma).ravel(),-relu(-beta)
        #mu_gamma#mu_beta
        mu_beta,tu_beta=posrelu_beta,negrelu_beta
        mu_gamma,mu_inv=np.unique(posrelu_gamma*mu_leaf_sig,axis=0,return_inverse=True)
        tu_gamma,tu_inv=np.unique(negrelu_gamma*tu_leaf_sig,axis=0,return_inverse=True)
        ##Careful because entries in mu_leaf_weights not all same shape len
        ##but compose_affine seems to broadcast correctly anyway
        mu_leaf_weights=[time_weights[hidden_layer],[mu_gamma[...,None],mu_beta]]
        tu_leaf_weights=[time_weights[hidden_layer],[tu_gamma[...,None],tu_beta]]
        new_mu_Gamma,new_mu_Beta=compose_affine(mu_leaf_weights)#n_unqxnlx1,n_unqx1
        new_tu_Gamma,new_tu_Beta=compose_affine(tu_leaf_weights)


        ##How to decide op order##
        ###OLD v1###
        if OrAndPivot==1:#old#v1
            c1_op=leaf_parent_operator
        ####always Or###
        elif OrAndPivot==2:#new
            c1_op=init_op
        c2_op=swap_bool(c1_op)#always
        c1_opt=AndOr2MinMax(c1_op)
        c2_opt=AndOr2MinMax(c2_op)

        if c1_op is Or:
            #c1_sig,c1_inv=pos_sig,pos_inv#pre v3
            c1_Gamma,c1_Beta,c1_inv=new_mu_Gamma,new_mu_Beta,mu_inv
            c2_Gamma,c2_Beta,c2_inv=new_tu_Gamma,new_tu_Beta,tu_inv
            c1_Sig= pos_sig#keep track of which mu/tau was used
            c2_Sig=-neg_sig
            c1_is_mu_=True
        elif c1_op is And:
            c1_Gamma,c1_Beta,c1_inv=new_tu_Gamma,new_tu_Beta,tu_inv
            c2_Gamma,c2_Beta,c2_inv=new_mu_Gamma,new_mu_Beta,mu_inv
            c1_Sig=-neg_sig
            c2_Sig= pos_sig
            c1_is_mu_=False
        else:
            raise ValueError('c1_op is ',c1_op)

        #c2_sig,c2_inv=unq_sig,unq_inv

#        if c1_op is Or:
#            #c1_sig,c1_inv=pos_sig,pos_inv#pre v3
#            c1_sig,c1_inv=mu_sig,mu_inv
#            c2_sig,c2_inv=tu_sig,tu_inv#new decouple in v3
#        else:
#            #c1_sig,c1_inv=neg_sig,neg_inv
#            c1_sig,c1_inv=tu_sig,tu_inv
#            c2_sig,c2_inv=mu_sig,mu_inv#new decouple in v3
#        #c2_sig,c2_inv=unq_sig,unq_inv

        #new_leaf_weights=rescale_weights(leaf_weights,[c2_sig])#(d+1*2*wtshape, d*n_sigxnl)
        #new_Gamma,new_Beta=compose_affine(new_leaf_weights)# num_remain x layer_l shape x layer l-1 shape

        sig2name=lambda a: ''.join([str(b) for b in a])
        c1_sym_list=[]
        c1_opt_list=[]
        for i1,c1_gamma in enumerate(c1_Gamma):
            c1_beta=c1_Beta[i1]
            c1_sig=c1_Sig[i1]
            #c1_sym=sig2name(c1_sig)

            ##Decoupled last operator
            if len(c2_Gamma)>1:
                last_operator=c2_op
            else:
                last_operator=c1_op
            ##Only relevant when pairing down c2 by c1##
            #if len(i2_corresp_i1)>1:
            #    last_operator=c2_op
            #else:
            #    last_operator=c1_op

            c2_sym_list=[]
            for i2,c2_gamma in enumerate(c2_Gamma):
                c2_beta=c2_Beta[i2]
                c2_sig=c2_Sig[i2]
                c2_sym=sig2name(c2_sig)
                assert(np.dot(np.abs(c1_sig),np.abs(c2_sig))==0.)

                new_ext=sig2name(c1_sig+c2_sig)
                new_name=new_ext+'.'+str(a_leaf)
                new_node=symbols(new_name)
                c2_sym_list.append(new_node)

                ##New Affine##
                new_gamma=c1_gamma+c2_gamma
                new_beta=c1_beta+c2_beta

                ####new idx####
                if LeafSigMethod==1:#v1
                    #new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
                    if c1_is_mu_:#mu is first coord
                        new_id_mu_remain_sig=id_mu_remain_sig[c1_inv==i1]
                        new_id_tu_remain_sig=id_tu_remain_sig[c2_inv==i2]
                        mu_sig,tu_sig=c1_sig,c2_sig#for debug
                    else:
                        new_id_mu_remain_sig=id_mu_remain_sig[c2_inv==i2]
                        new_id_tu_remain_sig=id_tu_remain_sig[c1_inv==i1]
                        mu_sig,tu_sig=c2_sig,c1_sig#for debug
                elif LeafSigMethod==2:#all idx always
                    #new_idx_remain_sig=idx_remain_sig
                    new_id_mu_remain_sig=id_mu_remain_sig
                    new_id_tu_remain_sig=id_tu_remain_sig
                new_leaf_info={
                                'last_operator'  :last_operator,
                                #'idx_remain_sig' :new_idx_remain_sig,
                                'id_mu_remain_sig' :new_id_mu_remain_sig,
                                'id_tu_remain_sig' :new_id_tu_remain_sig,
                                'affine' :[new_gamma,new_beta],
                                #'name'           :new_node,
                              }

                if hidden_layer>0:
                    new_leaf_info['hidden_layer']=hidden_layer
                    leaf_stack[new_node]=new_leaf_info
                else:
                    #assert(len(new_idx_remain_sig)==1)#corresp to 1 guy
                    terminal_leafs[new_node]=new_leaf_info

                #debug at top of network
                if hidden_layer>=1:
                    debug_stack[new_node]=new_leaf_info

            c1_sym_list.append( c2_op(*c2_sym_list ) )
            c1_opt_list.append( c2_opt(*c2_sym_list) )
        new_branch    =c1_op(*c1_sym_list)
        new_branch_opt=c1_opt(*c1_opt_list)
        Bool_Tree  =Bool_Tree.subs(a_leaf,new_branch)
        MinMax_Tree=MinMax_Tree.subs(a_leaf,new_branch_opt)

    new_weights=rescale_weights(time_weights,sigl)#(d+1*2*wtshape, d*n_sigxnl)
    Linear,Bias=compose_affine(new_weights)
    new_weights0=rescale_weights(time_weights,sigl0)#(d+1*2*wtshape, d*n_sigxnl)
    Linear0,Bias0=compose_affine(new_weights0)
    ####PCA specifici#### (if config.pca)
    Linear ,Bias =compose_affine([[projW,projB],[Linear ,Bias ]])
    Linear0,Bias0=compose_affine([[projW,projB],[Linear0,Bias0]])
    for leaf_info in terminal_leafs.values():
        leaf_info['affine']=compose_affine([[projW,projB],leaf_info['affine']])

    Leafs,List_Leaf=zip(*terminal_leafs.items())#Defines ordering from here on
    #List_Leaf=terminal_leafs.values()
    #List_Leaf.sort(key=lambda info:info['id_mu_remain_sig'][0])

    ##Make readable##
    alphabet=symbols('a0:%d'%len(Leafs))
    subdict={atom:let for atom,let in zip(Leafs,alphabet)}
    abstract_Bool_Tree=Bool_Tree.subs(subdict)
    abstract_MinMax_Tree=MinMax_Tree.subs(subdict)
    abt=abstract_Bool_Tree

    print 'Bool Tree','\n',abstract_Bool_Tree
    print 'MinMax Tree','\n',abstract_MinMax_Tree

    GammaTree=np.stack([L['affine'][0] for L in List_Leaf],axis=0)
    BetaTree =np.stack([L['affine'][1] for L in List_Leaf],axis=0)
    GammaTree=np.squeeze(GammaTree)
    #BetaTree=np.squeeze(BetaTree)
    Linear=np.squeeze(Linear)
    #Bias=np.squeeze(Bias)
    Linear0=np.squeeze(Linear0)

    np.save(record_dir+'/GammaTree',GammaTree)
    np.save(record_dir+'/BetaTree',BetaTree)



    ###Save BoolTree###

    #import dill
    import pickle
    logic_dir=os.path.join(log_dir,'logic')
    if not os.path.exists(logic_dir):
        os.makedirs(logic_dir)

    with open( logic_dir+'/sympy_abstract_Bool_Tree.pkl','wb') as f:
        pickle.dump(abstract_Bool_Tree, f)
    with open( logic_dir+'/sympy_Bool_Tree.pkl','wb') as f:
        pickle.dump(Bool_Tree, f)
    with open( logic_dir+'/terminal_leafs.pkl','wb') as f:
        pickle.dump(terminal_leafs,f)
    with open( logic_dir+'/leaf_substitute_symbols.pkl','wb') as f:
        pickle.dump(subdict,f)

    #dill.settings['recurse']=True
    #with open( logic_dir+'/sympy_abstract_Bool_Tree.pkl','wb') as f:
    #    dill.dump(abstract_Bool_Tree, f)
    #with open( logic_dir+'/sympy_Bool_Tree.pkl','wb') as f:
    #    dill.dump(Bool_Tree, f)
    #with open( logic_dir+'/terminal_leafs.pkl','wb') as f:
    #    dill.dump(terminal_leafs,f)
    #with open( logic_dir+'/leaf_substitute_symbols.pkl','wb') as f:
    #    dill.dump(subdict,f)

    stophere

    ##PlotConfig
    ####Not really working unless pcadim=2
    PlotHM=0
    res=800

    BoolSub=0
    BoolPlot=1
    BoolMatchPlot=1

    MinMaxSub=0
    MinMaxPlot=1
    MinMaxMatchPlot=1

    BP=BoolSub*(BoolPlot+BoolMatchPlot)
    MP=MinMaxSub*(MinMaxPlot+MinMaxMatchPlot)
    ExistPlots=(BP+MP>=1)

    #gX=gridX#much faster
    #gX=GridX
    #gX=resample_grid(gridX,res)#slower
    gX=ResX


    ####Plot contour####
    ###Check that function is the same##
    #levels=np.linspace(0,1,11)
    plt.figure()
    #levels=np.linspace(-5,5,11)
    tanh_levels=np.linspace(-1,1,11)
    PLayers=vec_get_neuron_values(gX,time_weights)#Layers*(wpad,xpad,time,layersize)
    #Pred=np.squeeze(PLayers[-1])
    np_net=np.squeeze(PLayers[-1])
    tanh_Pred=np.tanh(np_net)
    if PlotHM:
        plt.contourf(gX[:,:,0],gX[:,:,1],tanh_Pred,
                     cmap=plt.cm.bwr_r,vmin=0,vmax=1,levels=tanh_levels)
        plt.savefig(record_dir+'/prob_heatmap.pdf')
        #plt.show()
        #stophere
        print 'done with heatmap'


    ##Testing Bool Formula
    t_leaves,t_values=zip(*terminal_leafs.items())
    t_gammas,t_betas=zip(*[val['affine'] for val in t_values])
    t_Gamma=np.stack(t_gammas)
    t_Beta=np.stack(t_betas)
    EvalArray=np.dot(gX,t_Gamma)+t_Beta
    EvalArray=EvalArray[...,0]
    EvalThresh=EvalArray>=0
    BoolGridList=splitL(EvalThresh)#iterates along last axis
    NumeGridList=splitL(EvalArray)

    if BoolSub:
        #This vectorizatio op itself takes time
        t0=time.time()
        vec_BoolOp=lambdify(t_leaves,Bool_Tree)#vectorize sympy substitutions
        print 'Boolean lambidify took ',time.time()-t0, 's'
        t0=time.time()
        bool_pred=vec_BoolOp(*BoolGridList)
        print 'Bool Subs call took ',time.time()-t0,'s'
        gridP=bool_pred.astype('float')

        if BoolPlot:
            plt.figure()
            plt.contourf(gX[:,:,0],gX[:,:,1],gridP)
            plt.colorbar()
            plt.title('(vec)bool formula output')
            plt.savefig(record_dir+'/(vec)booloutput'+config_str+'.pdf')
            print 'done with (vec)boolmap'

        Pred=(tanh_Pred>=0).astype('float')
        Match=(Pred==gridP)
        Match=(Pred==gridP)
        Match=(Pred==gridP).astype('float')
        if BoolMatchPlot:
            plt.figure()
            plt.contourf(gX[:,:,0],gX[:,:,1],Match,levels=[-1.,0.,1.,2.])
            plt.title('Pred match bool')
            plt.colorbar()
            plt.savefig(record_dir+'/MatchBool_'+config_str+'.pdf')

    if MinMaxSub:
        if len(t_leaves)==1:#no "Max or Min to lambdify
            print 'only 1 leaf!'
            mm_net=NumeGridList[0]#one item
        else:
            t0=time.time()
            #vec_MinMaxOpt=lambdify(t_leaves,MinMax_Tree)#vectorize sympy substitutions
            import tensorflow as tf
            tf.reset_default_graph()
            sess=tf.Session()
            vec_MinMaxOpt=lambdify(t_leaves,MinMax_Tree,'tensorflow')#sympy doesn't support np.maximum
            print 'MinMax lambidify took ',time.time()-t0, 's'
            t0=time.time()
            tf_mm_net=vec_MinMaxOpt(*NumeGridList)
            print 'MinMax Subs call took ',time.time()-t0,'s'
            t0=time.time()
            mm_net=sess.run(tf_mm_net)
            print 'sess.run(tf_mm_net) time ',time.time()-t0,'s'

        if MinMaxPlot:
            plt.figure()
            plt.contourf(gX[:,:,0],gX[:,:,1],mm_net)
            plt.colorbar()
            plt.title('(vec)MinMax formula output')
            plt.savefig(record_dir+'/(vec)booloutput_'+config_str+'.pdf')

        if MinMaxMatchPlot:
            fig,axes=plt.subplots(1,3)
            ax_net,ax_mm,ax_diff=axes
            ctf_net=ax_net.contourf(gX[:,:,0],gX[:,:,1],np_net)
            ctf_mm=ax_mm.contourf(gX[:,:,0],gX[:,:,1],mm_net)
            ctf_diff=ax_diff.contourf(gX[:,:,0],gX[:,:,1],mm_net-np_net)
            for ax in axes:
                ax.set(adjustable='box-forced',aspect='equal')
            fig.colorbar(ctf_net,ax=ax_net)
            fig.colorbar(ctf_mm,ax=ax_mm)
            fig.colorbar(ctf_diff,ax=ax_diff)
            ax_net.set_title('Net(x)')
            ax_mm.set_title('MinMaxExpr(x)')
            ax_diff.set_title('MinMax minus Actual')
            plt.savefig(record_dir+'/MinMaxSideby_'+config_str+'.pdf')
        print 'done with mm_net'

    #Numerical Inspect
    #loc=np.array([500,200])
    #x3=gX[loc[0],loc[1]]
    #states=[(p>=0).astype('int') for p in PLayers]
    #gX_weights=rescale_weights(time_weights,states[:-1])#dont get last one
    #gX_gamma,gX_beta=compose_affine( gX_weights )
    #gX_ComposeNet=np.sum(gX[...,None]*gX_gamma,axis=2)+gX_beta
    #ca_net=np.squeeze(gX_ComposeNet)
    #net_val3=np_net[loc[0],loc[1]]
    #ca_val3=ca_net[loc[0],loc[1]] #same as np
    #mm_val3=mm_net[loc[0],loc[1]]
    #print 'net_val:',net_val3,'\n','mm_val:',mm_val3


    print 'log_dir:',log_dir
    print 'config:',config_str
    if ExistPlots:
        plt.show()


