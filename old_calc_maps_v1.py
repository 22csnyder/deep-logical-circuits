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

'''
This is a save copy from the first time I tried to implement boolean network
This method keeps mu,tau consistent with all previous mu and tau


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
    PLayers=vec_get_neuron_values(X,weights)#Layers*(wpad,xpad,time,layersize)
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
    '''
    wts=weights
    sigl=activations

    w0,b0=wts[0]
    wpad=len(w0.shape)-2
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
    xdim=w0.shape[-2]

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



def swap_bool(Op):
    if Op is Or:
        return And
    elif Op is And:
        return Or
    else:
        raise ValueError('Either simpy.And or simpy.Or must be passed')

from ArrayDict import ArrayDict
if __name__=='__main__':
    #log_dir='./logs/Model_0220_093548_triforce test 4442'
    #log_dir='./logs/Model_0220_102206_triforce_mac_test'##Code dev done on this
    log_dir='./logs/Model_0221_151433_trforce7772_positiveweights'

    #Setup
    record_dir=os.path.join(log_dir,'records')
    id_str=str(file2number(log_dir))
    all_weights=load_weights(log_dir)
    all_step=np.load(get_path('step','wwatch',log_dir))
    dt=10
    weights=[[w[::dt],b[::dt]] for w,b in all_weights]
    arch=[b.shape[-1] for w,b in weights[:-1]]#net architecture
    del_weights=get_del_weights(weights)
    step=all_step[::dt]
    del_weights=get_del_weights(weights)
    gridX=np.load(get_path('gridX','hmwatch',log_dir))
    GridX=resample_grid(gridX)
    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)


    #Does everything at last time point
    time=-1
    if True:
        if 'positiveweights' not in log_dir:
            raise ValueError('are you sure you want to relu weights?')
    #if False:
        print 'WARNING. Making mul_weights positive'
        time_weights=[[w[time],b[time]] for w,b in del_weights]#d*2*wtshape
        pos_weights=[[relu(w[time]),b[time]] for w,b in del_weights]

        #time_weights=time_weights[:1]+pos_weights[1:]
        print 'WARNING. DEBUG RELU Weights'
        #dont pos last layer
        time_weights=time_weights[:1]+pos_weights[1:-1]+time_weights[-1:]
    else:
        if 'positiveweights' in log_dir:
            print 'WARN relu not applied to weights'
        time_weights=[[w[time],b[time]] for w,b in del_weights]#d*2*wtshape



    ##exp worked
    ##time_weights=[ time_weights[0],[time_weights[1][0][:,1:2],time_weights[1][1][1:2]] ]
    #time_weights=[ time_weights[0],[time_weights[1][0][:,2:3],time_weights[1][1][2:3]] ]
    #arch=[4]

    #exp2.worked with sigl0 once I sorted out a bug
    #arch=[4,4]
    #time_weights=[ time_weights[0],
    #               time_weights[1],
    #              [time_weights[2][0][:,2:3],time_weights[2][1][2:3]]
    #             ]

    ###Should be GridX!DEBUG
    #Sig,Centers,Cnts,Idx0=get_net_states(gridX,time_weights,return_bdry_idx=True)
    Sig,Centers,Cnts,Idx0=get_net_states(GridX,time_weights,return_bdry_idx=True)

    Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]


    #future, could aggregate across time before this part
    sigl=np.split(Sig,#split up into layers again
                  indices_or_sections=np.cumsum(arch)[:-1],
                  axis=-1)


    sigl0=np.split(Sig0,#split up into layers again
                  indices_or_sections=np.cumsum(arch)[:-1],
                  axis=-1)

###Build Boolean###

    #using sigl for DEBUG
    #n_sig=len(sigl[0])
    #depth=len(sigl)

    ###arguments
    #sigl (sigl or sigl0)
    #set of weights
    ###returns
    #terminal_leafs, Bool_Tree

    list_unq_sig=sigl0
    #list_unq_sig=sigl

    #init
    n_sig=len(list_unq_sig[0])
    depth=len(list_unq_sig)
    Bool_Tree=symbols('0')
    leaf=Bool_Tree.atoms().pop()#atoms() returns a set
    leaf_stack={leaf:{
                      'last_operator'  :Or,#default start
                      'idx_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
                      'hidden_layer'   :depth,#0,...,depth-1(first is dummy)
                      'affine'         :time_weights[-1],
                     }
               }
    debug_stack=leaf_stack.copy()#shallow
    terminal_leafs={}#so ctrl-f finds it


    LOOP=-1
    while leaf_stack:#until empty
        LOOP+=1#DEBUG
        #start loop
        a_leaf=leaf_stack.keys()[0]
        leaf_info=leaf_stack.pop(a_leaf)#while nonempty
        leaf_parent_operator=leaf_info['last_operator']#sympy Or/And
        idx_remain_sig=leaf_info['idx_remain_sig']
        ##Increment## hidden_layer -= 1
        hidden_layer=leaf_info['hidden_layer']-1   #hidden layers iterate 0,..,depth-1
        gamma,beta=leaf_info['affine']#current linear map
        #bool_remain_sig=leaf_info['bool_remain_sig']
        print 'leaf:',a_leaf
        print 'G',gamma,'b',beta,'\n'


        leaf_sig=list_unq_sig[hidden_layer][idx_remain_sig]
        #leaf_weights=time_weights[hidden_layer:]#includes weights feeding into layer
        leaf_weights=[time_weights[hidden_layer],[gamma,beta]]#includes weights feeding into layer

        gamma_pos=(gamma>=0).astype('int').ravel()
        gamma_neg=(gamma< 0).astype('int').ravel()
        pos_sig,pos_inv=np.unique(gamma_pos*leaf_sig,axis=0,return_inverse=True)
        neg_sig,neg_inv=np.unique(gamma_neg*leaf_sig,axis=0,return_inverse=True)
        unq_sig,unq_inv=np.unique(leaf_sig,axis=0,return_inverse=True)

        c1_op=leaf_parent_operator
        c2_op=swap_bool(c1_op)

        if c1_op is Or:
            c1_sig,c1_inv=pos_sig,pos_inv
        else:
            c1_sig,c1_inv=neg_sig,neg_inv
        c2_sig,c2_inv=unq_sig,unq_inv

        new_leaf_weights=rescale_weights(leaf_weights,[c2_sig])#(d+1*2*wtshape, d*n_sigxnl)
        new_Gamma,new_Beta=compose_affine(new_leaf_weights)# num_remain x layer_l shape x layer l-1 shape

        sig2name=lambda a: ''.join([str(b) for b in a])
        unq_name_exts=[sig2name(sig) for sig in unq_sig]
        c1_sym_list=[]
        for i1,sig1 in enumerate(c1_sig):
            i2_corresp_i1=np.unique(c2_inv[c1_inv==i1])
            if len(i2_corresp_i1)>1:
                last_operator=c2_op
            else:
                last_operator=c1_op

            #print last_operator
            c2_sym_list=[]
            for i2 in i2_corresp_i1:
                new_name=unq_name_exts[i2]+'.'+str(a_leaf)#progress right to left
                new_node=symbols(new_name)
                c2_sym_list.append(new_node)
                new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
                new_leaf_info={
                                'last_operator'  :last_operator,
                                #'last_operator'  :Or,#DEBUG
                                #'last_operator'  :And,#DEBUG
                                'idx_remain_sig' :new_idx_remain_sig,
                                'affine'         :[new_Gamma[i2],new_Beta[i2]],
                                #'name'           :new_node,
                              }

                if hidden_layer>0:
                    new_leaf_info['hidden_layer']=hidden_layer
                    leaf_stack[new_node]=new_leaf_info
                else:
                    assert(len(new_idx_remain_sig)==1)#corresp to 1 guy
                    terminal_leafs[new_node]=new_leaf_info

                #debug at top of network
                if hidden_layer>=1:
                    debug_stack[new_node]=new_leaf_info

                new_idx_remain=idx_remain_sig[c2_inv==i2]
                sig2=c2_sig[i2]
            c1_sym_list.append( c2_op(*c2_sym_list) )
        new_branch=c1_op(*c1_sym_list)
        #print new_branch

        Bool_Tree=Bool_Tree.subs(a_leaf,new_branch)
        #print Bool_Tree


    List_Leaf=terminal_leafs.values()
    List_Leaf.sort(key=lambda info:info['idx_remain_sig'][0])

    Gamma=np.stack([L['affine'][0] for L in List_Leaf],axis=0)
    Beta =np.stack([L['affine'][1] for L in List_Leaf],axis=0)



    ##Testing Bool Formula
    if False:
    #if True:
        #i=10
        #j=11
        gX=gridX#much faster
        if True:
            print 'WARN if slowdown try sparser grid'
            gX=GridX
        gridP=np.zeros_like(gX[:,:,0])
        for i in range(gX.shape[0]):
            for j in range(gX.shape[1]):
                x=gX[i,j]
                leaf_evals={k:(np.dot(x,n['affine'][0])+n['affine'][1]>=0.)[0] for k,n in
                            terminal_leafs.items()}
                bool_pred=Bool_Tree.subs(leaf_evals)
                gridP[i,j]=np.float(bool_pred is sympy.true)
                #gridP[i,j]=np.float(bool_pred)


        plt.contourf(gX[:,:,0],gX[:,:,1],gridP)
        plt.colorbar()

        plt.show()









###This is good functioning code. Just need something else right now###
#    #has to be adj when count out as state
    new_weights=rescale_weights(time_weights,sigl)#(d+1*2*wtshape, d*n_sigxnl)
    Linear,Bias=compose_affine(new_weights)
    new_weights0=rescale_weights(time_weights,sigl0)#(d+1*2*wtshape, d*n_sigxnl)
    Linear0,Bias0=compose_affine(new_weights0)
#    cnt_thresh=np.percentile(Cnts,5)
#    ix5=np.where( Cnts>=cnt_thresh )[0]
#
#    M=np.concatenate([Linear,np.expand_dims(Bias,-1)],axis=-2)#idx,xdim+1,1
#    cnt_thresh=np.percentile(Cnts,70)
#    ix_thresh=np.where( Cnts>=cnt_thresh )[0]
#
#    M2=M/np.linalg.norm(M,ord=2,axis=(-2,-1),keepdims=True)
#    M2=np.squeeze(M2)
#    print np.around(M2[ix_thresh],2)


    #####Plot contour####
    ####Check that function is the same##
    ##levels=np.linspace(0,1,11)
    #levels=np.linspace(-5,5,11)
    #PLayers=vec_get_neuron_values(GridX,time_weights)#Layers*(wpad,xpad,time,layersize)
    #Pred=np.squeeze(PLayers[-1])
    #plt.contourf(GridX[:,:,0],GridX[:,:,1],Pred,
    #             cmap=plt.cm.bwr_r,vmin=0,vmax=1,levels=levels)
    #plt.show()
    #stophere

    #DEBUG
    Gamma=np.squeeze(Gamma)
    #Beta=np.squeeze(Beta)
    Linear=np.squeeze(Linear)
    #Bias=np.squeeze(Bias)

    Linear0=np.squeeze(Linear0)


