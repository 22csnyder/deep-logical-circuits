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
from sympy import Max,Min
from sympy.utilities.lambdify import lambdify

'''
This iteration of bool expansion is more flexible than v1 but still doesnt
work. The mistakes are very slight numerical mistakes on small networks and
large mistakes on bigger networks.

I think the failure is that mu/tau need to be decoupled per layer. Even though
terms with mu,tau mismatch never are the most critical, not doing And over
EVERY tau for a given mu can change which mu are optimal at that moment.


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


def AndOr2MaxMin(Op):
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

from ArrayDict import ArrayDict
if __name__=='__main__':
    plt.close('all')
    #log_dir='./logs/Model_0220_093548_triforce test 4442'
    #log_dir='./logs/Model_0220_102206_triforce_mac_test'##Code dev done on this
    #log_dir='./logs/Model_0221_151433_trforce7772_positiveweights'
    #log_dir='./logs/Model_0424_183723_checker332'
    #log_dir='./logs/Model_0426_103718_checker332'
    #log_dir='./logs/Model_0426_113703_checker332_boolv1_succeed'#eff 1hl
    #log_dir='./logs/Model_0130_034237_bulli_test'#1HL OR

    #log_dir='./logs/Model_0426_114651_checker332_bool_vexp'#v1,v2 work #All works
    #log_dir='./logs/Model_0221_123649_Valley7772'##CHALLENGE
    #log_dir='./logs/Model_0220_142344_valley'#011 works. much explored
    log_dir='./logs/Model_0221_101830_R2Clean4442'

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
    GridX=resample_grid(gridX)#200
    HighResX=resample_grid(gridX,5000)
    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)


    #Does everything at last time point
    trn_iter=-1
    #if True:
    if False:
        if 'positiveweights' not in log_dir:
            raise ValueError('are you sure you want to relu weights?')
    #if False:
        print 'WARNING. Making mul_weights positive'
        time_weights=[[w[trn_iter],b[trn_iter]] for w,b in del_weights]#d*2*wtshape
        pos_weights=[[relu(w[trn_iter]),b[trn_iter]] for w,b in del_weights]

        #time_weights=time_weights[:1]+pos_weights[1:]
        print 'WARNING. DEBUG RELU Weights'
        #dont pos last layer
        time_weights=time_weights[:1]+pos_weights[1:-1]+time_weights[-1:]
    else:
        if 'positiveweights' in log_dir:
            print 'WARN relu not applied to weights'
        time_weights=[[w[trn_iter],b[trn_iter]] for w,b in del_weights]#d*2*wtshape

    pth_Sig =record_dir+'/res5000_Sig.npy'
    pth_Sig0=record_dir+'/res5000_Sig0.npy'
    if os.path.exists(pth_Sig):
        Sig=np.load(pth_Sig)
        Sig0=np.load(pth_Sig0)
    else:
        print 'Getting high res Sig data'
        Sig,Centers,Cnts,Idx0=get_net_states(HighResX,time_weights,return_bdry_idx=True)
        Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]
        np.save(record_dir+'/res5000_Sig.npy',Sig)
        np.save(record_dir+'/res5000_Sig0.npy',Sig0)
        print 'finished with Sig data'

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



    #config
    WhichSig='All'
    #WhichSig=0

    OrAndPivot=2
        #1 only start last layer with init_op, alternate
        #2 always start each layer with init_op
    LeafSigMethod=1
        #1 sets new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
        #2 LeafSig includes all list_unq_sig[hidden_layer]

    #init_op=Or
    init_op=And

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
        #print 'leaf:',a_leaf
        #print 'G',gamma,'b',beta,'\n'
        #print 'DEBUG','IDX_REMAIN_SIG',idx_remain_sig
        leaf_sig=list_unq_sig[hidden_layer][idx_remain_sig]

            #leaf_sig=list_unq_sig[hidden_layer]#simply use all
        #leaf_weights=time_weights[hidden_layer:]#includes weights feeding into layer
        leaf_weights=[time_weights[hidden_layer],[gamma,beta]]#includes weights feeding into layer

        gamma_pos=(gamma>=0).astype('int').ravel()
        gamma_neg=(gamma< 0).astype('int').ravel()
        pos_sig,pos_inv=np.unique(gamma_pos*leaf_sig,axis=0,return_inverse=True)
        neg_sig,neg_inv=np.unique(gamma_neg*leaf_sig,axis=0,return_inverse=True)
        unq_sig,unq_inv=np.unique(leaf_sig,axis=0,return_inverse=True)

        ##How to decide op order##
        ###OLD v1###
        if OrAndPivot==1:#old#v1
            c1_op=leaf_parent_operator
        ####always Or###
        elif OrAndPivot==2:#new
            c1_op=init_op
        c2_op=swap_bool(c1_op)#always
        c1_opt=AndOr2MaxMin(c1_op)
        c2_opt=AndOr2MaxMin(c2_op)

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
        c1_opt_list=[]
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

                if LeafSigMethod==1:#v1
                    new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
                elif LeafSigMethod==2:
                    new_idx_remain_sig=idx_remain_sig

                new_leaf_info={
                                'last_operator'  :last_operator,
                                'idx_remain_sig' :new_idx_remain_sig,
                                'affine'         :[new_Gamma[i2],new_Beta[i2]],
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

                new_idx_remain=idx_remain_sig[c2_inv==i2]
                sig2=c2_sig[i2]
            c1_sym_list.append( c2_op(*c2_sym_list ) )
            c1_opt_list.append( c2_opt(*c2_sym_list) )
        new_branch=c1_op(*c1_sym_list)
        new_branch_opt=c1_opt(*c1_opt_list)
        #print new_branch

        Bool_Tree=Bool_Tree.subs(a_leaf,new_branch)
        MinMax_Tree=MinMax_Tree.subs(a_leaf,new_branch_opt)
    print 'Bool Tree','\n',Bool_Tree
    print 'MinMax Tree','\n',MinMax_Tree


    List_Leaf=terminal_leafs.values()
    List_Leaf.sort(key=lambda info:info['idx_remain_sig'][0])

    Gamma=np.stack([L['affine'][0] for L in List_Leaf],axis=0)
    Beta =np.stack([L['affine'][1] for L in List_Leaf],axis=0)

    #stophere

    ##PlotConfig
    PlotHM=True
    res=800
    BoolSub=1
    BoolPlot=1
    BoolMatchPlot=1

    MinMaxSub=1
    MinMaxPlot=1
    MinMaxMatchPlot=1


    BP=BoolSub*(BoolPlot+BoolMatchPlot)
    MP=MinMaxSub*(MinMaxPlot+MinMaxMatchPlot)
    ExistPlots=(BP+MP>=1)

    #gX=gridX#much faster
    #gX=GridX
    gX=resample_grid(gridX,res)#slower


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
    EvalArray=np.squeeze(np.dot(gX,t_Gamma)+t_Beta)
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
        mm_net=sess.run(tf_mm_net)

        if MinMaxPlot:
            plt.figure()
            plt.contourf(gX[:,:,0],gX[:,:,1],mm_net)
            plt.colorbar()
            plt.title('(vec)MinMax formula output')
            plt.savefig(record_dir+'/(vec)booloutput_'+config_str+'.pdf')

        if MinMaxMatchPlot:
            #plt.figure()
            #plt.contourf(gX[:,:,0],gX[:,:,1],mm_net-np_net)
            #plt.colorbar()
            #plt.title('MinMax minus Actual')
            #plt.savefig(record_dir+'/MinMaxMismatch_'+config_str+'.pdf')

            fig,axes=plt.subplots(1,3)
            ax_net,ax_mm,ax_diff=axes
            ctf_net=ax_net.contourf(gX[:,:,0],gX[:,:,1],np_net)
            ctf_mm=ax_mm.contourf(gX[:,:,0],gX[:,:,1],mm_net)
            ctf_diff=ax_diff.contourf(gX[:,:,0],gX[:,:,1],mm_net-np_net)
            fig.colorbar(ctf_net,ax=ax_net)
            fig.colorbar(ctf_mm,ax=ax_mm)
            fig.colorbar(ctf_diff,ax=ax_diff)
            for ax in axes:
                ax.set(adjustable='box-forced',aspect='equal')
            ax_net.set_title('Net(x)')
            ax_mm.set_title('MinMaxExpr(x)')
            ax_diff.set_title('MinMax minus Actual')
            plt.savefig(record_dir+'/MinMaxSideby_'+config_str+'.pdf')
        print 'done with mm_net'

    #mm_net
    #np_net
    loc=np.array([500,200])
    x3=gX[loc[0],loc[1]]
    states=[(p>=0).astype('int') for p in PLayers]
    gX_weights=rescale_weights(time_weights,states[:-1])#dont get last one
    gX_gamma,gX_beta=compose_affine( gX_weights )
    gX_ComposeNet=np.sum(gX[...,None]*gX_gamma,axis=2)+gX_beta
    ca_net=np.squeeze(gX_ComposeNet)
    net_val3=np_net[loc[0],loc[1]]
    ca_val3=ca_net[loc[0],loc[1]] #same as np
    mm_val3=mm_net[loc[0],loc[1]]




    print 'config:',config_str
    if ExistPlots:
        plt.show()


    ##if False:
    #if True:
    #    #i=10
    #    #j=11
    #    gX=gridX#much faster
    #    #if True:
    #    if False:
    #        print 'WARN if slowdown try sparser grid'
    #        gX=GridX
    #    gridP=np.zeros_like(gX[:,:,0])
    #    for i in range(gX.shape[0]):

    #        for j in range(gX.shape[1]):
    #            x=gX[i,j]
    #            leaf_evals={k:(np.dot(x,n['affine'][0])+n['affine'][1]>=0.)[0] for k,n in
    #                        terminal_leafs.items()}
    #            bool_pred=Bool_Tree.subs(leaf_evals)
    #            gridP[i,j]=np.float(bool_pred is sympy.true)
    #            #gridP[i,j]=np.float(bool_pred)


    #    plt.figure()
    #    plt.contourf(gX[:,:,0],gX[:,:,1],gridP)
    #    plt.colorbar()
    #    plt.title('bool formula output')
    #    plt.savefig(record_dir+'/booloutput'+config_str+'.pdf')
        #print 'done with boolmap'
        #plt.show()






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



    #DEBUG
    Gamma=np.squeeze(Gamma)
    #Beta=np.squeeze(Beta)
    Linear=np.squeeze(Linear)
    #Bias=np.squeeze(Bias)

    Linear0=np.squeeze(Linear0)


