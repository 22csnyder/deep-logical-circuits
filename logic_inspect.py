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
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import sympy
from sympy import symbols
from sympy.logic.boolalg import And,Or
from sympy import Max,Min,srepr
from sympy.utilities.lambdify import lambdify
from sympy import preorder_traversal,postorder_traversal


from vis_utils import (split_posneg , get_path, get_np_network,
                        get_neuron_values, splitL, load_weights,
                        resample_grid,vec_get_neuron_values,
                        get_del_weights  )
from tboard import file2number
from nonlinearities import relu#np version


import pickle
#import dill


##a np.nan workaround
#    val_names=[]
#    trn_names=[]
#    for ix in range(len(val_true_frac)):
#        if np.isnan(val_true_frac[ix]):
#            val_names.append('NA')
#        else:
#            val_names.append(str((100*np.round(val_true_frac[ix],2)).astype('int')))
#        if np.isnan(trn_true_frac[ix]):
#            trn_names.append('NA')
#        else:
#            trn_names.append(str((100*np.round(trn_true_frac[ix],2)).astype('int')))
#    
#    print 'names',trn_names,val_names
#    name_list=[tn+'('+vn+')' for vn,tn in zip(val_names,trn_names)]
def show_composite(composite,val_true_frac,trn_true_frac,ax=None,hide_xlabel=False):
    if not ax:
        fig,ax=plt.subplots()

    val_true_perc=(100*np.round(val_true_frac,2)).astype('int')
    trn_true_perc=(100*np.round(trn_true_frac,2)).astype('int')

    #name_list=[str(vtp)+'('+str(ttp)+')' for vtp,ttp in zip(val_true_perc,trn_true_perc)]
    name_list=[str(ttp)+'('+str(vtp)+')' for vtp,ttp in zip(val_true_perc,trn_true_perc)]

    #for ix in range(len(name_list)):
    #    if val_true_frac[ix]==np.nan or trn_true_frac[ix]==np.nan:
    #        name_list[ix]='NA'

    #name_list[-1]=name_list[-1]+'%'
    location=28*np.arange(10)+14
    #name_list.append('%')
    #location=28*np.arange(11)+14

    ax.imshow(composite,cmap='gray')
    ax.set_xticks(location)
    ax.set_xticklabels(name_list)
    if hide_xlabel:
        ax.set_xticks([])
        ax.set_xticklabels([])
    ax.set_yticks([14,28+14])
    #ax.set_yticklabels(['TRUE','FALSE'])
    ax.set_yticklabels(['T','F'])
    ax.tick_params(axis=u'both', which=u'both',length=0)

    for item in ax.get_xticklabels():
        item.set_fontsize(11)
    for item in ax.get_yticklabels():
        item.set_fontsize(20)



if __name__=='__main__':
    plt.close('all')


    #log_dir='./logs/Model_0513_120107_MnistPCA3A2_labs01.67'#toy
    #log_dir='./logs/Model_0513_162218_Mnist55kA2PCA4_LabsAll'#bigboy

    #log_dir='./logs/Model_0514_010526_Mnist20kA3PCA4_LabsAll'
    #   |Sig0|=139

    #log_dir='./logs/Model_0514_011608_Mnist5kA3PCA4_LabsAll'

    #log_dir='./logs/Model_0514_012756_Mnist5kA3PCA4_LabsAll'
    #97/89

    #log_dir='./logs/Model_0514_013017_Mnist5kA3PCA4_LabsAll_50kiter'
    #95/84 |Sig0|=135,|Sig|=3213 #still working

    #log_dir='./logs/Model_0514_014452_Mnist5kA3PCA4_LabsAll_50kiter';m=5000
    #|Sig0|=71   .92/.83
    #Learns 6,7 and 5,8,9 separately then combines with OR

    #Same experimental settings for the next two!
    #log_dir='./logs/Model_0514_162548_Mnist1kA3PCA4_LabsAll_50kiter';m=1000
    #0.91/0.91 |Sig0|=3 |Sig|=191
    ##Essentially just a linear classifier. The 4's get looped in with the pos
    #Essentially all validation take one path--others may be source of adversarial attack

    ###FigC###
    #log_dir='./logs/Model_0514_151155_Mnist1kA3PCA4_LabsAll_50kiter';m=1000
    #1./.781 |Sig0|=55 |Sig|=2432
    ###SORTOF interesting. Similar to 162548 but builds in exception for half the 4's

    #Try A1
    #log_dir='./logs/Model_0514_201828_Mnist20kA1PCA6_LabsAll_50kiter'#.98/.91
    #|Sig| 44/182    ###Some interesting structure And0.0

    ###FigB###
    log_dir='./logs/Model_0514_210136_Mnist50kA1PCA6_LabsAll_50kiter';m=50000
    #.96/.93 #65/386
    ##Also interesting 5.0,6.0
    #1.5.0 has Or( [0,7,9] , [5,6,half7,8,9]) = [0,5,6,7,8,9] = 5.0\approx 0

    ###FigA###
    #log_dir='./logs/Model_0514_220539_Mnist50kA1PCA6_LabsAll_50kiter';m=50000
    #.985/.94    14/147
    #Pretty awesome. Just AND. Should make for a good figure


    print 'm=',m#Remember to set him
    print 'LogDir',log_dir
    descrip,id_str=log_dir.split('_')[-1],str(file2number(log_dir))

    data_fn=get_toy_data('mnist')
    trn_data=data_fn(m=m,return_numpy=True)#mx784
    trn_X=trn_data['input']
    trn_Images=trn_X.reshape([-1,28,28])
    trn_Y=trn_data['label']
    trn_Labels=trn_data['orig label'].ravel()

    from sympy import preorder_traversal,postorder_traversal
    logic_dir=os.path.join(log_dir,'logic')
    decision_dir=logic_dir+'/Arrays'
    split_decision_dir=logic_dir+'/SingleArrays'
    if not os.path.exists(logic_dir):
        os.makedirs(logic_dir)
    if not os.path.exists(decision_dir):
        os.makedirs(decision_dir)
    if not os.path.exists(split_decision_dir):
        os.makedirs(split_decision_dir)


    with open( logic_dir+'/sympy_abstract_Bool_Tree.pkl','rb') as f:
        abstract_Bool_Tree=pickle.load(f)
        abt=abstract_Bool_Tree
    with open( logic_dir+'/sympy_Bool_Tree.pkl','rb') as f:
        Bool_Tree=pickle.load( f)
        BT=Bool_Tree
    with open( logic_dir+'/terminal_leafs.pkl','rb') as f:
        terminal_leafs=pickle.load(f)
    with open( logic_dir+'/leaf_substitute_symbols.pkl','rb') as f:
        subdict=pickle.load(f)

    stophere

    mnist_datasets=mnist_data.read_data_sets('./data/mnist/',reshape=False,validation_size=5000)
    #Images=mnist_datasets.train.images
    #Images=np.squeeze(Images)
    #Labels=mnist_datasets.train.labels
    #fl_inputs=[Images[Labels==i].reshape([-1,784]) for i in range(10)]
    val_Images=np.squeeze(mnist_datasets.validation.images)
    val_Labels=mnist_datasets.validation.labels

    val_X=val_Images.reshape([-1,784])
    #trn_X=trn_X

    #Leafs,List_Leaf=zip(*terminal_leafs.items())#Defines ordering from here on

    nice_leafs={subdict[k]:v for k,v in terminal_leafs.items()}#key shortening

    #t_leaves,t_values=zip(*terminal_leafs.items())
    t_leaves,t_values=zip(*nice_leafs.items())
    t_gammas,t_betas=zip(*[val['affine'] for val in t_values])
    t_Gamma=np.stack(t_gammas)
    t_Beta=np.stack(t_betas)

    print 'Done loading'


    print 'beginning eval numpy..'
    EvalArray=np.dot(val_X,t_Gamma)+t_Beta
    EvalArray=EvalArray[...,0]
    EvalThresh=EvalArray>=0
    EvalBoolGridList=splitL(EvalThresh)#iterates along last axis indexing atoms

    TrainArray=np.dot(trn_X,t_Gamma)+t_Beta
    TrainArray=TrainArray[...,0]
    TrainThresh=TrainArray>=0
    TrainBoolGridList=splitL(TrainThresh)

    ValBool={t_leaves[i]:boolgrid for i,boolgrid in enumerate(EvalBoolGridList)}
    TrnBool={t_leaves[i]:boolgrid for i,boolgrid in enumerate(TrainBoolGridList)}

    #NumeGridList=splitL(EvalArray)#not used

    PeOT=list(preorder_traversal(abstract_Bool_Tree))
    ROOT=PeOT[0]

    ##Try to recurse with Numpy. Lambdify was a bit rough
    #The goal here is to build up ValBool so that it has the truth value of
    #the set of valuation images for every intermediate expression
    print 'Beginning traversal..'
    essential={ROOT:True}
    for expr in postorder_traversal(abstract_Bool_Tree):
        #print expr
        if expr.func==sympy.Symbol:#already in ValBool
            #print 'symbol'
            pass
        elif expr.func==sympy.And:
            #print 'AND'
            ValBool[expr]=np.logical_and.reduce([ ValBool[sym] for sym in expr.args ])
            TrnBool[expr]=np.logical_and.reduce([ TrnBool[sym] for sym in expr.args ])

            for sym in expr.args:#is it false where all others are true?
                where_true= np.logical_and.reduce(
                    [ ValBool[s] for s in expr.args if s!= sym])
                if np.logical_not(ValBool[sym][where_true]).any():
                    essential[sym]=True
                else:
                    essential[sym]=False
        elif expr.func==sympy.Or:
            #print 'OR'
            ValBool[expr]=np.logical_or.reduce([ ValBool[sym] for sym in expr.args ])
            TrnBool[expr]=np.logical_or.reduce([ TrnBool[sym] for sym in expr.args ])
            for sym in expr.args:#is it true where no others are?
                where_false=np.logical_not(
                    np.logical_or.reduce([ ValBool[s] for s in expr.args if s!= sym]))
                if ValBool[sym][where_false].any():
                    essential[sym]=True
                else:
                    essential[sym]=False
        else:
            raise ValueError('Parent operator should wasnt expected to be ',expr.func)
    print '..done!'

        #either expr in ValBool.keys()

    with open( logic_dir+'/essential.pkl','wb') as f:
        pickle.dump(essential,f)
    with open( logic_dir+'/ValBool.pkl','wb') as f:
        pickle.dump(ValBool,f)
    with open( logic_dir+'/TrnBool.pkl','wb') as f:
        pickle.dump(TrnBool,f)



    #stophere

    ##Starting with pre order traversal, go through depth 3 or so
    ###Some of the networks are just too big to do the whole tree
    ##Config##
    depth_limit=3
    ##Config##

    dummy=sympy.Not(ROOT)#Has ROOT as only arg

    expr_stack={}
    expr_info={'depth':0,
                'repr':'',
                #'repr':'0',
              }
    expr_completed={}
    expr_stack={dummy:expr_info}
    repr2expr={}
    #expr_stack={ROOT:expr_info}
    #repr2expr={'0':ROOT}

    val_ids=[np.where(val_Labels==i)[0] for i in range(10)]
    trn_ids=[np.where(trn_Labels==i)[0] for i in range(10)]

    #manual preorder traversal
    while expr_stack:
        expr=expr_stack.keys()[0]
        expr_info=expr_stack.pop(expr)#while nonempty
        expr_depth=expr_info['depth']
        expr_repr=expr_info['repr']
        print 'depth',expr_depth,'expr',expr_info['repr'],expr.func

        if expr in abt.atoms():#linear classifier#not combination
            continue

        Args=expr.args
        LA=len(Args)
        fig,axes=plt.subplots(LA,1,figsize=(8,LA*2))
        axes=[axes] if LA==1 else axes
        for ii,arg in enumerate(Args):

            val_Bools=ValBool[arg]
            val_mask=val_Bools.astype('float')[:,None,None]
            val_true_frac=[np.mean(val_mask[ix]) for ix in val_ids]
            assert(len(val_mask.shape)==len(val_Images.shape))#else potential for memory crash

            trn_Bools=TrnBool[arg]
            trn_mask=trn_Bools.astype('float')[:,None,None]
            trn_true_frac=[np.mean(trn_mask[ix]) for ix in trn_ids]
            assert(len(trn_mask.shape)==len(trn_Images.shape))#else potential for memory crash


            ###Switch by commenting###
            ##Use either val or train images##
            val_pos_msk_imgs=val_mask*val_Images
            val_neg_msk_imgs=(1.-val_mask)*val_Images
            val_pos_Agg=[np.mean(val_pos_msk_imgs[ix],axis=0) for ix in val_ids]
            val_neg_Agg=[np.mean(val_neg_msk_imgs[ix],axis=0) for ix in val_ids]
            val_maxes=[max(pmi.max(),nmi.max()) for pmi,nmi in zip(val_pos_Agg,val_neg_Agg)]
            val_pos_Agg=[pA/m for pA,m in zip(val_pos_Agg,val_maxes)]
            val_neg_Agg=[nA/m for nA,m in zip(val_neg_Agg,val_maxes)]
            composite=np.vstack([np.hstack(val_pos_Agg),np.ones([1,280]),np.hstack(val_neg_Agg)])

            #trn_pos_msk_imgs=trn_mask*trn_Images
            #trn_neg_msk_imgs=(1.-trn_mask)*trn_Images
            #trn_pos_Agg=[np.mean(trn_pos_msk_imgs[ix],axis=0) for ix in trn_ids]
            #trn_neg_Agg=[np.mean(trn_neg_msk_imgs[ix],axis=0) for ix in trn_ids]
            #trn_maxes=[max(pmi.max(),nmi.max()) for pmi,nmi in zip(trn_pos_Agg,trn_neg_Agg)]
            #trn_pos_Agg=[pA/m for pA,m in zip(trn_pos_Agg,trn_maxes)]
            #trn_neg_Agg=[nA/m for nA,m in zip(trn_neg_Agg,trn_maxes)]
            #composite=np.vstack([np.hstack(trn_pos_Agg),np.ones([1,280]),np.hstack(trn_neg_Agg)])

            show_composite(composite,val_true_frac,trn_true_frac,ax=axes[ii])

            arg_repr=str(ii)+'.'+expr_repr if expr_repr else str(ii)
            repr2expr[arg_repr]=arg


            arg_info={'repr'     :arg_repr,
                      'depth'    :expr_depth+1,
                      'composite':composite,
                      'val true frac':val_true_frac,
                      'trn true frac':trn_true_frac,
                     }
            if expr_depth<depth_limit:
                expr_stack[arg]=arg_info
            expr_completed[arg]=arg_info


        #str_func=str(expr.func) if expr.func!=sympy.Not else ''
        str_func=str(expr.func) if expr.func!=sympy.Not else 'ROOT'
        plt.savefig(decision_dir+'/TruthTable_'+id_str+'_'+str(str_func)+str(expr_repr)+'.png')
    print 'finished preorder trav'


    with open( logic_dir+'/expr_composite.pkl','wb') as f:
        pickle.dump(expr_completed,f)



    for expr,info in expr_completed.items():

        fig,ax=plt.subplots(figsize=(8,2))
        #str_func=str(expr.func) if expr.func!=sympy.Not else 'ROOT'
        str_func=str(expr.func) if expr.func!=sympy.Symbol else 'Linear'
        show_composite(info['composite'],info['val true frac'],
                       info['trn true frac'],ax=ax,
                       hide_xlabel=False,
                       )
        save_pth=split_decision_dir+'/TT_'+id_str+'_func'+str(str_func)+str(info['repr'])+'.pdf'
        plt.savefig(save_pth)


        fig,ax=plt.subplots(figsize=(8,2))
        #str_func=str(expr.func) if expr.func!=sympy.Not else 'ROOT'
        str_func=str(expr.func) if expr.func!=sympy.Symbol else 'Linear'
        show_composite(info['composite'],info['val true frac'],
                       info['trn true frac'],ax=ax,
                       hide_xlabel=True,
                       )
        save_pth=split_decision_dir+'/TT_noXLabel_'+id_str+'_func'+str(str_func)+str(info['repr'])+'.pdf'
        plt.savefig(save_pth)


    print 'Done with ',log_dir
    plt.close('all')






