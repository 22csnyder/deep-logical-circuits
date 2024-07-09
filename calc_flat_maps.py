import pandas as pd
import numpy as np
from config import get_config
from utils import prepare_dirs_and_logger,save_config
#from toydata import get_toy_data

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
from sympy import preorder_traversal,postorder_traversal

'''
This is file to test out DNF formulation of operand at boundary

questions that might help with the proof
    is tau2 opt (sigma) the same as tau2 opt (M1m1)?
    is {x:taubar opt} convex?
    can we sequentially set mu2=tau2opt, mu1=tau1opt ... and so on
        to construct mubar opt?

    could visualize every row of mubar,taubar table

    graph out 1-d [mubar,taubar],[taubar,mubar] crossing

    let g1=min_taubar mubar^    g2=min_taubar mubar2^
        is {x:g1<g2} convex?
        is {x:g1 => g2} convex?

    ** if pi in Sig0 is opt for u+=sig(x), is it opt for pi ? min_t [pi,t]?



Analysis todo
    show M_mu tau=phi implies M_mu m_tau

    show never have 2 layers on opposite sides of X_mubar - x - X_taubar
            can be avoided
    induction argument looks identical to what we need for 2HL
    restricting first layer to Sig0 doesnt change tau2 opt unless in X+  -> farka?


    construct mubar by mu2=phi(tau2).. show tau2 doesnt react to choice of mu2.
        now assume m2=t2=v2. (lwr bound of M2m2). Show that t1 doesnt react to
        choice of m1.

    X+ is a union of convex sets indexed by u+ in Sig+;  min_tau [u+,tau] >=0
        the question is whether it is covered by sets indexed by u0 in Sig0.

    Incrementally move between Sig++ and Sig0. Sig0(k)=== would be in Sig0 if
    we truncated each sigbar to (sig1...sigk).
        Sig++(k)==first k coords directly imply N>0.


    Is mM opt with mu=tau !?

    Compare Mm with min [sig0,sig0]

    Can try fixed point iteration (using mu/tau swap)

    if Mm=[pi,eta], how does
        fixing u1=eta1 affect opt of Mm.

    useful to know if {X_sig0} are convex as new Trop formulation says they are

    Compare to trop poly


BUGS sig(x) regions not convex sometimes..
'''



from vis_utils import (split_posneg , get_path, get_np_network,
                        get_neuron_values, splitL, load_weights,
                        resample_grid,vec_get_neuron_values,
                        get_del_weights, subplots)
from tboard import file2number
from nonlinearities import relu#np version


from calc_maps import *

def diff_affine(Aff1,Aff2):
    return [Aff1[0]-Aff2[0],Aff1[1]-Aff2[1]]

def get_Net_Operand(weights):
    signsplit=lambda X : (relu(X),relu(-X))#X=Xp-Xn
    def Net_Operand(mubar,taubar):
        '''
        mubar,taubar should be a length d list of activations
        '''
        OPs_norm=[]
        OPs_conj=[]
        OPs_norm.append(weights[0])
        OPs_conj.append(weights[0])

        for lyr in range(1,len(weights)):#1,..,d
            W,B=weights[lyr]
            mu,tau=mubar[lyr-1],taubar[lyr-1]
            W_pos,W_neg=signsplit(W) #W= W_pos - W_neg
            B_pos,B_neg=signsplit(B)
            Aff_pos=[W_pos,B_pos]
            Aff_neg=[W_neg,B_neg]

            #"norm" just means "not conjugate" or "normal". not length
            R_norm=OPs_norm[-1]
            R_conj=OPs_conj[-1]

            On1=rescale_weights([R_norm,Aff_pos],[mu] )
            On2=rescale_weights([R_conj,Aff_neg],[tau])

            Oc1=rescale_weights([R_conj,Aff_pos],[tau])
            Oc2=rescale_weights([R_norm,Aff_neg],[mu] )

            An1=compose_affine(On1)
            An2=compose_affine(On2)

            Ac1=compose_affine(Oc1)
            Ac2=compose_affine(Oc2)

            Rnew_norm=diff_affine(An1,An2)
            Rnew_conj=diff_affine(Ac1,Ac2)

            OPs_norm.append(Rnew_norm)
            OPs_conj.append(Rnew_conj)

        #return OPs_norm,OPs_conj #debug
        return OPs_norm[-1]
    return Net_Operand


def mtidx2str(mu_idx,tau_idx):#*(mu,tau) idx to a string symbol
    return 'm'+str(mu_idx)+'t'+str(tau_idx)

def build_flat_bool(n_sig,mode='DNF'):
    '''
    This is a function that builds a flat DNF(mode) tree simply given
    n_sig = len(Sig)
    the number of unique sig being used.
    Both mu and tau iterate through all options
    mode = either CNF or DNF
    '''
    ##Flat Fn n_sig->Tree, leaf_stack
    #n_sig=len(sigl[0])
    assert mode in ['CNF','DNF']

    if mode == 'DNF':
        outer_op,inner_op=Or,And
        leaf_stack={}
        outer_sym_list=[]
        for mu_idx in range(n_sig):
            inner_sym_list=[]
            for tau_idx in range(n_sig):
                sym=symbols( mtidx2str(mu_idx,tau_idx) )
                leaf_stack[sym]={'mu_idx':mu_idx,
                                 'tau_idx':tau_idx}
                inner_sym_list.append(sym)
            outer_sym_list.append(  inner_op(*inner_sym_list) )
        Bool_Tree=outer_op(*outer_sym_list)
        return Bool_Tree,leaf_stack

    elif mode == 'CNF':
        outer_op,inner_op=And,Or
        leaf_stack={}
        outer_sym_list=[]
        for tau_idx in range(n_sig):
            inner_sym_list=[]
            for mu_idx in range(n_sig):
                sym=symbols( mtidx2str(mu_idx,tau_idx) )
                leaf_stack[sym]={'mu_idx':mu_idx,
                                 'tau_idx':tau_idx}
                inner_sym_list.append(sym)
            outer_sym_list.append(  inner_op(*inner_sym_list) )
        Bool_Tree=outer_op(*outer_sym_list)
        return Bool_Tree,leaf_stack


def fill_tree(Bool_Tree,Leafs):
    '''
    This function takes Leafs with float_X and bool_X
    filled out and infers the rest of the tree for every expr in bool_tree
    This is done in place so no value is returned
    '''
    ##Try to recurse with Numpy. Lambdify was a bit rough
    #The goal here is to build up ValBool so that it has the truth value of
    #the set of valuation images for every intermediate expression
    print 'Fill out tree values with postorder_traversal..'
    for expr in postorder_traversal(Bool_Tree):
        #print expr

        children=[Leafs[sym] for sym in expr.args]
        float_children=[c['float_X'] for c in children]
        bool_children=[c['bool_X'] for c in children]
        if expr.func==sympy.Symbol:#already in ValBool
            #print 'symbol'
            pass
        elif expr.func==sympy.And:
            #print 'AND'
#            children=[Leafs[sym] for sym in expr.args]
#            float_children=[c['float_X'] for c in children]
#            bool_children=[c['bool_X'] for c in children]

            Leafs[expr]={'float_X':np.minimum.reduce(float_children),
                         'bool_X' :np.logical_and.reduce(bool_children)}
        elif expr.func==sympy.Or:
            #print 'OR'
#            children=[Leafs[sym] for sym in expr.args]
#            float_children=[c['float_X'] for c in children]
#            bool_children=[c['bool_X'] for c in children]

            Leafs[expr]={'float_X':np.maximum.reduce(float_children),
                         'bool_X' :np.logical_or.reduce(bool_children)}
        else:
            raise ValueError('Parent operator should wasnt expected to be ',expr.func)
    print '..done!'


def name_junc(cdJunc):
    s_args=[str(a) for a in cdJunc.args]
    s_mu=[s[:s.find('t')] for s in s_args]
    s_tu=[s[s.find('t'):] for s in s_args]
    id_mu=np.unique(s_mu)
    id_tu=np.unique(s_tu)
    if cdJunc.func is And:
        assert(len(id_mu)==1)
        return int(id_mu[0][1:])
    elif cdJunc.func is Or:
        assert(len(id_tu)==1)
        return int(id_tu[0][1:])



if __name__=='__main__':
    plt.close('all')

    #log_dir='./logs/Model_0220_093548_triforce test 4442'#[*]
    #log_dir='./logs/Model_0220_102206_triforce_mac_test'##Code dev done on this
    #log_dir='./logs/Model_0221_151433_trforce7772_positiveweights'#[*]
    #log_dir='./logs/Model_0424_183723_checker332'
    #log_dir='./logs/Model_0426_103718_checker332'
    #log_dir='./logs/Model_0426_113703_checker332_boolv1_succeed'#eff 1hl
    #log_dir='./logs/Model_0130_034237_bulli_test'#1HL OR
    #log_dir='./logs/Model_0426_114651_checker332_bool_vexp'#v1,v2 work #All works

    ##Code dev done on this
    #v3 A11 works!!
    #works for flat bool
    #log_dir='./logs/Model_0220_102206_triforce_mac_test'

    ##CHALLENGE
    #v3 011 works!!! A11 took took too long
    #log_dir='./logs/Model_0221_123649_Valley7772'

    #v2:011 works. much explored
    #v3: now also A21 works
    #flat bool works
    #revisited this to add mubar,taubar plots
    ##Figure out why these regions are not cvx for CNF
    #log_dir='./logs/Model_0220_142344_valley'

    #Interesting, Xtau* was not convex for CNF (where N>0)
    #log_dir='./logs/Model_0426_103703_checker332'

    #log_dir='./logs/Model_0221_101830_R2Clean4442'#works with v3, lsm1


    #####PubModels#####
    from vis_utils import Pub_Model_Dirs
    ###Index by [arch#-1][data#-1]

    #log_dir=Pub_Model_Dirs[0][2]#[*]
    #log_dir=Pub_Model_Dirs[1][2]#[*]


    ##OH. I thin it's because it is juuuust barely noncvx, (net(x)approx 0)
    #numerical issue probably
    #Some BUG EXISTS #be warned
    #CNF regions look wrong (not convex)
    #CNF actually has both a cvx and a cve pred region
    #It works. but some bug.. pred in CNF[DNF] 
        #regions should be attainable by cvx[ccv] fn
    log_dir=Pub_Model_Dirs[2][2]#[*]


    #Setup
    print 'using log_dir:',log_dir
    record_dir=os.path.join(log_dir,'records')
    id_str=str(file2number(log_dir))
    Mm_dir=os.path.join(log_dir,'Mm_analysis')
    if not os.path.exists(Mm_dir):
        os.makedirs(Mm_dir)

    all_step=np.load(get_path('step','wwatch',log_dir))
    all_weights=load_weights(log_dir)#d*2*Txn1xn2
    del_weights=get_del_weights(all_weights)
    arch=[b.shape[-1] for w,b in del_weights[:-1]]#net architecture

    step=all_step[-1]
    weights=[[w[-1],b[-1]] for w,b in del_weights]#d*2*wtshape
    Wweights,Bweights=zip(*weights)
    #time_weights=weights#backcompat

    gridX=np.load(get_path('gridX','hmwatch',log_dir))
    GridX=resample_grid(gridX)#200
    HighResX=resample_grid(gridX,5000)


    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)


    pth_Sig =record_dir+'/res5000_Sig.npy'
    pth_Sig0=record_dir+'/res5000_Sig0.npy'
    if os.path.exists(pth_Sig):
        Sig=np.load(pth_Sig)
        Sig0=np.load(pth_Sig0)
        print 'finished with Sig data'
    else:
        print 'Getting high res Sig data'
        Sig,Centers,Cnts,Idx0=get_net_states(HighResX,weights,return_bdry_idx=True)
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

    ###CONFIG###

    #WhichSig='All'#Sigbar (mode=Numeric)
    WhichSig=0   #Sigbar0  (mode=Logical)

    FlatMode= 'CNF'
    #FlatMode= 'DNF'

    #gX=GridX
    gX=HighResX#oh boy


    if gX.size==HighResX.size:
        print('WARNING gX=HighResX... may encouter delays')
        HR='_HR'
    else:
        HR=''

    config_str='Sig'+str(WhichSig)+'_'+FlatMode+HR




    print 'Configuration: ',config_str
    ###CONFIG###

    #Take weights, return fn net operand
    Net_Operand=get_Net_Operand(weights)


    if WhichSig=='All':
        pass
    elif int(WhichSig)==0:
        sigl=sigl0


    ##Flat Tree##

    n_sig=len(sigl[0])
    Bool_Tree,Leafs=build_flat_bool(n_sig,FlatMode)

    print 'Net_Operand calls'
    ##Fill with Affine
    for sym,leaf in Leafs.items():
        mubar=[s[leaf['mu_idx']] for s in sigl]
        taubar=[s[leaf['tau_idx']] for s in sigl]
        leaf['affine']=Net_Operand( mubar, taubar )

    print 'Eval each leaf over gX'
    ##Eval over samples
    for sym,leaf in Leafs.items():
        w_aff,b_aff=leaf['affine']
        leaf['float_X']=np.dot(gX,w_aff)+b_aff
        leaf['float_X']=leaf['float_X'][...,0]#last dim 1
        leaf['bool_X']=(leaf['float_X']>=0)


    ##DEBUG##

    #Conclude: Net_Operand(sig,sig) seems to work
    #rs_weights=rescale_weights(weights,sigl)
    #rs_gamma,rs_beta=compose_affine(rs_weights)
    #tstdx=7
    #for tstdx in range(len(sigl[0])):
    #    tst_aff=[rs_gamma[tstdx],rs_beta[tstdx]]

    #    tst_sig=[s[tstdx] for s in sigl]
    #    tst_operand=Net_Operand(tst_sig,tst_sig)
    #    print tst_aff[0]-tst_operand[0],tst_aff[1]-tst_operand[1]



    ##DEBUG##

    #stophere


    fill_tree(Bool_Tree,Leafs)

    PeOT=list(preorder_traversal(Bool_Tree))
    ROOT=PeOT[0]
    eval_root=Leafs[ROOT]['float_X']
    pred_root=Leafs[ROOT]['bool_X']


    PLayers=vec_get_neuron_values(gX,weights)#Layers*(wpad,xpad,time,layersize)
    #Pred=np.squeeze(PLayers[-1])
    eval_net=np.squeeze(PLayers[-1])
    pred_net=eval_net>=0.

    if not (pred_net==pred_root).all():
        print 'ERROR doesnt match network prediction'
    if not (pred_root==(eval_root>=0)).all():
        print 'ERROR Inconsistent tree predictions'


    fig,axes=plt.subplots(1,3,figsize=(15,5))
    ax_net,ax_mm,ax_diff=axes
    ctf_net=ax_net.contourf(gX[:,:,0],gX[:,:,1],eval_net)
    ctf_mm=ax_mm.contourf(gX[:,:,0],gX[:,:,1],eval_root)
    ctf_diff=ax_diff.contourf(gX[:,:,0],gX[:,:,1],eval_root-eval_net)
    for ax in axes:
        ax.set(adjustable='box-forced',aspect='equal')
    fig.colorbar(ctf_net,ax=ax_net)
    fig.colorbar(ctf_mm,ax=ax_mm)
    fig.colorbar(ctf_diff,ax=ax_diff)
    ax_net.set_title('Net(x)')
    ax_mm.set_title('MinMaxExpr(x)')
    ax_diff.set_title('MinMax minus Actual')
    #plt.savefig(record_dir+'/SideBy_FlatOperandFormula_MinMax_vs_net'+config_str+'.pdf')
    plt.savefig(Mm_dir+'/SideBy_FlatOperandFormula_MinMax_vs_net'+config_str+'.pdf')
    print 'done with mm_net'
    plt.close(fig)#not that interesting to look at


    fig,axes=plt.subplots(1,3,figsize=(15,5))
    ax_net,ax_mm,ax_diff=axes
    axes[0].contourf(gX[:,:,0],gX[:,:,1],pred_net)
    axes[1].contourf(gX[:,:,0],gX[:,:,1],pred_root)
    axes[2].contourf(gX[:,:,0],gX[:,:,1],(eval_root>=0))
    for ax in axes:
        ax.set(adjustable='box-forced',aspect='equal')
    axes[0].set_title('Net Pred')
    axes[1].set_title('Root AndOr Pred')
    axes[2].set_title('Root MaxMin Pred')
    plt.savefig(Mm_dir+'/SideBy_FlatOperandFormula_preds'+config_str+'.pdf')



    #See how Sigma and outer* compare
    #Not sure if diff index from Sig
    acts=np.concatenate( PLayers[:-1], axis=-1 )#gX.shape,n_neurons
    paths=(acts>=0.).astype('int')
    fl_paths=paths.reshape([-1,paths.shape[-1]])
    #fl_X=X.reshape([-1,X.shape[-1]])#xdim

    gX_Sig,gX_Inv,gX_Cnts=np.unique(fl_paths,axis=0,
                                    return_inverse=True,
                                    return_counts=True)
    idx_gX_Sig=gX_Inv.reshape(gX.shape[:2])

    Junctions=np.array([Leafs[J]['float_X'] for J in ROOT.args])
    idx_Juncs=np.array([name_junc(J) for J in ROOT.args])
    ss_Junctions=Junctions[idx_Juncs]#"Sig sorted"
    amax_J=np.argmax(ss_Junctions,axis=0)#DNF - mu indexed
    amin_J=np.argmin(ss_Junctions,axis=0)#CNF - tu indexed

    gX0=gX[:,:,0]
    gX1=gX[:,:,1]
    outer_J=amax_J if ROOT.func is Or else amin_J

    #plot Sig* vs outer*
    fig_Junc,ax_Junc=subplots()
    fig_Sig,ax_Sig  =subplots()
    fig_JS,axes     =subplots(1,2)
    ax_Sig2,ax_Junc2=axes
    for aS in [ax_Sig,ax_Sig2]:
        aS.contour(gX0,gX1,idx_gX_Sig,colors='k')
    for aJ in [ax_Junc,ax_Junc2]:
        aJ.contour(gX0,gX1,outer_J,colors='k')
    ax_Sig2.set_title('Sig(x)')
    if FlatMode=='CNF': #smallest of convex functions
        ax_Junc2.set_title('Convex Regions mM')
    elif FlatMode=='DNF': #largest of concave functions
        ax_Junc2.set_title('Concave Regions Mm')
    else:
        raise ValueError('FlatMode is ',FlatMode)

    fig_Sig.savefig(Mm_dir+'/LinearRegions_'+config_str+'.pdf')
    fig_Junc.savefig(Mm_dir+'/sig0_OuterOptimal_Regions_'+config_str+'.pdf')
    fig_JS.savefig(Mm_dir+'/OptRegion_Comparison_Sig.Sig0_'+config_str+'.pdf')
    plt.close(fig_Sig)#mostly for latex
    plt.close(fig_Junc)


#    fig_Junc,ax_Junc=plt.subplots()
#    fig_Sig,ax_Sig=plt.subplots()
#    fig_JS,axes=plt.subplots(1,2)
#    ax_Sig2,ax_Junc2=axes

    fig_evalnet,ax_evalnet=subplots()
    fig_prednet,ax_prednet=subplots()
    #fig_prednet.set_size_inches(6,6)
    #fig_evalnet.set_size_inches(6,6)

    levels11=np.linspace(-1.,1,11)#[5]=0
    norm_eval_net=eval_net/np.max(np.abs(eval_net))
    ax_evalnet.contourf(gX0,gX1,norm_eval_net,cmap=plt.cm.bwr_r,vmin=-1,vmax=+1,levels=levels11)
    ax_prednet.contourf(gX0,gX1,norm_eval_net,cmap=plt.cm.bwr_r,vmin=-1,vmax=+1,levels=[-1.01,0.,1.01])


    #an example
    #cmap = plt.cm.get_cmap("Purples")
    #ctf=ax.contourf(gX0,gX1,(lin_pred>0).astype('float'),
    #           levels=[-0.01,0,1.01],colors=['w',cmap(190)])

    for ax in [ax_prednet,ax_evalnet]:
        ax.contour( gX0,gX1,norm_eval_net,colors='k',levels=[0.],linestyles='dashed')
        ax.contour( gX0,gX1,idx_gX_Sig,colors='k',linewidth=4)#df is 1.5
        ax.contour( gX0,gX1,outer_J,colors='c',linewidth=1.)#,linestyles='dotted')
        #ax.contour( gX0,gX1,outer_J,colors='g',,linewidth=1.,linestyles='dotted')

        #ax.set_aspect('equal')
        #ax.axis('tight')

    fig_evalnet.savefig(Mm_dir+'/Overlay_evalnet_and_opt_sig_mu(tau)0_'+config_str+'.pdf',
                        bbox_inches='tight')
    fig_prednet.savefig(Mm_dir+'/Overlay_prednet_and_opt_sig_mu(tau)0_'+config_str+'.pdf',
                        bbox_inches='tight')
    #ax.contourf(gX0,gX1,norm_eval_net,cmap=plt.cm.bwr_r,vmin=-1,vmax=+1,levels=11)

    #ax.contourf(gX0,gX1,pred_net)
    plt.close(fig_evalnet)


    plt.show(block=False)
    #plt.show()




    #assert( ROOT.func==sympy.Or )
    #junctions=ROOT.args





#    #Compare orig
#        ####Plot contour####
#        ###Check that function is the same##
#        #levels=np.linspace(0,1,11)
#        plt.figure()
#        #levels=np.linspace(-5,5,11)
#        tanh_levels=np.linspace(-1,1,11)
#        PLayers=vec_get_neuron_values(gX,time_weights)#Layers*(wpad,xpad,time,layersize)
#        #Pred=np.squeeze(PLayers[-1])
#        np_net=np.squeeze(PLayers[-1])
#        tanh_Pred=np.tanh(np_net)
#        if PlotHM:
#            plt.contourf(gX[:,:,0],gX[:,:,1],tanh_Pred,
#                         cmap=plt.cm.bwr_r,vmin=0,vmax=1,levels=tanh_levels)
#            plt.savefig(record_dir+'/prob_heatmap.pdf')
#            #plt.show()
#            #stophere
#            print 'done with heatmap'






        ###Need generic code for ###
    ###Make dict

    #alphabet=symbols('a0:%d'%len(Sig))

    ###Make tree
    ###do postorder traversal to find values 
    ##do preorder traversal to interpret

#    subdict={atom:let for atom,let in zip(Leafs,alphabet)}
#    abstract_Bool_Tree=Bool_Tree.subs(subdict)
#    abstract_MinMax_Tree=MinMax_Tree.subs(subdict)
#    abt=abstract_Bool_Tree


    #Fn sigl-> booltree

#    ###Build Boolean###
#
#        #using sigl for DEBUG
#        #n_sig=len(sigl[0])
#        #depth=len(sigl)
#
#        ###arguments
#        #sigl (sigl or sigl0)
#        #set of weights
#        ###returns
#        #terminal_leafs, Bool_Tree
#
#
#
#        #config
#        #0,1,1 is the config I settled on. The rest of it was messing around.
#        #WhichSig='All'#Sigbar (mode=Numeric)
#        WhichSig=0   #Sigbar0  (mode=Logical)
#
#        OrAndPivot=1
#            #1 only start last layer with init_op, alternate
#            #2 always start each layer with init_op
#        LeafSigMethod=1
#        assert(LeafSigMethod==1)#option 2 not recommended
#            #1 sets new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
#            #2 LeafSig includes all list_unq_sig[hidden_layer]
#
#        #Either is a fine choice
#        init_op=Or
#        #init_op=And
#
#        config_str='WS'+str(WhichSig)+'_OAP'+str(OrAndPivot)+'_LSM'+str(LeafSigMethod)
#        config_str+='_io'+str(init_op)
#
#
#        if WhichSig=='All':
#            list_unq_sig=sigl
#        elif WhichSig==0:
#            list_unq_sig=sigl0
#        else:
#            raise ValueError('WhichSig is ',WhichSig)
#
#        #init
#        n_sig=len(list_unq_sig[0])
#        depth=len(list_unq_sig)
#        Bool_Tree=symbols('0')
#        MinMax_Tree=symbols('0')
#        leaf=Bool_Tree.atoms().pop()#atoms() returns a set
#        leaf_stack={leaf:{
#                          'last_operator'  :init_op,#default start
#                          #'idx_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
#                          'id_mu_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
#                          'id_tu_remain_sig' :np.arange(n_sig),#subset range(len(Sig))
#                          'hidden_layer'   :depth,#0,...,depth-1(first is dummy)
#                          'affine'         :time_weights[-1],
#                         }
#                   }
#        debug_stack=leaf_stack.copy()#shallow
#        terminal_leafs={}#so ctrl-f finds it
#
#
#        print 'Entering While Loop'
#        LOOP=-1
#        while leaf_stack:#until empty
#            LOOP+=1#DEBUG
#            #start loop
#            a_leaf=leaf_stack.keys()[0]
#            leaf_info=leaf_stack.pop(a_leaf)#while nonempty
#            leaf_parent_operator=leaf_info['last_operator']#sympy Or/And
#            #idx_remain_sig=leaf_info['idx_remain_sig']
#            id_mu_remain_sig=leaf_info['id_mu_remain_sig']
#            id_tu_remain_sig=leaf_info['id_tu_remain_sig']
#            ##Increment## hidden_layer -= 1
#            hidden_layer=leaf_info['hidden_layer']-1   #hidden layers iterate 0,..,depth-1
#            gamma,beta=leaf_info['affine']#current linear map
#            #bool_remain_sig=leaf_info['bool_remain_sig']
#            #print 'leaf:',a_leaf
#            #print 'G',gamma,'b',beta,'\n'
#            #print 'DEBUG','IDX_REMAIN_SIG',idx_remain_sig
#            #leaf_sig=list_unq_sig[hidden_layer][idx_remain_sig]#becomes mu_,tu_leaf_sig
#            mu_leaf_sig=list_unq_sig[hidden_layer][id_mu_remain_sig]
#            tu_leaf_sig=list_unq_sig[hidden_layer][id_tu_remain_sig]
#                #leaf_sig=list_unq_sig[hidden_layer]#simply use all
#            #leaf_weights=time_weights[hidden_layer:]#includes weights feeding into layer
#            leaf_weights=[time_weights[hidden_layer],[gamma,beta]]#includes weights feeding into layer
#
#            gamma_pos=(gamma>0).astype('int').ravel()#strict ineq justified
#            gamma_neg=(gamma< 0).astype('int').ravel()
#            pos_sig,pos_inv=np.unique(gamma_pos*mu_leaf_sig,axis=0,return_inverse=True)
#            neg_sig,neg_inv=np.unique(gamma_neg*tu_leaf_sig,axis=0,return_inverse=True)
#            mu_Sig,tu_Sig=pos_sig,neg_sig#db
#            #pos_sig,pos_inv=np.unique(gamma_pos*leaf_sig,axis=0,return_inverse=True)
#            #neg_sig,neg_inv=np.unique(gamma_neg*leaf_sig,axis=0,return_inverse=True)
#            #unq_sig,unq_inv=np.unique(leaf_sig,axis=0,return_inverse=True)
#
#            #split into pos,neg pieces
#            #posrelu_gamma+negrelu_gamma=gamma
#            posrelu_gamma,posrelu_beta=relu(gamma).ravel(),relu(beta)
#            negrelu_gamma,negrelu_beta=-relu(-gamma).ravel(),-relu(-beta)
#            #mu_gamma#mu_beta
#            mu_beta,tu_beta=posrelu_beta,negrelu_beta
#            mu_gamma,mu_inv=np.unique(posrelu_gamma*mu_leaf_sig,axis=0,return_inverse=True)
#            tu_gamma,tu_inv=np.unique(negrelu_gamma*tu_leaf_sig,axis=0,return_inverse=True)
#            ##Careful because entries in mu_leaf_weights not all same shape len
#            ##but compose_affine seems to broadcast correctly anyway
#            mu_leaf_weights=[time_weights[hidden_layer],[mu_gamma[...,None],mu_beta]]
#            tu_leaf_weights=[time_weights[hidden_layer],[tu_gamma[...,None],tu_beta]]
#            new_mu_Gamma,new_mu_Beta=compose_affine(mu_leaf_weights)#n_unqxnlx1,n_unqx1
#            new_tu_Gamma,new_tu_Beta=compose_affine(tu_leaf_weights)
#
#
#            ##How to decide op order##
#            ###OLD v1###
#            if OrAndPivot==1:#old#v1
#                c1_op=leaf_parent_operator
#            ####always Or###
#            elif OrAndPivot==2:#new
#                c1_op=init_op
#            c2_op=swap_bool(c1_op)#always
#            c1_opt=AndOr2MinMax(c1_op)
#            c2_opt=AndOr2MinMax(c2_op)
#
#            if c1_op is Or:
#                #c1_sig,c1_inv=pos_sig,pos_inv#pre v3
#                c1_Gamma,c1_Beta,c1_inv=new_mu_Gamma,new_mu_Beta,mu_inv
#                c2_Gamma,c2_Beta,c2_inv=new_tu_Gamma,new_tu_Beta,tu_inv
#                c1_Sig= pos_sig#keep track of which mu/tau was used
#                c2_Sig=-neg_sig
#                c1_is_mu_=True
#            elif c1_op is And:
#                c1_Gamma,c1_Beta,c1_inv=new_tu_Gamma,new_tu_Beta,tu_inv
#                c2_Gamma,c2_Beta,c2_inv=new_mu_Gamma,new_mu_Beta,mu_inv
#                c1_Sig=-neg_sig
#                c2_Sig= pos_sig
#                c1_is_mu_=False
#            else:
#                raise ValueError('c1_op is ',c1_op)
#
#        #c2_sig,c2_inv=unq_sig,unq_inv
#
##        if c1_op is Or:
##            #c1_sig,c1_inv=pos_sig,pos_inv#pre v3
##            c1_sig,c1_inv=mu_sig,mu_inv
##            c2_sig,c2_inv=tu_sig,tu_inv#new decouple in v3
##        else:
##            #c1_sig,c1_inv=neg_sig,neg_inv
##            c1_sig,c1_inv=tu_sig,tu_inv
##            c2_sig,c2_inv=mu_sig,mu_inv#new decouple in v3
##        #c2_sig,c2_inv=unq_sig,unq_inv
#
#        #new_leaf_weights=rescale_weights(leaf_weights,[c2_sig])#(d+1*2*wtshape, d*n_sigxnl)
#        #new_Gamma,new_Beta=compose_affine(new_leaf_weights)# num_remain x layer_l shape x layer l-1 shape
#
#        sig2name=lambda a: ''.join([str(b) for b in a])
#        c1_sym_list=[]
#        c1_opt_list=[]
#        for i1,c1_gamma in enumerate(c1_Gamma):
#            c1_beta=c1_Beta[i1]
#            c1_sig=c1_Sig[i1]
#            #c1_sym=sig2name(c1_sig)
#
#            ##Decoupled last operator
#            if len(c2_Gamma)>1:
#                last_operator=c2_op
#            else:
#                last_operator=c1_op
#            ##Only relevant when pairing down c2 by c1##
#            #if len(i2_corresp_i1)>1:
#            #    last_operator=c2_op
#            #else:
#            #    last_operator=c1_op
#
#            c2_sym_list=[]
#            for i2,c2_gamma in enumerate(c2_Gamma):
#                c2_beta=c2_Beta[i2]
#                c2_sig=c2_Sig[i2]
#                c2_sym=sig2name(c2_sig)
#                assert(np.dot(np.abs(c1_sig),np.abs(c2_sig))==0.)
#
#                new_ext=sig2name(c1_sig+c2_sig)
#                new_name=new_ext+'.'+str(a_leaf)
#                new_node=symbols(new_name)
#                c2_sym_list.append(new_node)
#
#                ##New Affine##
#                new_gamma=c1_gamma+c2_gamma
#                new_beta=c1_beta+c2_beta
#
#                ####new idx####
#                if LeafSigMethod==1:#v1
#                    #new_idx_remain_sig=idx_remain_sig[c2_inv==i2]
#                    if c1_is_mu_:#mu is first coord
#                        new_id_mu_remain_sig=id_mu_remain_sig[c1_inv==i1]
#                        new_id_tu_remain_sig=id_tu_remain_sig[c2_inv==i2]
#                        mu_sig,tu_sig=c1_sig,c2_sig#for debug
#                    else:
#                        new_id_mu_remain_sig=id_mu_remain_sig[c2_inv==i2]
#                        new_id_tu_remain_sig=id_tu_remain_sig[c1_inv==i1]
#                        mu_sig,tu_sig=c2_sig,c1_sig#for debug
#                elif LeafSigMethod==2:#all idx always
#                    #new_idx_remain_sig=idx_remain_sig
#                    new_id_mu_remain_sig=id_mu_remain_sig
#                    new_id_tu_remain_sig=id_tu_remain_sig
#                new_leaf_info={
#                                'last_operator'  :last_operator,
#                                #'idx_remain_sig' :new_idx_remain_sig,
#                                'id_mu_remain_sig' :new_id_mu_remain_sig,
#                                'id_tu_remain_sig' :new_id_tu_remain_sig,
#                                'affine' :[new_gamma,new_beta],
#                                #'name'           :new_node,
#                              }
#
#                if hidden_layer>0:
#                    new_leaf_info['hidden_layer']=hidden_layer
#                    leaf_stack[new_node]=new_leaf_info
#                else:
#                    #assert(len(new_idx_remain_sig)==1)#corresp to 1 guy
#                    terminal_leafs[new_node]=new_leaf_info
#
#                #debug at top of network
#                if hidden_layer>=1:
#                    debug_stack[new_node]=new_leaf_info
#
#            c1_sym_list.append( c2_op(*c2_sym_list ) )
#            c1_opt_list.append( c2_opt(*c2_sym_list) )
#        new_branch    =c1_op(*c1_sym_list)
#        new_branch_opt=c1_opt(*c1_opt_list)
#        Bool_Tree  =Bool_Tree.subs(a_leaf,new_branch)
#        MinMax_Tree=MinMax_Tree.subs(a_leaf,new_branch_opt)
#    print 'Bool Tree','\n',Bool_Tree
#    print 'MinMax Tree','\n',MinMax_Tree



#rescale_weights(leaf_weights,[c2_sig])
#        map_pos=rescale_weights([W_pos,

#        Q_pos=






