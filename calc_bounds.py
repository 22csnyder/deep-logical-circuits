import pandas as pd
import numpy as np
from config import get_config
from utils import prepare_dirs_and_logger,save_config
from toydata import get_toy_data
import json

import tensorflow as tf
import math
import numpy as np
import os
import sys
import glob2
from itertools import product
#sys.path.append('/home/chris/Software/workspace/models/slim')
from tqdm import trange
import time
import copy

from ArrayDict import ArrayDict

#temp for debug
from config import get_config
from nonlinearities import name2nonlinearity


from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt


'''
This is a file to calculate various generalization bounds from existing papers
to evaluate them on simple datasets.

It works with numpy arrays

'''


from vis_utils import (split_posneg , get_path, get_np_network,
        get_neuron_values, splitL,
                       load_weights,resample_grid,vec_get_neuron_values,
                      get_del_weights)
from calc_maps import get_net_states
from tboard import file2number

from nonlinearities import relu#np version

from vis_utils import Pub_Model_Dirs


##Matrix norms
norm1inf=lambda W:np.linalg.norm( W ,np.inf) #max(sum(abs(x), axis=1))#p,q=1,inf
normFro=lambda W:np.linalg.norm( W ,'fro') #Frobenius
norm2=lambda W:np.linalg.norm( W , 2 ) #2-norm (largest sing. value)
norm12=lambda W:np.linalg.norm([np.linalg.norm(w,1) for w in W],2)#p,q=1,2


from scipy.spatial import distance
def diameter(npX):
    #npX is (m,xdim)
    D=distance.cdist(npX,npX,'euclidean')
    return D.max()


def calc_norm_bounds(weights,npX,npY):
    '''
    weights should already be subsampled to the desired timesteps
    '''
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)
    #PLayers=vec_get_neuron_values(GridX,weights)
    Net=vec_get_neuron_values(npX,weights)[-1]#(time,m,2)
    YNet=Net*npY
    Margins=YNet[...,1]-YNet[...,0]
    gamma=np.min(Margins,axis=-1)
    gamma5=np.percentile(Margins,5,axis=1)

    m_samples,xdim=npX.shape
    L=len(weights)#number of lin mats
    del_weights=get_del_weights(weights)
    mul_weights,_=zip(*del_weights)
#    mul_weights,_=zip(*weights)
#    mul_weights=list(mul_weights)
#    mul_weights[-1]=mul_weights[-1][...,1]-mul_weights[-1][...,0]
#    mul_weights[-1]=np.expand_dims(mul_weights[-1],-1) #make (Time,layerd,1)

    #_normbound accomulates all the factors related to
    #the bound through the norms of mats
    BartMend2002_normbound=[]
    Neysh2015_normbound=[]
    Bart2017_normbound=[]
    Neysh2017_normbound=[]
    T=mul_weights[0].shape[0]
    for time in range(T):
        Wnorms=[]
        Wnorms2=[]
        Wnorms12=[]
        WnormsFro=[]
        for W in mul_weights:
            ##Matrix norms
            #norm1inf=lambda W:np.linalg.norm( W ,np.inf) #max(sum(abs(x), axis=1))#p,q=1,inf
            #normFro=lambda W:np.linalg.norm( W ,'fro') #Frobenius
            #norm2=lambda W:np.linalg.norm( W , 2 ) #2-norm (largest sing. value)
            #norm12=lambda W:np.linalg.norm([np.linalg.norm(w,1) for w in W],2)#p,q=1,2
            Wnorms.append( norm1inf( W[time].transpose() ) )#Mats do right mul
            Wnorms2.append( norm2( W[time].T ) )
            Wnorms12.append( norm12( W[time].T ))
            WnormsFro.append( normFro( W[time].T ))
        Wnorms=np.array(Wnorms)
        Wnorms2=np.array(Wnorms2)
        Wnorms12=np.array(Wnorms12)
        WnormsFro=np.array(WnormsFro)

        BartMend2002_normbound.append(np.prod(Wnorms))
        Neysh2015_normbound.append( np.prod(  WnormsFro**2 ) )
        B1=np.prod( Wnorms2**2 )
        B2=np.sum( (Wnorms12**2)/(Wnorms2**2) )
        Bart2017_normbound.append( B1*B2 )
        N1=B1
        N2=np.sum(  (WnormsFro**2)/(Wnorms2**2)  )
        Neysh2017_normbound.append(N1*N2)


    BartMend2002_normbound=np.array(BartMend2002_normbound)
    Neysh2015_normbound=np.array(Neysh2015_normbound)
    Bart2017_normbound=np.array(Bart2017_normbound)
    Neysh2017_normbound=np.array(Neysh2017_normbound)

    #BartlettandMendelson2002(weights)
    #Need to check nuissance factors: also see Arora'18
    #(1/gamma^2)\Prod_{i=1}^d Norm{A^i}_{1,inf}*c*root(ln(xdim))/m 
    c=1 #unsure on constant
    BM2002=gamma**(-2.)*BartMend2002_normbound*c*np.log(xdim)**(0.5) / m_samples
    NS2015=gamma**(-2.)*Neysh2015_normbound / m_samples
    BT2017=gamma**(-2.)*Bart2017_normbound / m_samples
    NS2017=gamma**(-2.)*Neysh2017_normbound / m_samples

    return BM2002,NS2015,BT2017,NS2017

def param_counter(arch,xdim=2, outdim=1):
    #arch is a list of the hidden layer widths
    assert(arch[-1]>1)#probably a bug else
    arch=arch+[outdim]
    n_biases=np.sum(arch)
    w_arch=np.array([xdim,]+arch)
    n_mul_weights=np.sum(w_arch[1:]*w_arch[:-1])
    return n_biases+n_mul_weights

def arch_to_VC(xdim,arch):
    #xdim: the input dimension
    #arch: a list of the hidden layer widths
    #returns VC dimension as calculated in 
    #Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks
    ##Theorem 8
    #W weights and U computation units, and assume that the activation function psi is
    #piecewise polynomial of degree at most d with p pieces.
    #2Wlog2(16e * max{U + 1, 2dU} * (1 + p)U) = O(WU log2((1 + d)p)) for the VC-
    #dimension upper bound
    d=1#deg 1 for relu
    p=2#2 pieces forrelu
    U=np.sum(arch)
    #w_arch=np.array([xdim]+arch+ [1,])#outdim assume 1 (binary classification)
    #W=np.sum(w_arch[1:]*w_arch[:-1])
    W=param_counter(arch,xdim)
    #Thm8_Upper = 2*W*np.log2(16*np.e * max(U+1,2*d*U) * (1 + p)**U)#may overflow

    LogTerms=np.log2(16)+np.log2(np.e)+np.log2(max(U+1,2*d*U))+U*np.log2(1 + p)
    Thm8_Upper=2*W*LogTerms

    return Thm8_Upper

def logdir_to_Sig0(log_dir):
    record_dir=os.path.join(log_dir,'records')
    #pth_Sig =record_dir+'/res5000_Sig.npy'
    pth_Sig0=record_dir+'/res5000_Sig0.npy'
    if os.path.exists(pth_Sig0):
        #Sig=np.load(pth_Sig)
        Sig0=np.load(pth_Sig0)
        print 'finished with Sig data'
        return Sig0
    else:
        print 'Sig0 path did not exist:',pth_Sig0

def Sig_to_bound(Sig0,arch,xdim=2):#Usually Sig0,but could pass in Sig for fun

        #time_weights=[[w[trn_iter],b[trn_iter]] for w,b in step_dweights]#d*2*wtshape
        sigl0=np.split(Sig0,#split up into layers again
                      indices_or_sections=np.cumsum(arch)[:-1],
                      axis=-1)
        #Jerome Bound 1995
        #2*K_Jerrum * log2( 8e * D_Jerrum * S_Jerrum )

        #s_l = rank( Sig^l_0 ). Quick proxy: len( Sig^l_0)
        if len(Sig0)==0:
            #Net always positive or always negative
            S_Jerrum=1.#one atom that returns true
            K_Jerrum=1.
            D_Jerrum=1.
            r_Layers=[]
            Jerrum1995=1
        else:
            S_Jerrum=(len(Sig0)**2) #Number of inequalities
            #s_l = [len(Sl) for Sl in sigl0]
            s_l = [np.linalg.matrix_rank(Sl) for Sl in sigl0]
            #m_l = number of neurons that are never off at bdry
            m_l = [ np.sum(np.sum(Sl,axis=0)>0) for Sl in sigl0 ]
            #r_l = upr bnd on dim of subspace that can affect network output
            r_Layers=[1]#r_d+1=1 by convention
            for ss,mm in zip(s_l[::-1],m_l[::-1]):
                r_prev=r_Layers[-1]
                r_Layers.append( np.min([mm,ss*r_prev]) )
            r_Layers.append( xdim + 1 )#r_0=n_0+1
            r_Layers=np.array(r_Layers[::-1])

            K_Layers=r_Layers[1:]*(r_Layers[:-1]-1)

            K_Jerrum=(np.sum(K_Layers).astype('float'))
            D_Jerrum=(np.sum(K_Layers>0).astype('float'))#which layers actually contribute
            Jerrum1995=2 * K_Jerrum * np.log2( 8*np.e * S_Jerrum * D_Jerrum )
        return {'S':S_Jerrum,
                'K':K_Jerrum,
                'D':D_Jerrum,
                'r_Layers':r_Layers,
                'JerrumVC': Jerrum1995 }

if __name__=='__main__':
    #log_dir='./logs/Model_0220_093548_triforce test 4442'
    #log_dir='./logs/Model_0220_102206_triforce_mac_test'#vc competative with all except BM2002
    #log_dir='./logs/Model_0221_101830_R2Clean4442'#VC better than all but BM2002
    #log_dir='./logs/Model_0221_123649_Valley7772'

    #log_dir='./logs/Model_0503_102303_archBig1_R2Clean_25kiter'#VC does well.linear.Big arch
    #log_dir='./logs/Model_0503_104237_archBig1_triforce_25kiter'
    #log_dir='./logs/Model_0503_112240_archBig1_valley_25kiter'#vc8000

    #log_dir='./logs/Model_0503_125104_archSmall1_valley_10kiter' #760
    #log_dir='./logs/Model_0503_120336_archBig3_valley_10kiter'#VC 23627
    #log_dir='./logs/Model_0503_121706_archBig3_valley_10kiter'#VC 22109#akabig2

    #log_dir='./logs/Model_0503_122346_archBig3_R2Clean_10kiter'#VC 18
    #log_dir='./logs/Model_0503_122706_archSmall1_R2Clean_10kiter'#VC 18
    #log_dir='./logs/Model_0503_122850_archBig1_R2Clean_10kiter'#VC 18

    #log_dir='./logs/Model_0503_124214_archSmall1_triforce_10kiter'#VC 252
    #log_dir='./logs/Model_0503_124356_archBig1_triforce_10kiter'  #VC 252
    #log_dir='./logs/Model_0503_124528_archBig3_triforce_10kiter'  #VC 380


    #log_dir=Pub_Model_Dirs[0][0]    #[arch#-1][data#-1]
    #log_dir=Pub_Model_Dirs[0][1]
    #log_dir=Pub_Model_Dirs[0][2]
    #log_dir=Pub_Model_Dirs[1][0]
    #log_dir=Pub_Model_Dirs[1][1]
    #log_dir=Pub_Model_Dirs[1][2]
    #log_dir=Pub_Model_Dirs[2][0]
    #log_dir=Pub_Model_Dirs[2][1]
    #log_dir=Pub_Model_Dirs[2][2]


    #TODO Run these for Gen vs Depth Figure
    #log_dir='./logs/Parm_Model_0509_230036_D3A11'
    #log_dir='./logs/Parm_Model_0509_230931_D3A12'
    #log_dir='./logs/Parm_Model_0509_231404_D3A21'
    log_dir='./logs/Parm_Model_0509_231857_D3A22'



    #arch    |   vcdim
    #Small1  |   6889
    #Big1    |   90353
    #Big3    |   660300  |  1622W 9HL


    print 'Using model ',log_dir
    record_dir=os.path.join(log_dir,'records')
    id_str=str(file2number(log_dir))


    bounds_dir=os.path.join(log_dir,'bounds')
    if not os.path.exists(bounds_dir):
        os.makedirs(bounds_dir)


    all_weights=load_weights(log_dir)
    del_weights=get_del_weights(all_weights)
    #W_weights,b_weights=zip(*weights)

    all_step=np.load(get_path('step','wwatch',log_dir))

    iter_slice=np.arange(len(all_step))
    #dt=10#every 100
    dt=1#every 10
    #dt=70
    #dt=200
    iter_slice=iter_slice[::dt]
    #iter_slice=iter_slice[-1:]#just last entry
    print 'dt=',dt


    #step_weights=[[w[::dt],b[::dt]] for w,b in all_weights]
    #step=all_step[::dt]
    #step_dweights=[[w[::dt],b[::dt]] for w,b in del_weights]
    step=all_step[iter_slice]
    step_weights=[[w[iter_slice],b[iter_slice]] for w,b in all_weights]
    step_dweights=[[w[iter_slice],b[iter_slice]] for w,b in del_weights]
    time_weights=[[w[-1],b[-1]] for w,b in del_weights]#d*2*wtshape#For Sig0calc

    arch=[b.shape[-1] for w,b in step_weights[:-1]]#net architecture

    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)
    m_samples,xdim=npX.shape

    Thm8_Upper=arch_to_VC(xdim,arch) #VC class upper bound
    print 'Thm8_Upper ',Thm8_Upper


    ##Get HighRes Sig data for last time point (res=5000)
    pth_Sig =record_dir+'/res5000_Sig.npy'
    pth_Sig0=record_dir+'/res5000_Sig0.npy'
    if os.path.exists(pth_Sig):
        Sig=np.load(pth_Sig)
        Sig0=np.load(pth_Sig0)
        print 'finished with high res Sig data'
    else:
        print 'Getting high res Sig data'
        gridX=np.load(get_path('gridX','hmwatch',log_dir))
        HighResX=resample_grid(gridX,5000)
        Sig,Centers,Cnts,Idx0=get_net_states(HighResX,time_weights,return_bdry_idx=True)
        Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]
        np.save(record_dir+'/res5000_Sig.npy',Sig)
        np.save(record_dir+'/res5000_Sig0.npy',Sig0)
        print 'finished with high res Sig data'

    #Save HighRes bound
    bnd_info_HighRes=Sig_to_bound(Sig0,arch,xdim=2)
    bnd_info_HighRes['ArchVC']=Thm8_Upper
    pth_finalBnd=os.path.join(bounds_dir,'aftertrain_bounds.txt')
    df_finalBnd=pd.DataFrame.from_dict([bnd_info_HighRes])
    df_finalBnd.to_csv(pth_finalBnd,index=False)
    #df_recover=pd.read_csv(pth_finalBnd)#debug


    #with open(pth_finalBnd,'w') as f:
    #    f.write

    #bnd_final=pd.
    #{'S':S_Jerrum,
    #'K':K_Jerrum,
    #'D':D_Jerrum,
    #'r_Layers':r_Layers,
    #'JerrumVC': Jerrum1995 }


    #Comparision Bounds
    BM2002,NS2015,BT2017,NS2017=calc_norm_bounds(step_weights,npX,npY)


    #Boundary Combinatoric Bounds
    K_Jerrum=[]
    D_Jerrum=[]
    S_Jerrum=[]

    ##Use res=200 for each timepoint
    print 'entering loop'
    #trn_iter=-7#LOOP 1,...,len(step)
    #for trn_iter in range(len(step)):
    for trn_iter in trange(len(step)):
        time_weights=[[w[trn_iter],b[trn_iter]] for w,b in step_dweights]#d*2*wtshape

        gridX=np.load(get_path('gridX','hmwatch',log_dir))
        GridX=resample_grid(gridX)#200
        Sig,Centers,Cnts,Idx0=get_net_states(GridX,time_weights,return_bdry_idx=True)
        Sig0, Cnts0, Centers0 =Sig[Idx0], Cnts[Idx0], Centers[Idx0]


        sigl=np.split(Sig,#split up into layers again
                      indices_or_sections=np.cumsum(arch)[:-1],
                      axis=-1)


        sigl0=np.split(Sig0,#split up into layers again
                      indices_or_sections=np.cumsum(arch)[:-1],
                      axis=-1)


        #Jerome Bound 1995
        #2*K_Jerrum * log2( 8e * D_Jerrum * S_Jerrum )


        #s_l = rank( Sig^l_0 ). Quick proxy: len( Sig^l_0)

        if len(Sig0)==0:
            #Net always positive or always negative
            S_Jerrum.append(1.)#one atom that returns true
            K_Jerrum.append(1.)#
            D_Jerrum.append(1.)


        else:
            S_Jerrum.append(len(Sig0)**2) #Number of inequalities
            #s_l = [len(Sl) for Sl in sigl0]
            s_l = [np.linalg.matrix_rank(Sl) for Sl in sigl0]
            #m_l = number of neurons that are never off at bdry
            m_l = [ np.sum(np.sum(Sl,axis=0)>0) for Sl in sigl0 ]
            #r_l = upr bnd on dim of subspace that can affect network output
            r_Layers=[1]#r_d+1=1 by convention
            for ss,mm in zip(s_l[::-1],m_l[::-1]):
                r_prev=r_Layers[-1]
                r_Layers.append( np.min([mm,ss*r_prev]) )
            r_Layers.append( xdim + 1 )#r_0=n_0+1
            r_Layers=np.array(r_Layers[::-1])

            K_Layers=r_Layers[1:]*(r_Layers[:-1]-1)

            K_Jerrum.append(np.sum(K_Layers).astype('float'))
            D_Jerrum.append(np.sum(K_Layers>0).astype('float'))#which layers actually contribute
    K_Jerrum=np.array(K_Jerrum)
    D_Jerrum=np.array(D_Jerrum)
    S_Jerrum=np.array(S_Jerrum)


    Jerrum1995=2 * K_Jerrum * np.log2( 8*np.e * S_Jerrum * D_Jerrum )
    #np.save(record_dir+'/Jerrum1995',Jerrum1995)
    np.savetxt(record_dir+'/Jerrum1995.txt',Jerrum1995)

    delta=0.05
    VC_Bound=np.sqrt(
        (8*Jerrum1995*np.log(2*np.e*m_samples/Jerrum1995)+8*np.log(4/delta))/m_samples  )


    #3*np.sqrt( np.log(2/delta) / (2*m) )#add to rademacher complexities

    OurBound=np.sqrt(Jerrum1995/m_samples)


    print 'VC dim',Jerrum1995[-1]
    print 'OurBound',OurBound[-1]
    print 'BM2002',BM2002[-1]
    print 'NS2015',NS2015[-1]
    print 'BT2017',BT2017[-1]
    print 'NS2017',NS2017[-1]


    dict_time_bound={'S':S_Jerrum,
                     'K':K_Jerrum,
                     'D':D_Jerrum,
                     'JerrumVC':Jerrum1995,
                     'OurBound':OurBound,
                     'BM2002':BM2002,
                     'NS2015':NS2015,
                     'BT2017':BT2017,
                     'NS2017':NS2017,
                     'step':step.ravel()
                    }
    pth_time_bound=os.path.join(bounds_dir,'duringtrain_bounds_'+'dt='+str(dt)+'.txt')
    pth_time_bound=os.path.join(bounds_dir,'duringtrain_bounds.txt')
    #df_time_bound=pd.DataFrame.from_dict([dict_time_bound])
    df_time_bound=pd.DataFrame.from_dict(dict_time_bound)
    df_time_bound.to_csv(pth_time_bound,index=False)
    #df_time_recover=pd.read_csv(pth_time_bound)#debug

    #{'S':S_Jerrum,
    #'K':K_Jerrum,
    #'D':D_Jerrum,
    #'r_Layers':r_Layers,
    #'JerrumVC': Jerrum1995 }


    if len(iter_slice)>1:
        plt.figure()
        #plt.plot(step,Jerrum1995, label='VC Dimension')
        plt.plot(step,OurBound,label='Our Bound (VC based)')
        plt.plot(step,BM2002,label='BM2002')
        plt.plot(step,NS2015,label='NS2015')
        plt.plot(step,BT2017,label='BT2017')
        plt.plot(step,NS2017,label='NS2017')
        plt.xlabel('Training Steps')

        plt.legend()
        plt.savefig(record_dir+'/BoundComparison.pdf')


        plt.show()

#    #PLayers=vec_get_neuron_values(GridX,weights)
#    Net=vec_get_neuron_values(npX,weights)[-1]#(time,m,2)
#    YNet=Net*npY
#    Margins=YNet[...,1]-YNet[...,0]
#    gamma=np.min(Margins,axis=-1)
#    gamma5=np.percentile(Margins,5,axis=1)














    ###pack rat, old code repo###





    #PosNet=vec_get_neuron_values(Xpos,weights)[-1]#(time,n_pos,2)
    #NegNet=vec_get_neuron_values(Xpos,weights)[-1]
    #PosOut=PosNet[...,1]-PosNet[...,0]#(time,n_pos)
    #NegOut=NegNet[...,1]-NegNet[...,0]
    #margins=PosOut*Ypos.ravel()


    #Pred=PLayers[-1][...,1]-PLayers[-1][...,0]
    #Pfinal=PLayers[-1][...,1]-PLayers[-1][...,0]
    #plt.contour(GridX[:,:,0],GridX[:,:,1],Pfinal[-1],levels=[0.])
    #plt.show()


    #for w,b in rs_weights:
    #    print w.shape, b.shape

    #W1,W2,W3,W4=W_weights
    #b1,b2,b3,b4=b_weights
    #W1f,W2f,W3f,W4f=W1[-1],W2[-1],W3[-1],W4[-1]#final weights
    #b1f,b2f,b3f,b4f=b1[-1],b2[-1],b3[-1],b4[-1]#final weights
    #delW4f=np.reshape(W4[-1,:,1]-W4[-1,:,0], [-1,1])
    #delb4f=b4[-1,1]-b4[-1,0]


    #X=GridX #try
    #w0,b0=weights[0]
    #wpad=len(w0.shape)-2
    #sh_widx=w0.shape[:wpad]
    #xpad=len(X.shape)-1

    #def wrs(W):#also works on biases
    #    new_shape=sh_widx + (1,)*xpad + W.shape[wpad:]
    #    return W.reshape(new_shape)

    #Xshape=(1,)*wpad+X.shape+(1,)#last dim to multiply w
    #rs_X=X.reshape(Xshape)

    #PLayers=[]
    #act=rs_X
    #rs_weights=[[wrs(W),wrs(b)] for W,b in weights]
    #for rs_W,rs_b in rs_weights:
    #    h=np.sum(act*rs_W,axis=-2)+rs_b#widx,xidx,layershape
    #    PLayers.append(h)
    #    act=np.expand_dims( relu( h ), -1 )
    #return PLayers

