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
#sys.path.append('/home/chris/Software/workspace/models/slim')
from tqdm import trange
import time

from ArrayDict import ArrayDict

#temp for debug
from config import get_config
from nonlinearities import name2nonlinearity


from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

'''
This file is to take numpy weights from traiing and generate a network polytope


'''


from tboard import file2number

#def split_posneg(real_val,binary_val):
#    pos_real=real_val[  np.where(npY==+1)[0] ]
#    neg_real=real_val[  np.where(npY==-1)[0] ]
#    pos_bin=binary_val[ np.where(npY==+1)[0] ]
#    neg_bin=binary_val[ np.where(npY==-1)[0] ]
#    return pos_real,neg_real,pos_bin,neg_bin
#
#npX=np.load(os.path.join(record_dir,'dataX.npy'))
#npY=np.load(os.path.join(record_dir,'dataY.npy'))
#
#Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)

def get_path(name,prefix,log_dir):
    listglob=glob2.glob(log_dir+'/records/'+prefix+'*.npy')
    hasname=filter(lambda s:name in s[s.rfind(prefix):], listglob)
    if len(hasname)==1:
        return hasname[0]
    else:
        exactname='_'+name+'.'
        hasname=filter(lambda s:exactname in s[s.rfind(prefix):], listglob)
        if len(hasname)==1:
            return hasname[0]
        else:
            raise ValueError('Multiple matches',[hn[hn.rfind('/'):] for hn in hasname])


###input list [ [mat1,bias1], ]

def minkowski(list_topes):
    #returns verticies correspond to minkowski sum
    lverts=[hsplit(tope) for tope in list_topes]
    new_lverts=[]
    for x in product(*lverts):
        new_lverts.append(sum(x))
    return np.hstack(new_lverts)



#class VTope(object):
#    verticies=None
#    def __init__(self,V):
#        self.verticies=V

def hsplit(mat):
    return np.split(mat,mat.shape[1],axis=1)

##Inefficient class
class LayerTope(object):
    def __init__(self):
        self.pos_topes=[]#list of sets
        self.neg_topes=[]

    def relu(self):
        ##Union pos and neg topes##
        new_pos_topes=[]
        for pt,nt in zip(self.pos_topes,self.neg_topes):
            new_pos_topes.append(np.hstack([pt,nt]))
        self.pos_topes=new_pos_topes

    def init_wmat(self,mat,bias=None):
        if bias is None:
            bias=np.zeros((1,mat.shape[1]),dtype=mat.dtype)
        mat=np.vstack([mat,bias])
        #use columns of weight matrix
        Wcols=np.split(mat,mat.shape[1],axis=1)
        for ai in Wcols:
            #T=VTope(np.hstack([ ai , np.zeros_like(ai)]))
            T=np.hstack([ ai , np.zeros_like(ai)])
            self.pos_topes.append( T )
            self.neg_topes.append( np.zeros_like(ai) )

    def next_layertope(self,mat,bias=None):
        if bias is None:
            #bias=np.array(bias) or np.zeros((1,mat.shape[1]),dtype=mat.dtype)
            bias=np.zeros((1,mat.shape[1]),dtype=mat.dtype)

        NextTope=LayerTope()
        #for w,b in zip(hsplit(mat),np.split(bias,len(bias))):
        for w,b in zip(hsplit(mat),bias.flatten()):
            assert len(w)==len(self.pos_topes)
            plist=[]
            nlist=[]
            for ii in range(len(w)): #mat.shape[0]
                if w[ii]>=0:
                    plist.append(self.pos_topes[ii]*w[ii])
                    nlist.append(self.neg_topes[ii]*w[ii])
                elif w[ii]<0:
                    plist.append(self.neg_topes[ii]*-w[ii])
                    nlist.append(self.pos_topes[ii]*-w[ii])

            #print 'w',w
            #print 'plist',plist
            #print 'nlist',nlist
            #m=minkowski( plist )
            #print 'm',m
            #print 'b',b
            #print b+m

            ptope=np.max([b,0.])+minkowski( plist )
            ntope=np.max([-b,0.])+minkowski( nlist )

            NextTope.pos_topes.append(ptope)
            NextTope.neg_topes.append(ntope)

        return NextTope


def plot_ptope(pos_pts,neg_pts,fig=None,ax=None):
    if not ax:
        fig,ax=plt.subplots()

    all_pts=np.vstack([pos_pts,neg_pts])
    hull=ConvexHull(all_pts)

    ax.plot(pos_pts[:,0],pos_pts[:,1],'bo')
    ax.plot(neg_pts[:,0],neg_pts[:,1],'ro')
    for simplex in hull.simplices:
        ax.plot(all_pts[simplex,0],all_pts[simplex,1],'k-')

    return hull,fig,ax


if __name__=='__main__':
    log_dir='./logs/Model_0101_115945'
    #log_dir='./logs/Model_0130_034237_bulli_test'
    record_dir=os.path.join(log_dir,'records')
    id_str=str(file2number(log_dir))

    W1=np.load(get_path('W1','wwatch',log_dir))
    W2=np.load(get_path('W2','wwatch',log_dir))
    W3=np.load(get_path('W3','wwatch',log_dir))
    W4=np.load(get_path('W4','wwatch',log_dir))
    W1f,W2f,W3f,W4f=W1[-1],W2[-1],W3[-1],W4[-1]#final weights
    delu=np.load(get_path('delu','wwatch',log_dir))
    deluf=delu[-1]
    step=np.load(get_path('step','wwatch',log_dir))
    gridP=np.load(get_path('Prob','hmwatch',log_dir))
    gridX=np.load(get_path('gridX','hmwatch',log_dir))
    Pfinal=gridP[-1,:,:,1]


    def get_layers(time):
        L1=LayerTope()
        L1.init_wmat(W1[time])
        L2=L1.next_layertope(W2[time])
        L2.relu()
        L3=L2.next_layertope(W3[time])
        L3.relu()
        L4=L3.next_layertope(np.reshape(delu[time],[-1,1]))
        return L1,L2,L3,L4


    plt.close('all')

    #L1=LayerTope()
    #L1.init_wmat(W1f)
    #L2=L1.next_layertope(W2f)
    #L3=L2.next_layertope(W3f)
    #L4=L3.next_layertope(np.reshape(deluf,[-1,1]))

    L1,L2,L3,L4=get_layers(time=-1)

#    pos_pts=L3.pos_topes[0][:2].transpose()
#    neg_pts=L3.neg_topes[0][:2].transpose()
#    all_pts=np.vstack([pos_pts,neg_pts])
#    hull=ConvexHull(all_pts)
#
#    fig,ax=plt.subplots()
#    ax.plot(pos_pts[:,0],pos_pts[:,1],'bo')
#    ax.plot(neg_pts[:,0],neg_pts[:,1],'ro')
#    for simplex in hull.simplices:
#        ax.plot(all_pts[simplex,0],all_pts[simplex,1],'k-')

#    pos_pts=L3.pos_topes[0][:2].transpose()
#    neg_pts=L3.neg_topes[0][:2].transpose()

#    pos_pts=L2.pos_topes[0][:2].transpose()
#    neg_pts=L2.neg_topes[0][:2].transpose()

    pos_pts=L4.pos_topes[0][:2].transpose()
    neg_pts=L4.neg_topes[0][:2].transpose()

    hull,fig,ax=plot_ptope(pos_pts,neg_pts)
    n_edges=np.sum(np.sum(hull.simplices>=len(pos_pts),axis=1)==1)
    print 'there were ',n_edges,' mixed edges'

    #hull.equations[np.sum(hull.simplices>=len(pos_pts),axis=1)==1]#These are the upper equations


    plt.show()



