import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
#from scipy.misc import comb
from scipy.special import comb


#from nonlinearities import softplus,lrelu,relu,exp,elu,tanh

softplus=lambda x:np.log(1+np.exp(x))
lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
relu=lambda x:np.maximum(0.,x)
exp=np.exp
#elu=lambda x: np.max(0.,x)+np.int(x<0)*(0.2)*(np.exp(x)-1.)
#elu=lambda x: np.maximum(np.exp(x)-1.,x)#####WARN
elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
tanh=np.tanh



if __name__=='__main__':
    plt.close('all')

    m=6
    k=2
    n=3
    rho=exp
    #rho=elu
    #rho=lrelu

    W=np.random.rand(k,n)*2.-1#unif[-1,1]
    #W=np.vstack([ [0.,0.],
    #              [1.,0.],
    #              [0.,1.]]).transpose()
    #W=np.vstack([ [ 0., 0.], #--> two parallel lines
    #              [ 1., 0.],
    #              [-1., 0.]]).transpose()+0.5*(np.random.rand(2,1)*2.-1)

    x1=np.random.rand(2,1)*2.-1
    x2=np.random.rand(2,1)*2.-1



    delta=0.05
    dx=dy=np.arange(-3.,3.01,delta)
    dx31,dx32=np.meshgrid(dx,dy)
    dx3=np.stack([dx31,dx32],axis=2)

    tx1=np.tile(x1.reshape(1,1,2), [dx3.shape[0],dx3.shape[1],1])
    tx2=np.tile(x2.reshape(1,1,2), [dx3.shape[0],dx3.shape[1],1])

    dX=np.stack([tx1,tx2,dx3],axis=-1)
    e_dX=np.expand_dims(dX,axis=3)
    e_W=W[None,None,:,:,None]

    dWtX=np.sum(e_dX*e_W,axis=2)
    dA=rho(dWtX)

    Dets=np.linalg.det(dA)
    levels=np.arange(-2.,2,0.4)
    plt.contourf(dx,dy,Dets,levels)
    plt.colorbar()

    plt.plot([x1[0],x2[0]],[x1[1],x2[1]],marker='o',markersize=12)
    plt.plot(W[0],W[1],marker='$w$',markersize=12,linestyle='none')

    plt.show()




    #rs_dx3=dx3.reshape(dx3.shape[0],dx3.shape[1],1,1,2)
    #dX=x1[None,None,:,None]*x2[None,None,None,:]*rs_dx3

    #dX=x1[None,:,None,None]*x2[None,:,None,None]*dx3[:,None,None]

