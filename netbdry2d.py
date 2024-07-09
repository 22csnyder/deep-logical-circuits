import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
#from scipy.misc import comb
from scipy.special import comb
import os
import glob2
from datetime import datetime

#from nonlinearities import softplus,lrelu,relu,exp,elu,tanh

softplus=lambda x:np.log(1+np.exp(x))
lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
relu=lambda x:np.maximum(0.,x)
exp=np.exp
#elu=lambda x: np.max(0.,x)+np.int(x<0)*(0.2)*(np.exp(x)-1.)
#elu=lambda x: np.maximum(np.exp(x)-1.,x)#####WARN
elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
tanh=np.tanh



def setup_net(W,b,u,rho):
    def net(x):
        #x should be batch x n
        mm=np.matmul(x,W.transpose())
        h=mm+b
        act=rho(h)
        out=np.dot(act,u)
        return out
    return net


def net2eval( netcall ):
    delta=0.05
    R=5.
    dx=dy=np.arange(-R,R,delta)
    dx31,dx32=np.meshgrid(dx,dy)
    dx3=np.stack([dx31,dx32],axis=2)
    rs_dx3=dx3.reshape([-1,n])
    eval=netcall(rs_dx3)
    evalgrid=netcall(rs_dx3).reshape([dx.shape[0],dy.shape[0]])
    return dx,dy,evalgrid

def contour_plot(dx,dy,data_grid,fig=None,ax=None):
    if not ax:
        fig,ax=plt.subplots()

    levels=np.arange(-2.,2,0.4)
    CS=ax.contourf(dx,dy,data_grid,levels)
    fig.colorbar(CS)
    CS2=ax.contour(CS,levels=[0.],colors='k')
    return fig,ax

def weight_plot(W,u,fig=None,ax=None):
    if not ax:
        fig,ax=plt.subplots()

    posidx=np.where(u>0)[0]
    negidx=np.where(u<=0)[0]
    ms0=5
    for i in posidx:
        msi=10*np.abs(u[i])+ms0
        ax.plot(W[i,0],W[i,1],markersize=msi,marker='$w$',c='b',)
               #markeredgecolor='k',markeredgewidth=.01*msi)
    for i in negidx:
        msi=10*np.abs(u[i])+ms0
        ax.plot(W[i,0],W[i,1],markersize=10*np.abs(u[i])+ms0,marker='$w$',c='r')

    return fig,ax

    plt.show(block=False)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

if __name__=='__main__':
    plt.close('all')

    #m=6
    k=5#num neurons
    n=2#dim input
    #rho=exp;rhostr='exp'
    #rho=elu;rhostr='elu'
    rho=lrelu;rhostr='lrelu'
    #rho=relu;rhostr='relu'


    bigfoldername='./templogs' #noted in gitignore
    if not os.path.exists(bigfoldername):
        os.makedirs(bigfoldername)
    info=(rhostr+'_'+'k'+str(k)+'n'+str(n))
    run_name = "{}_{}".format(get_time(),info)
    foldername=os.path.join(bigfoldername,run_name)
    os.makedirs(foldername)

    fig_name='dbdry_net{0}.png'


    ##Generate Many example plots##
    Nii=40
    for ii in range(Nii):
        #plt.close('all')

        W=np.random.rand(k,n)*2.-1#unif[-1,1]
        W*=2#unif[-2,2]
        b=np.ones((1,k),dtype=np.float)
        u=np.random.rand(k)*2.-1#unif[-1,1]

        netcall=setup_net(W,b,u,rho)
        dx,dy,neteval=net2eval(netcall)
        fig,ax=contour_plot(dx,dy,neteval)
        weight_plot(W,u,fig,ax)

        fname=os.path.join(foldername,fig_name.format(ii))
        fig.savefig(fname)



#    #Try scaling u and compare
#    W=np.random.rand(k,n)*2.-1#unif[-1,1]
#    b=np.ones((1,k),dtype=np.float)
#    u=np.random.rand(k)*2.-1#unif[-1,1]
#    netcall=setup_net(W,b,u,rho)
#    dx,dy,neteval=net2eval(netcall)
#    fig,ax=contour_plot(dx,dy,neteval)
#    weight_plot(W,u,fig,ax)
#    plt.title('u orig')
#
#    #W=np.random.rand(k,n)*2.-1#unif[-1,1]
#    #b=np.ones((1,k),dtype=np.float)
#    #u=3*u
#    W*=2
#    netcall=setup_net(W,b,u,rho)
#    dx,dy,neteval=net2eval(netcall)
#    fig2,ax2=contour_plot(dx,dy,neteval)
#    weight_plot(W,u,fig2,ax2)
#    W*=2
#    netcall=setup_net(W,b,u,rho)
#    dx,dy,neteval=net2eval(netcall)
#    fig3,ax3=contour_plot(dx,dy,neteval)
#    weight_plot(W,u,fig3,ax3)
#    plt.show()


