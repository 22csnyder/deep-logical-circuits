import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


softplus=lambda x:np.log(1+np.exp(x))
lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
relu=lambda x:np.maximum(0.,x)
exp=np.exp
#elu=lambda x: np.max(0.,x)+np.int(x<0)*(0.2)*(np.exp(x)-1.)
#elu=lambda x: np.maximum(np.exp(x)-1.,x)#####WARN
elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
tanh=np.tanh
sigmoid=lambda x: 1./(1.+np.exp(-x))


def tf_elu(x):
    #elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
    return x*(0.5)*(tf.sign(x)+1) + (tf.exp(x)-1.)*(0.5)*(1-tf.sign(x))
def tf_lrelu(x,leak=0.1,name='lrelu'):
    with tf.variable_scope(name):
        f1=0.5 * (1+leak)
        f2=0.5 * (1-leak)
        return f1*x + f2*tf.abs(x)

def name2nonlinearity(name):
    #name should be a string
    if name=='relu':
        fn=tf.nn.relu
    elif name=='lrelu':
        fn=tf_lrelu
    elif name=='tanh':
        fn=tf.nn.tanh
    elif name=='abs':
        fn=tf.abs
    elif name=='elu':
        fn=tf_elu
    elif isinstance(name,str):
        raise ValueError('name was not recognized as nonlinearity:',name)

    elif callable(name):#already fn was passed 
        print 'warn: function was passed instead of name',name
        fn=name

    else:
        raise ValueError('expected string or function but got',type(name))

    return fn

def plt_test(rho,title=None):
    X=np.mgrid[-3:3:.1]
    Y=rho(X)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(X,Y)
    if title:
        ax.set_title(title)
    plt.show(block=False)



if __name__=='__main__':
#####REDO because ELU wasn't implemented correctly#####
    plt.close('all')
    #Test implementation
#    plt_test(elu,'elu')
#    plt_test(lrelu,'lrelu')
#    plt_test(relu,'relu')
#    plt_test(exp,'exp')
#    plt_test(softplus,'softplus')



#    #Test wronskian
#    #rho=softplus
#    rho=lrelu
#
#
#    w1=-3.
#    w2=-1.
#    #w1=2.
#    #w1=-1.
#    #w2=3.
#
#    X=np.mgrid[-5:5:.1]
#    P12=rho(w1*X)
#    P22=rho(w2*X)
#    #x=1.
#    #x=3.
#    x=2.
#    #x=-2.
#    P11=rho(w1*x)
#    P21=rho(w2*x)
#    det=P11*P22-P21*P12
#    table=np.vstack([X,det])[:,::5]
#    print 'table'
#    print table



#    ###W3
#    w1=-1.
#    w2=1.
#    w3=2.
#    #w1=2.
#    #w1=-1.
#    #w2=3.
#
#    X=np.mgrid[-5:5:.1]
#    P12=rho(w1*X)
#    P22=rho(w2*X)
#    #x=1.
#    #x=3.
#    x1=-1.
#    x2=2.
#    #x3=3.5
#    #x=-2.
#
#    P11=rho(w1*np.tile(x1,len(X)))
#    P21=rho(w2*np.tile(x1,len(X)))
#    P31=rho(w3*np.tile(x1,len(X)))
#
#    P12=rho(w1*np.tile(x2,len(X)))
#    P22=rho(w2*np.tile(x2,len(X)))
#    P32=rho(w3*np.tile(x2,len(X)))
#
#    P13=rho(w1*X)
#    P23=rho(w2*X)
#    P33=rho(w3*X)
#
#    R1=np.stack([P11,P21,P31],axis=-1)
#    R2=np.stack([P12,P22,P32],axis=-1)
#    R3=np.stack([P13,P23,P33],axis=-1)
#    W=np.stack([R1,R2,R3]    ,axis=-1)
#    det=np.linalg.det(W)
#
#    table=np.vstack([X,det])#[:,::5]
#
#    print table[:,np.where(X<=x1)[0]][:,::5]
#    print table[:,np.where((x1<=X) * (X<=x2))[0]][:,::5]
#    print table[:,np.where(X>=x2)[0]][:,::5]
#    #print table[:,X<=x2]
#
#
#    #print 'table'
#    #print table

    #Counter example for Softplus:
    #w1=-3.
    #w2=-2.
    #w3=0.
    #w4=2.0
    #x1=-1.
    #x2=0.
    #x3=3.5

    ###W4
    rho=softplus
    #rho=relu
    #rho=lrelu
    #rho=elu
    #rho=tanh
    #rho=exp
    w1=-3.
    w2=-2.
    w3=0.
    w4=2.0
    #w1=2.
    #w1=-1.
    #w2=3.

    #x=1.
    #x=3.
    x1=0.5
    x2=1.
    x3=3.5
    #x=-2.

    #X=np.mgrid[-5:5:.1]
    X=np.mgrid[-5:5:.01]
    P12=rho(w1*X)
    P22=rho(w2*X)

    P11=rho(w1*np.tile(x1,len(X)))
    P21=rho(w2*np.tile(x1,len(X)))
    P31=rho(w3*np.tile(x1,len(X)))
    P41=rho(w4*np.tile(x1,len(X)))

    P12=rho(w1*np.tile(x2,len(X)))
    P22=rho(w2*np.tile(x2,len(X)))
    P32=rho(w3*np.tile(x2,len(X)))
    P42=rho(w4*np.tile(x2,len(X)))

    P13=rho(w1*np.tile(x3,len(X)))
    P23=rho(w2*np.tile(x3,len(X)))
    P33=rho(w3*np.tile(x3,len(X)))
    P43=rho(w4*np.tile(x3,len(X)))

    P14=rho(w1*X)
    P24=rho(w2*X)
    P34=rho(w3*X)
    P44=rho(w4*X)

    R1=np.stack([P11,P21,P31,P41],axis=-1)
    R2=np.stack([P12,P22,P32,P42],axis=-1)
    R3=np.stack([P13,P23,P33,P43],axis=-1)
    R4=np.stack([P14,P24,P34,P44],axis=-1)
    W=np.stack([R1,R2,R3,R4]    ,axis=-1)
    det=np.linalg.det(W)

    table=np.vstack([X,det])#[:,::5]

    print table[:,np.where(X<=x1)[0]][:,::5],'\n'
    print table[:,np.where((x1<=X) * (X<=x2))[0]][:,::5],'\n'
    print table[:,np.where((X>=x2)*(X<=x3))[0]][:,::5],'\n'
    print table[:,np.where(X>=x3)[0]][:,::5],'\n'

    #print table[:,np.where(X<=x1)[0]],'\n'
    #print table[:,np.where((x1<=X) * (X<=x2))[0]],'\n'
    #print table[:,np.where((X>=x2)*(X<=x3))[0]],'\n'
    #print table[:,np.where(X>=x3)[0]],'\n'


