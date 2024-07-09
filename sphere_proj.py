import matplotlib.pyplot as plt
import numpy as np

#helper
def dot(W,X):
    return np.dot(W.transpose(),X)
det=np.linalg.det


softplus=lambda x:np.log(1+np.exp(x))
softplus0=lambda x:np.log(1+np.exp(x))-np.log(2.)
lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
relu=lambda x:np.maximum(0.,x)
exp=np.exp
elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))


d=3
#rows are dims, cols index samples

def get_sphere_data(n,d):
    W=np.random.rand(d,n)*2.-1#unif[-1,1]
    #X=np.random.rand(d,n)*2.-1#unif[-1,1]
    #normX=np.linalg.norm(X,axis=1,keepdims=True)
    normW=np.linalg.norm(W,axis=1,keepdims=True)
    #X/=normX
    W/=normW
    return W


if __name__=='__main__':
    plt.close('all')
    #np.random.seed(22)

    X=get_sphere_data(d,d)
    W=get_sphere_data(d,d)

    #rho=lrelu
    rho=elu
    #rho=softplus0

    N=100
    out=[]
    for i in range(N):
        X=get_sphere_data(d,d)
        W=get_sphere_data(d,d)
        X[-1]=np.abs(X[-1])#flip for bias

        #X only positive
        #X=np.abs(X)

        #W[-1]=np.abs(W[-1])#experiment

        #both in orthant
        #X=np.abs(X)
        #W=np.abs(W)*np.sign(W[:,0].reshape([-1,1]))

        #Neg dot
        X=np.abs(X)
        W=-np.abs(W)

        Xaff=X/X[-1,:]
        out.append([det(X),det(W),det(dot(W,X)),
                    det(rho(dot(W,X))),
                    det(rho(dot(W,Xaff)))
                   ])

        panic=False
        if panic:
            if np.sign(det(rho(dot(W,X))))!=np.sign(det(rho(dot(W,Xaff)))):
                print 'oh no!'
                break

    Out=np.array(out)
    Sout=np.sign(Out)


print Sout




