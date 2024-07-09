import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
#from scipy.misc import comb
from scipy.special import comb


from nonlinearities import softplus,lrelu,relu,exp,elu,tanh
softplus=lambda x:np.log(1+np.exp(x))
lrelu=lambda x:np.minimum(0.2*x,0)+np.maximum(0.,x)
relu=lambda x:np.maximum(0.,x)
exp=np.exp
#elu=lambda x: np.max(0.,x)+np.int(x<0)*(0.2)*(np.exp(x)-1.)
#elu=lambda x: np.maximum(np.exp(x)-1.,x)#####WARN
elu=lambda x: x*(0.5)*(np.sign(x)+1) + (np.exp(x)-1.)*(0.5)*(1-np.sign(x))#Correct
tanh=np.tanh


'''
The goal here is to analyze the interaction of the polytope w1,..,wk  and
x1,..,xk in producing A=[rho(<wi,xj>)]i,j cols [a1,..,am]
'''



#Q1. when do x1,..,xk and a1,..,ak share a common covector regardless of wi


def find_signature(X):
    NullX=linalg.null_space(X)
    m,g=NullX.shape
    n_coupons=float(2**min(g,m))
    C=3000 #safety factor
    #C=30000 #safety factor
    N=np.ceil(C*n_coupons*np.log(n_coupons)).astype(np.int)#how many we need to be pretty sure

    mean=np.zeros(g,dtype=np.float)
    ball=np.random.multivariate_normal(mean,cov=np.eye(g),size=N)#Nxg

    vecs=np.dot(ball,np.transpose(NullX))#Nxm
    #vecs=np.dot(NullX,np.transpose(ball))

    signed_vecs=np.unique(np.sign(vecs),axis=0)#n_unique x m

    return signed_vecs


def dim2nquads(n,k):
    #k-dim subspace in n-space
    #adapted from stackexchange (which I think was wrong)
    #2*(sum_d=0^k choose(n-1,d-1))
    l=[comb(n-1,d) for d in range(k)]
    return 2*sum(l)

#def enumerate_unique_codes(df_codes):
#    df2=df_codes.drop_duplicates()
#    df2 = df2.reset_index(drop = True).reset_index()
#    df2=df2.rename(columns = {'index':'ID'})
#    return df2

if __name__=='__main__':
    plt.close('all')

    #Config
    #m=5 #6
    m=7
    k=2
    n=3
    rho=exp

    #I think the m5n3k2 that worked had flats
    #np.random.seed(22)#m5
    #np.random.seed(24)#m5 no circuits
    #np.random.seed(26)#m5 no circuits#interesting conf
    #np.random.seed(27)#m5 no circuits#interesting conf

    #np.random.seed(30)#m6 1 circuit preserved (4,2)#Oops. No circuit preserved
    #np.random.seed(36)#m6 1 circuit (3,3)#This time (4,2) preserved#no good circuits
    #np.random.seed(38)#m6 1 circuit (3,3)
    #np.random.seed(39)#m6 0 circuits !
        #Also though it's interesting that no points are in neg orthant

    #m7 series
    #np.random.seed(700)#fig_701
    np.random.seed(704)

    X=np.random.rand(k,m)*2.-1#unif[-1,1]
    W=np.random.rand(k,n)*2.-1#unif[-1,1]

    #DEBUG
    #X=np.vstack([
    #    np.sin(np.arange(m)*2*np.pi/m),
    #    np.cos(np.arange(m)*2*np.pi/m)
    #            ])

    #X=np.vstack([ #6 circuits
    #    np.sin(np.arange(m)*2*np.pi/(3.*m)),
    #    np.cos(np.arange(m)*2*np.pi/(3.*m))
    #            ])
    #X=np.vstack([ #3 circuits
    #    np.sin(np.arange(m)*2*np.pi/(2.*m)),
    #    np.cos(np.arange(m)*2*np.pi/(2.*m))
    #            ])


    #X+=2
    #X-=1


    WtX=np.dot(np.transpose(W),X)
    A=rho(WtX)

    NullX=linalg.null_space(X)


    circX=find_signature(X)
    circA=find_signature(A)


    W2=np.random.rand(k,n)*2.-1#unif[-1,1]
    WtX2=np.dot(np.transpose(W2),X)
    A2=rho(WtX2)
    circA2=find_signature(A2)

    pdX =pd.DataFrame(circX).apply(tuple,1)
    pdA =pd.DataFrame(circA).apply(tuple,1)
    pdA2=pd.DataFrame(circA2).apply(tuple,1)

    L=1000
    ws=[]
    pdas=[]
    isins=[]
    np.random.seed(None)
    for i in range(L):
        W=np.random.rand(k,n)*2.-1#unif[-1,1]
        WtX=np.dot(np.transpose(W),X)
        A=rho(WtX)
        circA=find_signature(A)
        pdA =pd.DataFrame(circA).apply(tuple,1)
        boolinA=pdX.isin(pdA.values)

        ws.append(W)
        pdas.append(pdA)
        isins.append(boolinA)


    pdTotal=pd.concat(isins,axis=1)
    preserved=pdTotal.all(axis=1)

    print np.sum(preserved),' Good circuits:'
    print pdX[preserved]

    plt.scatter(X[0,:],X[1,:],c=range(X.shape[1]))
    plt.colorbar()


    plt.show()

    #pdX10=pdX.loc[:10]
    #pdX515=pdX.loc[5:15]
#
#    pdX.isin(pdA.values)






