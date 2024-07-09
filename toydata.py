import tensorflow as tf
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

#Doesnt work in tf2.0beta
#from tensorflow.examples.tutorials.mnist import input_data as mnist_data

datafns={}

def RotMat(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],
                    [np.sin(theta),np.cos(theta)]])
def Ylabels(npos,nneg):
    npY=np.vstack([np.ones((npos,1)),-np.ones((nneg,1))]).astype(np.int64)
    return npY
def sample_circle(N):
    X=[]
    while len(X)<N:
        n=5*N
        X=np.random.uniform(-1.,1.,2*n).reshape(n,2)
        X=X[np.linalg.norm(X,axis=1)<=1]
    return X[:N]
def sample_disk(major,minor,theta,N):
    X=sample_circle(N)
    X2=np.array([[major,minor]])*X
    R=RotMat(theta)
    X3=np.dot(X2,R.T)
    return X3
def make_bullseye(rad1,rad2,rad3,N1,N2):
    #rad1-inner circle
    #rad2,3 are for the outer ring
    inner=rad1*sample_circle(N1)
    outer=[]
    while len(outer)<N2:
        n=10*N2
        outer=rad3*sample_circle(n)
        outer=outer[np.linalg.norm(outer,axis=1)>rad2]
    outer=outer[:N2]
    return inner,outer

def show_scatter(npX,npY):
    Xpos=npX[np.where(npY>0)[0]]
    Xneg=npX[np.where(npY<0)[0]]
    plt.scatter(Xpos[:,0],Xpos[:,1],c='b')
    plt.scatter(Xneg[:,0],Xneg[:,1],c='r')

def datafn_to_scatter(datafn):
    data_dict=datafn(return_numpy=True)
    npX=data_dict['input']
    npY=data_dict['label']
    show_scatter(npX,npY)

#def mnist(return_numpy=False,m=5000,seed=22,subset='train'):
#def mnist(return_numpy=False,m=1000,seed=22,subset='train'):
#def mnist(return_numpy=False,m=20000,seed=22,subset='train'):
def mnist(return_numpy=False,m=55000,seed=22,subset='train'):
    from tensorflow.examples.tutorials.mnist import input_data as mnist_data
    mnist_datasets=mnist_data.read_data_sets('./data/mnist/',reshape=False,validation_size=5000)

    if subset=='train':
        Images=mnist_datasets.train.images
        Labels=mnist_datasets.train.labels
    elif subset=='valid':
        Images=mnist_datasets.validation.images
        Labels=mnist_datasets.validation.labels
        m=len(Labels)#an old bug when m<5000

    #labs=[[5,6],[8,9]]#23139 total
    #labs=[[0,1],[6,7]]
    labs=[[0,1,2,3,4],[5,6,7,8,9]]
    #labs=[[0,1,2],[5,6,7]]
    lab_neg,lab_pos=labs

    print 'Using Labels ',labs

    id_neg=np.where(np.logical_or.reduce([Labels==ln for ln in lab_neg]))[0]
    id_pos=np.where(np.logical_or.reduce([Labels==ln for ln in lab_pos]))[0]
    #id_lab=np.concatenate([id_pos,id_neg])


    Xpos=Images[id_pos]
    Xneg=Images[id_neg]
    npY=Ylabels(len(Xpos),len(Xneg))
    npX=np.vstack([Xpos,Xneg])
    rsLabels=Labels.reshape(npY.shape)
    npLabels=np.vstack([rsLabels[id_pos],rsLabels[id_neg]])

    if subset=='train':
        np.random.seed(seed)
        train_perm=np.random.permutation(len(npX))
        npX=npX[train_perm]
        npY=npY[train_perm]
        npLabels=npLabels[train_perm]

    npX=npX[:m]
    npY=npY[:m]
    npLabels=npLabels[:m]
    npX=npX.reshape([len(npX),-1])

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY,
                'orig label':npLabels,#for logic_inspect
               }
    else:
        data={'input':X,'label':Y}
        return data
datafns['mnist']=mnist


def mnist4(return_numpy=False,m=1000,seed=22,subset='train'):
    '''
    This function is specifically to do surgery on model 151155 (circuitC)
    It learns to classify 4 as False and 5,6,7,8,9 as positive.
    Hopefully this learner can help improve the generalization error of the
    Has to be done very very carefully to make sure all the settings are the
    same because its coded poorly
        orig model info
        ###FigC###
        log_dir='./logs/Model_0514_151155_Mnist1kA3PCA4_LabsAll_50kiter';m=1000
        #1./.781 |Sig0|=55 |Sig|=2432
        ###SORTOF interesting. Similar to 162548 but builds in exception for half the 4's

    run like this;
        %run topenet.py --is_train True --optimizer='adam' --num_iter 50000
        --dataset='mnist4' --pca=4 --arch=3
        --descrip='Mnist1kA3PCA4_Labs4.56789_50kiter' --prefix='Surg'

        python tboard.py --logdir ./logs/Surg_...
    '''
    from tensorflow.examples.tutorials.mnist import input_data as mnist_data
    mnist_datasets=mnist_data.read_data_sets('./data/mnist/',reshape=False,validation_size=5000)

    if subset=='train':
        Images=mnist_datasets.train.images
        Labels=mnist_datasets.train.labels
    elif subset=='valid':
        Images=mnist_datasets.validation.images
        Labels=mnist_datasets.validation.labels
        m=len(Labels)#an old bug when m<5000

    #Have to keep same for now so that random.permutation is same
    labs=[[0,1,2,3,4],[5,6,7,8,9]]
    #Functionally, we are using
    ##labs=[[4],[5,6,7,8,9]]
    lab_neg,lab_pos=labs

    #print 'Using Labels ',labs

    id_neg=np.where(np.logical_or.reduce([Labels==ln for ln in lab_neg]))[0]
    id_pos=np.where(np.logical_or.reduce([Labels==ln for ln in lab_pos]))[0]
    #id_lab=np.concatenate([id_pos,id_neg])

    Xpos=Images[id_pos]
    Xneg=Images[id_neg]
    npY=Ylabels(len(Xpos),len(Xneg))
    npX=np.vstack([Xpos,Xneg])
    rsLabels=Labels.reshape(npY.shape)
    npLabels=np.vstack([rsLabels[id_pos],rsLabels[id_neg]])

    if subset=='train':
        np.random.seed(seed)
        train_perm=np.random.permutation(len(npX))
        npX=npX[train_perm]
        npY=npY[train_perm]
        npLabels=npLabels[train_perm]


    npX=npX[:m]
    npY=npY[:m]
    npLabels=npLabels[:m]

    #new#
    if subset in ['train','valid']:#FOR DEBUG ONLY
        print 'WARNING, VALID SET BEING RESTRICTED'#still works but harder to compare
    #if subset=='train':#dont restrict for validation,want indicies to line up
        id_4exp=np.where(npLabels>=4)[0]#toss labels 0,1,2,3
        npX=npX[id_4exp]
        npY=npY[id_4exp]
        npLabels=npLabels[id_4exp]

    if subset=='train':
        #some class imbalance issues
        pos_subsample=0.3
        print 'Using pos_subsample ',pos_subsample
        id_pos=np.where(npLabels>=5)[0]
        id4=np.where(npLabels==4)[0]
        np.random.seed(seed+1)
        pos_subidx=np.random.choice(id_pos,np.int(pos_subsample*len(id_pos)),replace=False)
        subidx=np.union1d(id_pos,id4)
        npX=npX[subidx]
        npY=npY[subidx]
        npLabels=npLabels[subidx]


    npX=npX.reshape([len(npX),-1])

    print 'for subset ',subset,' excluding labels 0,1,2,3 \n we have ',len(npX),'samples in 4_experiment'

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY,
                'orig label':npLabels,#for logic_inspect
               }
    else:
        data={'input':X,'label':Y,
              'orig label':npLabels,#for debug
             }
        return data
datafns['mnist4']=mnist4



def uniforce(return_numpy=False,m=20,seed=22):
    '''
    a third of the triforce map
    WARN:m is not truely the number of samples, but rather some multiplier
    return is 2*m samples
    '''
    np.random.seed(seed)

    N1=m #20
    #xpos1=sample_disk(2.5,0.75,0.,N1)+np.array([[0.,3.]])
    #xpos2=sample_disk(2.5,0.75,np.pi/3.,N1)+np.array([[2.,-1.]])
    xpos3=sample_disk(2.5,0.75,-np.pi/3.,N1)+np.array([[-2.,-1.]])

    #N2=2*m#40
    N2=m#40 #dont imbalance
    xneg=sample_circle(N2)+np.array([[0.,0.5]])

    #npX=np.vstack([xpos1,xpos2,xpos3,xneg])
    npX=np.vstack([xpos3,xneg])
    #npY=Ylabels(3*N1,N2)
    npY=Ylabels(N1,N2)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['uniforce']=uniforce

def noisy1_uniforce(return_numpy=False,m=20,seed=22):
    '''
    a third of the triforce map
    WARN:m is not truely the number of samples, but rather some multiplier
    return is 2*m samples
    '''
    clean_data=uniforce(return_numpy=True,m=m,seed=seed)
    cleanX=clean_data['input']
    cleanY=clean_data['label']


    #add noise#
    #noise_seed=222
    #np.random.seed(noise_seed)
    #sz_noise=6
    #noiseX=np.random.uniform(cleanX.min(),cleanX.max(),[sz_noise,2])
    #noiseY=np.random.randint(0,2,[sz_noise,1])*2.-1
    #noiseX=np.array([[.5,-1.9],[-0.5,-1.],[-.2,.2],
    #                [0.7,-1.0],[-0.4,-1.6]])
    #noiseY=np.array([-1,-1,1,1,1]).reshape([-1,1])

    noiseX=np.array([[.5,-1.9],[-0.5,-1.],[0.2,-0.5],
                     [-.2,.2],[0.7,-1.0],[-0.4,-1.6],[-1.,1.4],[0.4,1.5]])
    noiseY=np.array([-1,-1,-1,1,1,1,1,1]).reshape([-1,1])

    npX=np.vstack([cleanX,noiseX])
    npY=np.vstack([cleanY,noiseY])

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['noisy1_uniforce']=noisy1_uniforce

def noisy2_uniforce(return_numpy=False,m=20,seed=22):
    '''
    a third of the triforce map
    WARN:m is not truely the number of samples, but rather some multiplier
    return is 2*m samples
    '''
    clean_data=uniforce(return_numpy=True,m=m,seed=seed)
    cleanX=clean_data['input']
    cleanY=clean_data['label']


    #add noise#
    noiseX=np.array([[-2.5,1.3],[-1.2,0.3],[-0.5,-1.6],
                    [0.5,-1.0],[-1.5,1.4]])
    noiseY=np.array([-1,-1,-1,1,1]).reshape([-1,1])

    npX=np.vstack([cleanX,noiseX])
    npY=np.vstack([cleanY,noiseY])

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['noisy2_uniforce']=noisy2_uniforce

def noisy3_uniforce(return_numpy=False,m=20,seed=22):
    '''
    a third of the triforce map
    WARN:m is not truely the number of samples, but rather some multiplier
    return is 2*m samples
    '''
    cleanX,cleanY=uniforce(return_numpy=True,m=40).values()

    npX1,npY1=R2Clean(return_numpy=True).values()
    npX1*=(-1./3.2)
    npX1+=np.array([0.2,-1.5])

    npX2,npY2=R2NoisyV2(return_numpy=True).values()
    npX2*=(-1/4.2)
    npX2+= np.array([-1.5,1.])
    show_scatter(npX2,npY2)

    npX=np.vstack([cleanX,npX1,npX2])
    npY=np.vstack([cleanY,npY1,npY2])
    show_scatter(npX,npY)


    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['noisy3_uniforce']=noisy3_uniforce

#def new_triforce(return_numpy=False,m=20,seed=22):
def new_triforce(return_numpy=False,m=40,seed=22):
    '''
    WARN:m is not truely the number of samples, but rather some multiplier
    return is 5*m samples
    '''
    np.random.seed(seed)

    N1=m #20
    #L1=2.5
    #L1=3.5
    L1=4.
    L2=0.75
    xpos1=sample_disk(L1,L2,0.,N1)+np.array([[0.,3.]])
    xpos2=sample_disk(L1,L2,np.pi/3.,N1)+np.array([[2.,-1.]])
    xpos3=sample_disk(L1,L2,-np.pi/3.,N1)+np.array([[-2.,-1.]])

    N2=2*m#40
    xneg=sample_circle(N2)+np.array([[0.,0.5]])

    npX=np.vstack([xpos1,xpos2,xpos3,xneg])
    #npX=np.vstack([xpos3,xneg])
    npY=Ylabels(3*N1,N2)
    #npY=Ylabels(N1,N2)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['new_triforce']=new_triforce


#def twin_triforce(return_numpy=False,m=20,seed=22):
def twin_triforce(return_numpy=False,m=60,seed=22):
    '''
    WARN:m is not truely the number of samples, but rather some multiplier
    return is 10*m samples
    '''
    np.random.seed(seed)
    second_seed=np.random.randint(seed+1,seed+101)
    npX1,npY1=new_triforce(return_numpy=True,m=m,seed=seed).values()
    npX2,npY2=new_triforce(return_numpy=True,m=m,seed=second_seed).values()
    assert npY1.shape[-1]==1#dicts are technically unordered
    shift=np.array([0.,-4.])
    npX1+=shift
    npX2+=shift
    npX2*=np.array([1.,-1])
    #npX2+=np.array([0.3,0.3])#firsttry. 
    npX2+=np.array([0.7,0.7])
    npX=np.vstack([npX1,npX2])*2.2
    npY=np.vstack([npY1,-npY2])#flip sign second sample
    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['twin_triforce']=twin_triforce


def valley2(return_numpy=False,seed=22):
    np.random.seed(seed)
    N1=15
    N2=40
    r1=1.5
    r2=4.
    r3=9.
    X1=np.vstack(make_bullseye(r1,r2,r3,N1,N2))
    X2=np.vstack(make_bullseye(r1,r2,r3,N1,N2))
    Y1=Ylabels(N1,N2)
    Y2=Ylabels(N1,N2)
    npX=X1
    npY=Y1
    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['valley2']=valley2


def valley_data(return_numpy=False,seed=22):
    np.random.seed(seed)
    N1=15
    N2=40
    r1=1.5
    r2=4.
    r3=9.
    X1=np.vstack(make_bullseye(r1,r2,r3,N1,N2))
    X2=np.vstack(make_bullseye(r1,r2,r3,N1,N2))
    Y1=Ylabels(N1,N2)
    Y2=Ylabels(N1,N2)

    skew=3.
    X1*=np.array([[skew,1.]])
    X2*=np.array([[skew,1.]])

    X1=np.dot(X1,RotMat(-np.pi/4.).T)
    X2=np.dot(X2,RotMat(-np.pi/4.).T)

    #shift=r3+0.25*(r2-r1)
    shift=r3
    X1+=np.array([[-shift,-shift]])
    X2+=np.array([[shift,shift]])

    npX=np.vstack([X1,X2])
    npY=np.vstack([Y1,-Y2])
    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['valley_data']=valley_data

def triforce_data(return_numpy=False):
    np.random.seed(22)
    N1=20
    xpos1=sample_disk(2.5,0.75,0.,N1)+np.array([[0.,3.]])
    xpos2=sample_disk(2.5,0.75,np.pi/3.,N1)+np.array([[2.,-1.]])
    xpos3=sample_disk(2.5,0.75,-np.pi/3.,N1)+np.array([[-2.,-1.]])

    N2=40
    xneg=sample_circle(N2)+np.array([[0.,0.5]])

    npX=np.vstack([xpos1,xpos2,xpos3,xneg])
    npY=Ylabels(3*N1,N2)
    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['triforce_data']=triforce_data

def toy_data():
    xdim=1
    halfN=30
    xpos=np.random.rand(halfN,xdim)+3
    xneg=np.random.rand(halfN,xdim)-3
    npX=np.vstack([xpos,xneg])
    npY=(npX>0.).astype(np.int64)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)

    data={'input':X,'label':Y}
    return data
datafns['toy_data']=toy_data

def checker_quad_data(return_numpy=False):
    xdim=2
    #halfN=30
    halfN=300
    npX_unsort=np.random.rand(2*halfN,xdim)*4-2#unif[-2,2)
    npY_unsort=np.sign( np.prod(npX_unsort,axis=1) ).reshape([-1,1])
    npY_unsort[npY_unsort==0]=1#prob 0
    inds=(-npY_unsort).argsort(axis=0).flatten()#list pos first
    npX=npX_unsort[inds]
    npY=npY_unsort[inds]

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)

    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['checker_quad_data']=checker_quad_data

def quad_data(return_numpy=False):
    xdim=2
    halfN=15
    xpos=np.random.rand(halfN,xdim)+0.5#unif[0.5,1.5)
    xneg=np.random.rand(halfN,xdim)-1.5#unif[-1.5,-.5)
    npX=np.vstack([xpos,xneg])
    npY=np.vstack([np.ones((halfN,1)),-np.ones((halfN,1))]).astype(np.int64)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)

    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['quad_data']=quad_data


def bullseye_data():
    xdim=2
    N=40
    X=np.random.randn(N,xdim)
    med=1.177
    norm=np.linalg.norm(X,axis=1)
    ipos=np.where(norm>=med)[0]
    ineg=np.where(norm<med)[0]

    xpos=X[ipos]
    xneg=X[ineg]
    npX=np.vstack([xpos,xneg])
    npY=np.vstack([np.ones((xpos.shape[0],1)),-np.ones((xneg.shape[0],1))]).astype(np.int64)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)

    data={'input':X,'label':Y}
    return data
datafns['bullseye_data']=bullseye_data


def sv2_data():
    print 'sv2 data'
    xdim=1
    halfN=30
    xpos1=np.random.rand(halfN,xdim)+2
    xpos2=np.random.rand(halfN,xdim)-3
    xneg=np.random.rand(2*halfN,xdim)-0.5
    npX=np.vstack([xpos1,xpos2,xneg])
    npY=np.vstack([np.ones((2*halfN,1)),-np.ones((2*halfN,1))]).astype(np.int64)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)

    data={'input':X,'label':Y}
    return data
datafns['sv2_data']=sv2_data

def sv1R2(return_numpy=False):
    print 'sv3 data'
    xdim=1
    halfN=30
    cov=0.2*np.array([[1.,-.8],[-.8,1.]])
    xpos1=np.random.multivariate_normal([1.,1.]  ,cov, halfN)
    xneg1=np.random.multivariate_normal([-1.,-1.],cov, halfN)
    npX=np.vstack([xpos1,xneg1])
    npY=np.vstack([np.ones((halfN,1)),-np.ones((halfN,1))]).astype(np.int64)

    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)
    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':X,'label':Y}
        return data
datafns['sv1R2']=sv1R2

def R2Noisy(return_numpy=False):
    np.random.seed(27)
    rX=np.random.rand(5,2)*4.-2.#Not used. need for seed to work.
    #ax.scatter(rX[:,0],rX[:,1])
    pX=np.random.multivariate_normal([-1.,0.]  ,[[0.3,0],[0,1.5]], 8)
    nX=np.random.multivariate_normal([+1.,0.]  ,[[0.3,0],[0,1.5]], 8)
    rX=np.array([[0.1,-0.5],
                 [0.9,-1.3],
                 [-1.4,2.8],
                 [0.3,2.5],
                 [1.7,2.0]])
    rY=np.array([1.,-1,-1.,1.,-1]).reshape([-1,1])
    npX=np.vstack([pX,nX,rX])
    npY=np.vstack([np.ones((pX.shape[0],1)),-np.ones((nX.shape[0],1)),rY]).astype(np.int64)

    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':tf.constant(npX,dtype=tf.float32),
              'label':tf.constant(npY)}
        return data
datafns['R2Noisy']=R2Noisy

def R2Clean(return_numpy=False):
    np.random.seed(27)
    rX=np.random.rand(5,2)*4.-2.#Not used. need for seed to work.
    #ax.scatter(rX[:,0],rX[:,1])
    pX=np.random.multivariate_normal([-1.,0.]  ,[[0.3,0],[0,1.5]], 8)
    nX=np.random.multivariate_normal([+1.,0.]  ,[[0.3,0],[0,1.5]], 8)
    #rX=np.array([[0.1,-0.5],
    #             [0.9,-1.3],
    #             [-1.4,2.8],
    #             [0.3,2.5],
    #             [1.7,2.0]])
    #rY=np.array([1.,-1,-1.,1.,-1]).reshape([-1,1])
    npX=np.vstack([pX,nX])
    npY=np.vstack([np.ones((pX.shape[0],1)),-np.ones((nX.shape[0],1))]).astype(np.int64)

    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':tf.constant(npX,dtype=tf.float32),
              'label':tf.constant(npY)}
        return data
datafns['R2Clean']=R2Clean

def R2Noisy28(return_numpy=False):
    np.random.seed(27)
    rX=np.random.rand(5,2)*4.-2.#Not used. need for seed to work.
    #ax.scatter(rX[:,0],rX[:,1])
    pX=np.random.multivariate_normal([-1.,0.]  ,[[0.3,0],[0,1.5]], 8)
    nX=np.random.multivariate_normal([+1.,0.]  ,[[0.3,0],[0,1.5]], 8)

    np.random.seed(28)
    rn=4
    rX1=np.random.uniform(-2,2,rn)
    rX2=np.random.uniform(-1,3,rn)
    rX=np.c_[rX1,rX2]
    rY=np.array([+1.,+1.,-1.,-1.]).reshape([-1,1])

    npX=np.vstack([pX,nX,rX])
    npY=np.vstack([np.ones((pX.shape[0],1)),-np.ones((nX.shape[0],1)),rY]).astype(np.int64)

    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':tf.constant(npX,dtype=tf.float32),
              'label':tf.constant(npY)}
        return data
datafns['R2Noisy28']=R2Noisy28



def R2NoisyV2(return_numpy=False):
    np.random.seed(22)
    pX=np.random.multivariate_normal([-1.5,0.]  ,[[0.3,0],[0,1.5]], 8)
    nX=np.random.multivariate_normal([+1.5,0.]  ,[[0.3,0],[0,1.5]], 8)
    rX=np.array([[0.1,-0.5],
                 [0.9,-1.3],
                 [-1.4,2.8],
                 [-0.3,2.2],
                 [1.7,2.5]])
    #rY=np.array([1.,-1,-1.,1.,-1]).reshape([-1,1])
    rY=np.array([-1.,-1,-1.,+1.,1]).reshape([-1,1])#corners
    #rY=np.array([+1.,+1,1.,1.,1]).reshape([-1,1])
    #rY=np.array([-1.,-1,-1.,-1.,-1]).reshape([-1,1])
    npX=np.vstack([pX,nX,rX])
    npY=np.vstack([np.ones((pX.shape[0],1)),-np.ones((nX.shape[0],1)),rY]).astype(np.int64)

    if return_numpy:
        return {'input':npX,'label':npY}
    else:
        data={'input':tf.constant(npX,dtype=tf.float32),
              'label':tf.constant(npY)}
        return data
datafns['R2NoisyV2']=R2NoisyV2


##############################################
##########DEFINE data1,data2,data3############

datafns['data1']=uniforce
datafns['data2']=new_triforce
datafns['data3']=twin_triforce

##########DEFINE data1,data2,data3############
##############################################



def get_toy_data(name):
    if name in datafns.keys():
        return datafns[name]
    else:
        raise ValueError('expected dataset string but got ',type(config.dataset))



