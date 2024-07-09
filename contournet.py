import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
import os
import sys


delta=0.25


###This is how you generate heatmaps###
if __name__=='__main__':
    x1=x2=np.arange(-2.5,2.51,delta)
    #X1,X2=np.meshgrid(x,y)
    gridX=np.stack(np.meshgrid(x1,x2),axis=-1)#LxLx2
    gX=np.reshape(gridX,[-1,2])
    gP=sess.run(model.prob,feed_dict={model.phx:gX})
    gridP=gP.reshape(gridX.shape)

    fig,ax=plt.subplots()
    ax.contourf(gridX[:,:,0],gridX[:,:,1],gridP[:,:,1],cmap=plt.cm.bwr_r)

