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


from vis_utils import (split_posneg , get_path, get_np_network,
        get_neuron_values, splitL,
                       load_weights,resample_grid,vec_get_neuron_values,
                      get_del_weights)
from calc_maps import get_net_states
from tboard import file2number

from nonlinearities import relu#np version

from vis_utils import Pub_Model_Dirs


def calc_error(log_dir):
    print 'Using model ',log_dir
    record_dir=os.path.join(log_dir,'records')
    id_str=str(file2number(log_dir))

    #config load
    cf_path=os.path.join(log_dir,'params.json')
    print '  Loading params from: ',cf_path
    with open(cf_path,'r') as f:
        load_config=json.load(f)
    dataset=load_config['dataset']
    data_fn=get_toy_data(dataset)#used to sample test data (diff seed)

    #weights load
    all_step=np.load(get_path('step','wwatch',log_dir))
    iter_slice=np.arange(len(all_step))
    dt=10#every 100
    iter_slice=iter_slice[::dt]
    all_weights=load_weights(log_dir)
    del_weights=get_del_weights(all_weights)
    step=all_step[iter_slice]
    step_weights=[[w[iter_slice],b[iter_slice]] for w,b in all_weights]
    step_dweights=[[w[iter_slice],b[iter_slice]] for w,b in del_weights]

    #train data load
    npX=np.load(os.path.join(record_dir,'dataX.npy'))
    npY=np.load(os.path.join(record_dir,'dataY.npy'))
    Xpos,Xneg,Ypos,Yneg=split_posneg(npX,npY)
    m_samples,xdim=npX.shape
    trnX,trnY=npX,npY

    #test data sample
    d_tst_data=data_fn(return_numpy=True,m=1000,seed=999)
    tstX=d_tst_data['input']
    tstY=d_tst_data['label']


    #calc
    trn_PLayers=vec_get_neuron_values(trnX,step_dweights)
    trn_output=trn_PLayers[-1]#timexsamplesx1
    trn_acc=np.mean(trn_output*trnY>=0,axis=-2).ravel()

    tst_PLayers=vec_get_neuron_values(tstX,step_dweights)
    tst_output=tst_PLayers[-1]#timexsamplesx1
    tst_acc=np.mean(tst_output*tstY>=0,axis=-2).ravel()

    df_err=pd.DataFrame({'step':step.ravel(),
                         'trn_err':1-trn_acc,
                         'tst_err':1-tst_acc})
    pth_err=os.path.join(record_dir,'error.txt')
    df_err.to_csv(pth_err,index=False)
    return df_err

if __name__=='__main__':

    #log_dir=Pub_Model_Dirs[0][0]    #[arch#-1][data#-1]
    #log_dir=Pub_Model_Dirs[0][1]
    #log_dir=Pub_Model_Dirs[0][2]
    #log_dir=Pub_Model_Dirs[1][0]
    #log_dir=Pub_Model_Dirs[1][1]
    #log_dir=Pub_Model_Dirs[1][2]
    #log_dir=Pub_Model_Dirs[2][0]
    #log_dir=Pub_Model_Dirs[2][1]
    log_dir=Pub_Model_Dirs[2][2]


    for Arch in Pub_Model_Dirs:
        for log_dir in Arch:
            calc_error(log_dir)










