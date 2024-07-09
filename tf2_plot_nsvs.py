
from __future__ import absolute_import, division, print_function, unicode_literals
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import os
import pandas as pd
import numpy as np
from utils import prepare_dirs_and_logger,save_config,make_folders
from tf2_config import get_config
from tboard import file2number


from tf2_first import load_frogs_vs_ships, tuple_splits,peak

from sklearn.svm import SVC #precomputed best
from sklearn.svm import LinearSVC#okay
from sklearn.linear_model import SGDClassifier#experimental
import json


if __name__=='__main__':


    trials=[]
    trials.append( [-1,'./logs/Model_0804_141829_frogship_M2plus-1layers']) #1991 .971 .968[*]


    #log_dir='./logs/Model_0804_141829_frogship_M2plus-1layers' #1991 .971 .968[*]
    #log_dir='./logs/Model_0804_151908_frogship_M2plus+0layers'#1217 .976 .964[*]
    #log_dir='./logs/Model_0804_163337_frogship_M2plus+1layers'  #727 .990 .967[*]



    trials.append([-1,'./logs/Model_0804_141829_frogship_M2plus-1layers']) #1991 .971 .968[*]
    trials.append([-1,'./logs/Model_0804_151838_frogship_M2plus-1layers'])  #2446 .970 .970

    #model  0 * Dense(64)
    trials.append([0,'./logs/Model_0803_155517_frogship_debug'          ]) #2368 .971 .963
    trials.append([0,'./logs/Model_0804_151908_frogship_M2plus+0layers' ])#1217 .976 .964[*]

    #model +1 * Dense(64)
    trials.append([1,'./logs/Model_0804_163250_frogship_M2plus+1layers' ]) #998 .988 .969
    trials.append([1,'./logs/Model_0804_163337_frogship_M2plus+1layers' ])  #727 .990 .967[*]


    #model +2 * Dense(64)
    trials.append([2,'./logs/Model_0803_170758_frogship_M2plus2layers' ])  #576 .992 .976[*]
    trials.append([2,'./logs/Model_0804_185041_frogship_M2plus+2layers'])#379 .988 .97


    #model +3 * Dense(64)
    trials.append([3,'./logs/Model_0804_134351_frogship_M2plus3layers' ])#866 .993 .972[*]
    trials.append([3,'./logs/Model_0804_185059_frogship_M2plus+3layers']) #783 .992 .971


    #model +4 * Dense(64)
    trials.append([4,'./logs/Model_0803_164409_frogship_M2plus4layers'  ])  #259 .995 .974[*]
    trials.append([4,'./logs/Model_0804_202228_frogship_M2plus+4layers' ])#143 .991 .966

    #model +5 * Dense(64)
    trials.append([5,'./logs/Model_0804_213951_frogship_M2plus+5layers' ]) #184 .994 .971[*]
    trials.append([5,'./logs/Model_0804_215156_frogship_M2plus+5layers' ]) #358 .990 .971

    plus_layers,log_dirs=zip(*trials)


    results_fps=[ld+'/pac-tensor/results.json' for ld in log_dirs]

    Results=[]
    for ir,results_fp in enumerate(results_fps):
        with open(results_fp, 'r') as fp:
            results=json.load(fp)
            results['players']=plus_layers[ir]
            results['log_dir']=log_dirs[ir]
            Results.append(results)

    Results99=filter(lambda R:R['svc trn acc']>=0.99,Results)

    svc_trn_acc=[R['svc trn acc'] for R in Results99]
    svc_frac_sv=[R['svc frac sv'] for R in Results99]
    agreement=[R['val svc-net agree'] for R in Results99]
    fc_depth=2+np.array([R['players'] for R in Results99])
    LD=[R['log_dir'] for R in Results99]

    fig,ax=plt.subplots()

    ax.plot(fc_depth,svc_frac_sv,'bo')
    ax.set_ylabel('Fraction NSV (s/m)',fontsize=16)
    ax.set_xlabel('Fully-Connected(FC) Depth',fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/FrogShip_NsvsDepth.pdf')



