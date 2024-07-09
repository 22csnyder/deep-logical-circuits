import tensorflow as tf
#from datasets import mnist,cifar10,imagenet,flowers
#from model import lenet, load_batch
from config import get_config
from utils import prepare_dirs_and_logger,save_config
import tensorflow as tf
#import tensorflow.contrib.slim as tfslim
#from experiment import Experiment as IB_Experiment
#from mynets import lenet,fcmnist,vgg
#from pacnets import fcmnist as pac_fcmnist
from experiment import Experiment
import pandas as pd
import numpy as np

#from experiment import toy_data,sv2_data,sv1R2
from toydata import get_toy_data
'''
----------------
&&&&& TODO &&&&&
----------------

--------------------------------
&&&&& RESEARCH QUESTIONS &&&&&
--------------------------------
are gradients positive determinant

----------------------
&&&&& WORKING ON &&&&&
----------------------


'''

def main():
    tf.reset_default_graph()

    config,_=get_config()
    prepare_dirs_and_logger(config)
    save_config(config)

    print 'model_dir:',config.model_dir

    data=get_toy_data(config.dataset)()

    experiment=Experiment(config,data)

    return experiment


if __name__ == '__main__':
    exp=main()
    print 'Constructed experiment'
    sess=exp.sess
    #data=exp.data
    config=exp.config
    model=exp.model
    self=model #sorry#notsorry

    #tf.logging.set_verbosity(tf.logging.ERROR)

    if config.is_train:
        exp.train_loop()

    print config.model_dir
    #if config.is_eval:
    #    exp.eval_loop()

    V=tf.global_variables()

    #npv=sess.run(model.v).flatten()
    #npu=sess.run(model.u)
    #du=(npu[:,1]-npu[:,0]).transpose()
    #vu=[(v,u) for v,u in sorted(zip(npv,du))]
    #npdata=sess.run(exp.data)
    #X=npdata['input']
    #Y=npdata['label']


