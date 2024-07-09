from __future__ import print_function
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import shutil
import sys
import math
import json
import logging
import numpy as np
#from PIL import Image
from datetime import datetime
import pandas as pd
import logging


def bool_diff_sign(w1,w2):
    n_diff=tf.abs(tf.sign(w1)-tf.sign(w2))/2.
    return n_diff

def copy_name(w,name='copy'):
    name='_'+name
    #cur_scope=tf.contrib.framework.get_name_scope()#yuck
    end=w.name.rfind(':')
    return w.name[:end]+name

def copy_weights(weights,name=None):
    #{'L1':[W1,b1],'L2':[W2,b2]}
    initial_vals={}
    with tf.name_scope(None):
        for key,wt_pair in weights.items():
            w,b=wt_pair
            w_init=tf.Variable(w.initialized_value(),name=copy_name(w,name))
            b_init=tf.Variable(b.initialized_value(),name=copy_name(b,name))
            initial_vals[key]=[w_init,b_init]
    return initial_vals

def tally_labels(attr):
    '''
    inputs
    attr: dataframe of label attributes

    returns
    df2 : dataframe with each row a unique label combination that occurs in the
    dataset. The index is a unique 'ID' that corresp to that label combination
    real_pdf: dataframe with index='ID' and value is the probability of that
    label combination
    '''
    df2=attr.drop_duplicates()
    df2 = df2.reset_index(drop = True).reset_index()
    df2=df2.rename(columns = {'index':'ID'})
    real_data_id=pd.merge(attr,df2)
    real_counts = pd.value_counts(real_data_id['ID'])
    real_pdf=real_counts/len(attr)
    return df2,real_pdf



def make_folders(folder_list):
    for path in folder_list:
        if not os.path.exists(path):
            os.makedirs(path)



def old_get_model_dir(dataset,logs,descrip=''):
    model_name = "{}_{}".format(dataset, get_time())
    model_dir = os.path.join(logs,model_name)
    if descrip:
        model_dir+='_'+descrip
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def get_model_dir(config):
    #Each run gets a new model_name and model_dir
    #config.model_name = "{}_{}".format(config.dataset, get_time())
    config.model_name = "{}_{}".format(config.prefix, get_time())
    if config.descrip:
        config.model_name+='_'+config.descrip
    model_dir = os.path.join(config.log_dir,config.model_name)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def prepare_dirs_and_logger(config):
    config.model_dir=get_model_dir(config)

    config.record_dir=os.path.join(config.model_dir,'records')
    config.log_code_dir=os.path.join(config.model_dir,'code')
    #if not config.load_path:
    if config.load_path is not config.model_dir:
        for path in [config.log_dir, config.model_dir,
                     config.log_code_dir,config.record_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

        #Copy python code in directory into model_dir/code for future reference:
        code_dir=os.path.dirname(os.path.realpath(sys.argv[0]))
        model_files = [f for f in listdir(code_dir) if isfile(join(code_dir, f))]
        for f in model_files:
            if f.endswith('.py'):
                shutil.copy2(f,config.log_code_dir)

def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type=='GPU']

def distribute_input_data(data_loader,num_gpu):
    '''
    data_loader is a dictionary of tensors that are fed into our model

    This function takes that dictionary of n*batch_size dimension tensors
    and breaks it up into n dictionaries with the same key of tensors with
    dimension batch_size. One is given to each gpu
    '''
    if num_gpu==0:
        return {'/cpu:0':data_loader}

    gpus=get_available_gpus()
    if num_gpu > len(gpus):
        raise ValueError('number of gpus specified={}, more than gpus available={}'.format(num_gpu,len(gpus)))

    gpus=gpus[:num_gpu]


    data_by_gpu={g:{} for g in gpus}
    for key,value in data_loader.items():
        spl_vals=tf.split(value,num_gpu)
        for gpu,val in zip(gpus,spl_vals):
            data_by_gpu[gpu][key]=val

    return data_by_gpu


def rank(array):
    return len(array.shape)

