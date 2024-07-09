import tensorflow as tf
import math
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data as mnist_data


datafns={}


#    self.mnist_datasets=mnist_data.read_data_sets(config.dataset_dir,reshape=False,validation_size=0)
#    N_train=self.mnist_datasets.train.num_examples
#    N_test=self.mnist_datasets.test.num_examples
#
#    #I dont think following applies anymore
#    eff_train_size=config.validation_every_n_steps*config.batch_size
#    if eff_train_size<N_train:
#        print 'WARN: only seeing first ',eff_train_size,'/',N_train,' train samples\
#        each time before validation'
#
#    np.random.seed(22)
#    train_perm=np.random.permutation(N_train)
#    test_perm=np.random.permutation(N_test)
#
#    train_slices={
#         'image':tf.constant(self.mnist_datasets.train.images[train_perm]),
#         'label':tf.constant(self.mnist_datasets.train.labels[train_perm]),
#        #can try data.Dataset.range(N_train)
#         'index':tf.constant(np.linspace(1,N_train,N_train).reshape([-1,1]),dtype=tf.uint8)}
#    test_slices={
#         'image':tf.constant(self.mnist_datasets.test.images[test_perm]),
#         'label':tf.constant(self.mnist_datasets.test.labels[test_perm]),
#         'index':tf.constant(np.linspace(1,N_test,N_test).reshape([-1,1]),dtype=tf.uint8)}
#
#
#    print 'train slices keys',train_slices.keys()
#
#    ds_train=tf.data.Dataset.from_tensor_slices(train_slices)
#    ds_test=tf.data.Dataset.from_tensor_slices(test_slices)
#
#    print 'ds_train output shapes',ds_train.output_shapes



#    if config.binary_labels:
#        ds_train=ds_train.map(set_label_as_01,num_parallel_calls=npc)
#        ds_test=ds_test.map(set_label_as_01,num_parallel_calls=npc)
#
#    if not config.num_train_samples:
#        print "WARNING: USING ALL training samples"
#        self.small_ds_train=ds_train
#    else:
#        #-1 will use all the samples
#        self.small_ds_train=ds_train.take(config.num_train_samples)
#
#    #ds_train=ds_train.map(set_mask_to_zero,num_parallel_calls=npc)
#
#    #whole train dataset
#    self.stream_train_dataset=ds_train.repeat().batch(config.batch_size)
#    self.once_train_dataset=ds_train.batch(config.batch_size)
#    self.stream_train_dataset=self.stream_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)
#    self.once_train_dataset=self.once_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)
#
#    #subset
#
#    self.stream_small_train_dataset=self.small_ds_train.repeat().batch(config.batch_size)
#    self.once_small_train_dataset=self.small_ds_train.batch(config.batch_size)
#    self.stream_small_train_dataset=self.stream_small_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)
#    self.once_small_train_dataset=self.once_small_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)
#
#    #test set
#    self.test_dataset=ds_test.batch(config.batch_size)
#    self.test_dataset=self.test_dataset.map(self.image_preprocess,num_parallel_calls=16)


def Ylabels(npos,nneg):
    npY=np.vstack([np.ones((npos,1)),-np.ones((nneg,1))]).astype(np.int64)
    return npY


if __name__=='__main__':


    mnist_datasets=mnist_data.read_data_sets('./data/mnist/',reshape=False,validation_size=0)
    #5,6,8,9

    Images=mnist_datasets.train.images
    Labels=mnist_datasets.train.labels

    labs=[[5,6],
          [8,9]]
    lab0,lab1=labs

    id0=np.where(np.logical_or(Labels==lab0[0],Labels==lab0[1]))[0]#label0
    id1=np.where(np.logical_or(Labels==lab1[0],Labels==lab1[1]))[0]#label1

    Xpos=Images[id1]
    Xneg=Images[id0]
    npY=Ylabels(len(Xpos),len(Xneg))
    npX=np.vstack([Xpos,Xneg])

    np.random.seed(20)
    train_perm=np.random.permutation(len(npX))

    npX=npX[train_perm]
    npY=npY[train_perm]

    #[train_perm]
    #[train_perm]
#def set_label_as_01(input_data):
#    label=input_data['label']
#    new_label=tf.cast(tf.greater_equal(label,5),label.dtype)#uint8
#    input_data['label']=new_label
#    return input_data
#



