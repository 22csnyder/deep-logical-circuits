import tensorflow as tf
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from activation_mask import get_masks


num_preprocessing_threads=8
labels_offset=0
num_readers=16

sys.path.append('/home/chris/Software/workspace/models/slim')
from datasets import dataset_factory
from preprocessing import preprocessing_factory
slim = tf.contrib.slim

def set_label_as_01(input_data):
    label=input_data['label']
    new_label=tf.cast(tf.greater_equal(label,5),label.dtype)#uint8
    input_data['label']=new_label
    return input_data


class DataLoader:
    def image_preprocess(self,batch):
        images=batch['image']
        labels=batch['label']
        if self.default_image_size==28:
            rsz_images=images #do nothing
        else:
            rsz_images=tf.image.resize_bilinear(images,
                    [self.default_image_size,self.default_image_size])
        out_batch={'image':rsz_images,
                   'label':labels}
        batch.update(out_batch)
        return batch


    def __init__(self,config,default_image_size=28):
        self.config=config
        self.default_image_size=default_image_size

        #possible_splits=['train','eval','']

        if 'mnist'==config.dataset:
            if config.binary_labels:
                self.num_classes=2
            else:
                self.num_classes=10


            self.mnist_datasets=mnist_data.read_data_sets(config.dataset_dir,reshape=False,validation_size=0)
            N_train=self.mnist_datasets.train.num_examples
            N_test=self.mnist_datasets.test.num_examples

            #I dont think following applies anymore
            eff_train_size=config.validation_every_n_steps*config.batch_size
            if eff_train_size<N_train:
                print 'WARN: only seeing first ',eff_train_size,'/',N_train,' train samples\
                each time before validation'

            ###mask dataset for small_train_dataset###
            #masks are only for small subset so I'm doing them separately..
            #careful to align with small dataset

            #if config.r is None:
            #    raise ValueError('config --r was not passed')
            #n_hidden_layers=len(config.hidden_layers)
            #wid_layers=config.hidden_layers
            #nts=config.num_train_samples
            #masks=get_masks(r=config.r,k=nts,layers=wid_layers)

            #train_pad_masks=[np.zeros([N_train-nts,w],dtype='int') for w in wid_layers]
            #train_masks=[np.vstack([m,pm]) for m,pm in zip(masks,train_pad_masks)]
            #test_masks=[np.zeros([N_test,w],dtype='int') for w in wid_layers]


            #self.layer_names=self.config.hidden_layer_names

            #train_mask_slices={n:tf.constant(m,dtype=tf.uint8) for n,m in zip(self.layer_names,train_masks)}
            #test_mask_slices={n:tf.constant(m,dtype=tf.uint8) for n,m in zip(self.layer_names,test_masks)}

            #def set_mask_to_zero(input_data):
            #    #I want to zero out masks for train data
            #    for name in self.layer_names:
            #        input_data[name]*=0
            #    return input_data

            np.random.seed(22)
            train_perm=np.random.permutation(N_train)
            test_perm=np.random.permutation(N_test)

            train_slices={
                 'image':tf.constant(self.mnist_datasets.train.images[train_perm]),
                 'label':tf.constant(self.mnist_datasets.train.labels[train_perm]),
                #can try data.Dataset.range(N_train)
                 'index':tf.constant(np.linspace(1,N_train,N_train).reshape([-1,1]),dtype=tf.uint8)}
            test_slices={
                 'image':tf.constant(self.mnist_datasets.test.images[test_perm]),
                 'label':tf.constant(self.mnist_datasets.test.labels[test_perm]),
                 'index':tf.constant(np.linspace(1,N_test,N_test).reshape([-1,1]),dtype=tf.uint8)}

            #train_slices={
            #     'image':tf.constant(self.mnist_datasets.train.images),
            #     'label':tf.constant(self.mnist_datasets.train.labels),
            #    #can try data.Dataset.range(N_train)
            #     'index':tf.constant(np.linspace(1,N_train,N_train).reshape([-1,1]),dtype=tf.uint8)}
            #test_slices={
            #     'image':tf.constant(self.mnist_datasets.test.images),
            #     'label':tf.constant(self.mnist_datasets.test.labels),
            #     'index':tf.constant(np.linspace(1,N_test,N_test).reshape([-1,1]),dtype=tf.uint8)}

            #self.train_slices=train_slices
            #self.test_slices=test_slices

            #train_slices.update(train_mask_slices)
            #test_slices.update(test_mask_slices)

            print 'train slices keys',train_slices.keys()

            ds_train=tf.data.Dataset.from_tensor_slices(train_slices)
            ds_test=tf.data.Dataset.from_tensor_slices(test_slices)

            print 'ds_train output shapes',ds_train.output_shapes


            ##ds_train
            #ds_train=tf.data.Dataset.from_tensor_slices(
            #    {'image':tf.constant(self.mnist_datasets.train.images),
            #     'label':tf.constant(self.mnist_datasets.train.labels),
            #     'index':tf.constant(np.linspace(1,N_train,N_train).reshape([-1,1]))}
            #    )
            #ds_test=tf.data.Dataset.from_tensor_slices(
            #    {'image':tf.constant(self.mnist_datasets.test.images),
            #     'label':tf.constant(self.mnist_datasets.test.labels),
            #     'index':tf.constant(np.linspace(1,N_test,N_test).reshape([-1,1]))})}
            #    )

            npc=2#16#num_parallel calls

            if config.binary_labels:
                ds_train=ds_train.map(set_label_as_01,num_parallel_calls=npc)
                ds_test=ds_test.map(set_label_as_01,num_parallel_calls=npc)

            if not config.num_train_samples:
                print "WARNING: USING ALL training samples"
                self.small_ds_train=ds_train
            else:
                #-1 will use all the samples
                self.small_ds_train=ds_train.take(config.num_train_samples)

            #ds_train=ds_train.map(set_mask_to_zero,num_parallel_calls=npc)

            #whole train dataset
            self.stream_train_dataset=ds_train.repeat().batch(config.batch_size)
            self.once_train_dataset=ds_train.batch(config.batch_size)
            self.stream_train_dataset=self.stream_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)
            self.once_train_dataset=self.once_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)

            #subset

            self.stream_small_train_dataset=self.small_ds_train.repeat().batch(config.batch_size)
            self.once_small_train_dataset=self.small_ds_train.batch(config.batch_size)
            self.stream_small_train_dataset=self.stream_small_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)
            self.once_small_train_dataset=self.once_small_train_dataset.map(self.image_preprocess,num_parallel_calls=npc)

            #test set
            self.test_dataset=ds_test.batch(config.batch_size)
            self.test_dataset=self.test_dataset.map(self.image_preprocess,num_parallel_calls=16)


            #self.iterator=tf.data.Iterator.from_structure(self.stream_small_train_dataset.output_types,self.stream_small_train_dataset.output_shapes)
            ##print 'DEBUG','iterator shapes',self.iterator.output_shapes

            #self.batch=self.iterator.get_next()


            #self.stream_training_init_op=self.iterator.make_initializer(self.stream_train_dataset)
            #self.stream_small_training_init_op=self.iterator.make_initializer(self.stream_small_train_dataset)

            #self.eval_traindata_init_op=self.iterator.make_initializer(self.once_train_dataset)
            #self.eval_small_traindata_init_op=self.iterator.make_initializer(self.once_small_train_dataset)
            #self.eval_testdata_init_op=self.iterator.make_initializer(self.test_dataset)


            ##Each dataset has a stream and a eval component
            #train data remembers where you left off when you come back from doing eval
            self.stream_training_iterator      =self.stream_train_dataset.make_one_shot_iterator()
            self.stream_small_training_iterator=self.stream_small_train_dataset.make_one_shot_iterator()

            #eval data resets every time it is returned to so order preserved
            self.eval_traindata_iterator      =self.once_train_dataset.make_initializable_iterator()
            self.eval_small_traindata_iterator=self.once_small_train_dataset.make_initializable_iterator()
            self.eval_testdata_iterator       =self.test_dataset.make_initializable_iterator()

            self.eval_traindata_init_op       =self.eval_traindata_iterator.initializer
            self.eval_small_traindata_init_op =self.eval_small_traindata_iterator.initializer
            self.eval_testdata_init_op        =self.eval_testdata_iterator.initializer


            ###Switching to feedable iterator for switching between training & validation
            self.handle=tf.placeholder(tf.string,shape=[])
            self.iterator=tf.data.Iterator.from_string_handle(self.handle,
                                           self.test_dataset.output_types,
                                           self.test_dataset.output_shapes)
            self.batch=self.iterator.get_next()



        else:
            raise ValueError('dataset loading not implemented for\
                             dataset',config.dataset)


    def setup_iterator_handles(self,session):

        self.stream_training_handle=session.run(self.stream_training_iterator.string_handle())
        self.stream_small_training_handle=session.run(self.stream_small_training_iterator.string_handle())
        self.eval_traindata_handle=session.run(self.eval_traindata_iterator.string_handle())
        self.eval_small_traindata_handle=session.run(self.eval_small_traindata_iterator.string_handle())
        self.eval_testdata_handle=session.run(self.eval_testdata_iterator.string_handle())



