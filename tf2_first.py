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



'''
Getting started with tensorflow2.0
'''
"""
mental notes:
"""

##x,y,idx are protected words

def add_index(dataset):
    def flatten_index(index,elem):
        assert('index' not in elem.keys() )
        elem['idx']=index
        return elem
    enum_data=dataset.enumerate()
    return enum_data.map(flatten_index)

def float_image(elem):
    elem['x']=tf.cast(elem['image'],tf.float32)/255.
    return elem


def load_mnist():
    def binarize(elem):
        elem['y']=tf.cast( tf.math.greater(elem['label'],4), tf.uint8)
        return elem
    #WARN may shuffle train data#
    builder=tfds.builder('mnist')
    info=builder.info
    datasets=builder.as_dataset(shuffle_files=False)
    #datasets,info=tfds.load('mnist',with_info=True)
    datasets={k:v.apply(add_index) for k,v in datasets.items()}
    datasets={k:v.map(float_image) for k,v in datasets.items()}
    datasets={k:v.map(binarize)    for k,v in datasets.items()}
    return datasets,info

#    def get_fashion_mnist():
##    #Example
##    #https://www.tensorflow.org/beta/tutorials/keras/basic_classification
#        fashion_mnist = keras.datasets.fashion_mnist
#
#        (train_images, train_labels), (test_images, test_labels) =\
#            fashion_mnist.load_data()
#
#        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle, boot']
#
#        train_images = train_images / 255.0
#        test_images = test_images / 255.0
#
#        train_data=tf.data.Dataset.zip((train_images,train_labels))
#        test_data=tf.data.Dataset.zip((test_images,test_labels))
#        return train_data,test_data
    #train_data,test_data=get_fashion_mnist()


def proc_celeb_a(elem):
    x=tf.cast(elem['image'],tf.float32)/255.
    #Central crop [218x178]->[108,108]
    x=tf.image.resize_with_crop_or_pad(x, 108,108)
    x=tf.image.resize(x,[64,64], method='area')
    elem['x']=x
    #elem['y']=tf.cast( elem['attributes']['Attractive'] , tf.uint8 )
    #elem['y']=tf.cast( elem['attributes']['Black_Hair'], tf.uint8)
    #elem['y']=tf.cast( elem['attributes']['Wearing_Lipstick'], tf.uint8)
    return elem

def load_celeb_a():
    #WARN may shuffle train data#
    builder=tfds.builder('celeb_a')
    info=builder.info
    datasets=builder.as_dataset(shuffle_files=False)
    #datasets,info=tfds.load('celeb_a',with_info=True)
    datasets=tf.nest.map_structure(add_index                   ,datasets)
    datasets=tf.nest.map_structure(lambda D:D.map(proc_celeb_a),datasets)
    return datasets,info

def load_gender_celeb():
    def by_gender(elem):
        elem['y']=tf.cast( elem['attributes']['Wearing_Lipstick'], tf.uint8)
        return elem

    #['Attractive','Black_Hair','Male','Young','Smiling','Wearing_Lipstick']



    datasets,info=load_celeb_a()

#    datasets
#    datasets={k:v.map(binarize)    for k,v in datasets.items()}


def load_cifar10():
    def binarize(elem):
        elem['y']=tf.cast( tf.math.greater(elem['label'],4), tf.uint8)
        return elem
    #datasets,info=tfds.load('cifar10',with_info=True)#worried about shuffle
    builder=tfds.builder('cifar10')
    info=builder.info
    datasets=builder.as_dataset(shuffle_files=False)

    datasets={k:v.apply(add_index) for k,v in datasets.items()}
    datasets={k:v.map(float_image) for k,v in datasets.items()}
    datasets={k:v.map(binarize)    for k,v in datasets.items()}
    return datasets,info

def load_frogs_vs_ships():
    ##Frogs vs Ships##
    datasets,info=load_cifar10()
    lab0,lab1=map(info.features['label'].encode_example,['frog','ship'])
    def cifar_filter(elem):
        return tf.logical_or( tf.equal(elem['label'],lab0),
                       tf.equal(elem['label'],lab1) )
    def label2y(elem):
        elem['y']=tf.cast(
            tf.math.equal(elem['label'],lab0),
            tf.uint8)
        return elem
    datasets={k:v.filter(cifar_filter).map(label2y) for k,v in datasets.items()}
    return datasets,info

def inspect_dataset(name):
    '''
    Just for debug purposes. Quickly see what a batch is like
    '''
    train_data=tfds.load(name,split='train')
    batched=train_data.batch(32).take(1)

    batch=next(iter(batched))
    print('Keys:',batch.keys())
    return batch

def peak(dataset):
    '''also mainly for debug. see what it looks like'''
    return next(iter(dataset))

def tuple_splits(datasets):
    '''
    formats train test the way that tensorflow models.fit would like
    requires two special keys: 'x', 'y'
    '''
    #Makes format read for training
    target_data={k:v.map(lambda e:(e['x'],e['y']) ) for k,v in datasets.items()}
    train_data=target_data['train']
    test_data =target_data['test']
    return train_data,test_data



if __name__=='__main__':


    #-------#
    stophere
    #-------#

    config,_=get_config()
    prepare_dirs_and_logger(config)
    save_config(config)

    #print('model_dir:',config.model_dir)
    model_dir=config.model_dir
    #data_fn=get_toy_data(config.dataset)
    #experiment=Experiment(config,data_fn)
    #return experiment

    checkpoint_dir=os.path.join(model_dir,'checkpoints')
    model_file=os.path.join(checkpoint_dir,'Model_ckpt.h5')
    model_name=os.path.join(checkpoint_dir,'Model')
    summary_dir=os.path.join(model_dir,'summaries')
    make_folders([checkpoint_dir])#,summary_dir])
    print('[*] Model File:',model_file)


    ###--------End Main--Load Data----###

    #Prep Data
    datasets,info=load_frogs_vs_ships()
    #datasets,info=load_mnist()


    db_bat=peak(datasets['train'].batch(20))
    train_data,test_data=tuple_splits(datasets)

    train_size=info.splits['train'].num_examples
    test_size=info.splits['test'].num_examples
    buf=np.int(train_size*0.1)

    #Yes. was causing stall.hm
    #train_data=train_data.shuffle(buf)#was this causing stall?

    train_data=train_data.batch(config.batch_size).repeat()
    test_data=test_data.batch(config.batch_size).repeat()

#-------------Begin Model--------------#


    if config.load_path:
        new_model=keras.models.load_model(config.load_path)
        stophere



    #Model2   #also works fine on mnist
    model=keras.Sequential([
        keras.layers.Conv2D(64,3,activation='relu'),
        keras.layers.MaxPool2D( (2,2) ),
        #keras.layers.Conv2D(128,3,2,activation='relu'),
        keras.layers.Conv2D(128,3,activation='relu'),
        keras.layers.MaxPool2D( (2,2) ),
        keras.layers.Conv2D(64,3,activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64,activation='relu'),
        keras.layers.Dense(64,activation='relu'), #-1 if commented out

        #optional addl layers
#        keras.layers.Dense(64,activation='relu'),#+1
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(64,activation='relu'),#+6

        keras.layers.Dense(1,activation='sigmoid'),
    ])
    #'./logs/Model_0803_154817_frogship_debug'


    #model=CIFAR10Model()

    #lr=0.01 is what the mnist experiments were done on
    #lr=0.005 works for shallower cifar models
    opt=keras.optimizers.Adam(lr=0.005)#(lr=0.01)
    #opt=keras.optimizers.Adam(lr=0.0005)#(lr=0.01) #an unused exp for +5,6 layer models
    model.compile(
                #optimizer='adam',
                optimizer=opt,
                #loss='sparse_categorical_crossentropy',
                loss='binary_crossentropy',
                metrics=['accuracy'])


    print('begin train..')

    spe=np.int(train_size/config.batch_size)
    val_steps=np.int(test_size/config.batch_size)
    if config.is_train:
        model.fit(train_data,
                  epochs=20,
                  steps_per_epoch=spe,  #1563,
                  validation_data=test_data,
                  validation_freq=2,#how many train epoch between
                  validation_steps=val_steps,
                 )
        db_pred=model.predict(db_bat['x'])
        model.save(model_file)

    print('log_dir=',model_dir)
    #stophere



#-----------------
#tricks and useful code
    #Can use .batch(12000).take(1) to bring into memory

#    #debug
#    db_train_data=train_data.batch(250).take(1)
#    db_data=next(iter(db_train_data))


    #model.fit(train_data.batch(32))



    #model.fit(train_data.batch(32))
        #simply call again to continue training

    ##Useful!
    #model.get_config()
    #met=model.metrics[0]
        #met.count.numpy()




    #rz_data=train_data.map(preprocess_images)
    #rz_datum=next(iter(rz_data))
    #rz_img=rz_datum['image'].numpy()
    ##rz_img=next(iter(rz_data))

    #plt.imshow(rz_img.astype(np.uint8))
    #plt.show()

#    #mnist model  #works on mnist
#    model=keras.Sequential([
#        keras.layers.Conv2D(32,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.Flatten(),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(1,activation='sigmoid'),
#    ])



#Ex: fit()
#callbacks = [
#  # Write TensorBoard logs to `./logs` directory
#  keras.callbacks.TensorBoard(log_dir='./log/{}'.format(dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
#]
#model.fit(train_dataset, epochs=100, steps_per_epoch=1500,
#          validation_data=valid_dataset,
#          validation_steps=3, callbacks=callbacks)

#Example
#    ##mnist arch
#    model=keras.Sequential([
#        keras.layers.Conv2D(32,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,(3,3),activation='relu'),
#        keras.layers.Flatten(),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(1,activation='sigmoid'),
#    ])

#    #Example
#    #https://www.tensorflow.org/beta/tutorials/keras/basic_classification
#    fashion_mnist = keras.datasets.fashion_mnist
#
#    (train_images, train_labels), (test_images, test_labels) =\
#        fashion_mnist.load_data()
#
#    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle, boot']
#
#    train_images = train_images / 255.0
#    test_images = test_images / 255.0
#
#
#    model = keras.Sequential([
#            keras.layers.Flatten(input_shape=(28, 28)),
#            keras.layers.Dense(128, activation='relu'),
#            keras.layers.Dense(10, activation='softmax'),
#        ])
#
#
#    model.compile(optimizer='adam',
#                  loss='sparse_categorical_crossentropy',
#                  metrics=['accuracy'])
#
#    model.fit(train_images, train_labels, epochs=10)
#    test_loss, test_acc = model.evaluate(test_images, test_labels)
#    print('\nTest accuracy:', test_acc)
#
#    predictions = model.predict(test_images)
#
#    np.argmax(predictions[0])
#    test_labels[0]


#Example:
#https://adventuresinmachinelearning.com/keras-eager-and-tensorflow-2-0-a-new-tf-paradigm/
#class CIFAR10Model(keras.Model):
#    def __init__(self):
#        super(CIFAR10Model, self).__init__(name='cifar_cnn')
#        self.conv1 = keras.layers.Conv2D(64, 5,
#                                         padding='same',
#                                         activation=tf.nn.relu,
#                                         kernel_initializer=tf.initializers.variance_scaling,
#                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.max_pool2d = keras.layers.MaxPooling2D((3, 3), (2, 2), padding='same')
#        self.max_norm = keras.layers.BatchNormalization()
#        self.conv2 = keras.layers.Conv2D(64, 5,
#                                         padding='same',
#                                         activation=tf.nn.relu,
#                                         kernel_initializer=tf.initializers.variance_scaling,
#                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.flatten = keras.layers.Flatten()
#        self.fc1 = keras.layers.Dense(750, activation=tf.nn.relu,
#                                      kernel_initializer=tf.initializers.variance_scaling,
#                                      kernel_regularizer=keras.regularizers.l2(l=0.001))
#        self.dropout = keras.layers.Dropout(0.5)
#        self.fc2 = keras.layers.Dense(10)
#        self.softmax = keras.layers.Softmax()
#
#    def call(self, x):
#        x = self.max_pool2d(self.conv1(x))
#        x = self.max_norm(x)
#        x = self.max_pool2d(self.conv2(x))
#        x = self.max_norm(x)
#        x = self.flatten(x)
#        x = self.dropout(self.fc1(x))
#        x = self.fc2(x)
#        return self.softmax(x)

#Example:
#https://adventuresinmachinelearning.com/keras-eager-and-tensorflow-2-0-a-new-tf-paradigm/
class CIFAR10Model(keras.Model):
    def __init__(self):
        super(CIFAR10Model, self).__init__(name='cifar_cnn')
        self.conv1 = keras.layers.Conv2D(64, 5,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.VarianceScaling,
                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.max_pool2d = keras.layers.MaxPooling2D((3, 3), (2, 2), padding='same')
        self.max_norm = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(64, 5,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.initializers.VarianceScaling,
                                         kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(750, activation=tf.nn.relu,
                                      kernel_initializer=tf.initializers.VarianceScaling,
                                      kernel_regularizer=keras.regularizers.l2(l=0.001))
        self.dropout = keras.layers.Dropout(0.5)
        self.fc2 = keras.layers.Dense(10)
        self.softmax = keras.layers.Softmax()

    def call(self, x):
        x = self.max_pool2d(self.conv1(x))
        x = self.max_norm(x)
        x = self.max_pool2d(self.conv2(x))
        x = self.max_norm(x)
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

