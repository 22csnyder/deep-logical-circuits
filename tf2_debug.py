import tensorflow_datasets as tfds

from tf2_first import load_mnist


if __name__=='__main__':

    #datasets,info=tfds.load('mnist',with_info=True)
    #datasets=tfds.load('mnist')
    datasets,info=load_mnist()


    print('finished!')
