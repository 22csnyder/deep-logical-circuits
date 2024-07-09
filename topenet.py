'''
This file is to investigate the progression of the upper vertices during
training.
Probably will try bullseye data first.
'''

import pandas as pd
import numpy as np
from config import get_config
from utils import prepare_dirs_and_logger,save_config
from toydata import get_toy_data

import tensorflow as tf
import math
import numpy as np
import os
import sys
#sys.path.append('/home/chris/Software/workspace/models/slim')
from tqdm import trange
import time

from ArrayDict import ArrayDict

#temp for debug
from config import get_config
from nonlinearities import name2nonlinearity

#from data_loader import DataLoader
#from summary_helpers import (summary_stats,summary_class_separability,
#                             discrim_summary,make_summary,sorted_hist,summary_T,
#                             min_hamming_decode )
#from utils import make_folders,copy_weights,bool_diff_sign


class Model(object):
    def __init__(self,config,batch):
        self.name='Model'
        self.batch=batch
        self.config=config

        if self.config.optimizer =='grad_descent':
            self.optimizer=tf.train.GradientDescentOptimizer
        elif self.config.optimizer =='adam':#*
            self.optimizer = tf.train.AdamOptimizer
        self.net_opt=self.optimizer(self.config.learning_rate)
        self.build_model()

    def build_model(self):
        config=self.config

        self.x=self.batch['input'] #-1xdim float32
        self.y=self.batch['label'] #-1x1   int64
        self.phx=tf.placeholder_with_default(self.x,[None,self.x.shape[-1]]) #-1xdim float32
        self.phy=tf.placeholder_with_default(self.y,[None,self.y.shape[-1]]) #-1x1   int64
        self.y01=tf.cast(0.5*(tf.cast(self.phy,tf.float32)+1),tf.int64)
        self.oh_y = tf.one_hot(tf.reshape(self.y01,[-1]), depth=2, axis=-1)
        #self.y01=tf.cast(0.5*(tf.cast(self.y,tf.float32)+1),tf.int64)
        #self.oh_y = tf.one_hot(tf.reshape(self.y01,[-1]), depth=2, axis=-1)

        rho=name2nonlinearity(self.config.nonlinearity)
        self.rho=rho

        #initial=tf.truncated_normal_initializer(mean=0.,stddev=0.5,seed=None)#float32
        if not config.init_stdev1:
            initial=tf.truncated_normal_initializer(mean=0.,stddev=0.05)#float32
        else:
            initial=tf.truncated_normal_initializer(mean=0.,stddev=config.init_stdev1)#float32

        ##The actual model##

        with tf.variable_scope('weights',initializer=initial):

        #ReLU no biases!
            ##arch must end in 2 for 2 class
            #arch=[4,4,4,2]

            if config.arch==1:
                arch=[4,6,8,2]#Small    A1

            if config.arch==11:
                arch=[4,6,8,10,2]
            if config.arch==12:
                arch=[4,6,8,10,12,2]


            elif config.arch==2:
                arch=[4,6,8,10,12,14,2]#Big1     A2

            elif config.arch==21:
                arch=[4,6,8,10,12,14,16,2]
            elif config.arch==22:
                arch=[4,6,8,10,12,14,16,20,2]


            elif config.arch==3:
                arch=[4,6,8,10,12,14,16,20,30,2]#Big3  A3
            if config.arch==1:
                arch=[4,6,8,2]#Small    A1
            #  A1        A2       A3
            #W  107      517      1743
            #VC 8376     101110   709558

            #arch=[3,3,2]
            #arch=[3,3,3,2]
            print 'ARCH is set to ',arch
            #arch=[15,15,15,2]
            #arch=[7,7,7,2]
            #print 'Arch:',arch
            Wweights=[]
            Bweights=[]
            self.weights={}

            if config.pca:
                arch=[config.pca]+arch
                print 'Using Linear Projection Layer of size ',config.pca


            arch_shift=[self.x.shape[-1]]+arch[:-1]
            for l,widpair in enumerate(zip(arch_shift,arch)):
                now_wid,nxt_wid=widpair
                Wname='W'+str(l+1)#start at 1 for some reason
                Bname='b'+str(l+1)
                Wi=tf.get_variable(Wname,[now_wid,nxt_wid])
                Bi=tf.get_variable(Bname,[nxt_wid],initializer=tf.zeros_initializer())
                Wweights.append(Wi)
                Bweights.append(Bi)
                self.weights[Wname]=Wi
                self.weights[Bname]=Bi
            self.u=Wweights[-1]


            ai=self.phx#init
            self.preacts=[]
            self.acts=[]
            for Wi,Bi in zip(Wweights,Bweights):
                self.acts.append(ai)#the input is the first entry
                hi=tf.matmul(ai,Wi)+Bi
                if len(self.acts)==1 and config.pca:#First lyr do linear
                    ai=hi
                else:
                    ai=rho(hi)
                self.preacts.append(hi)#the output is the last entry
            #self.acts=self.acts[:-1]#last one nonsense#relu(logit)
            self.logits=self.preacts[-1]#output


            self.delu=self.u[:,1]-self.u[:,0]
            self.weights['u']=self.u
            self.weights['delu']=self.delu

        ###Some summaries###
        #delui=tf.split(self.delu,self.delu.shape[0])
        #vi=tf.split(self.v[0],self.v.shape[-1])
        ##with tf.name_scope('weights'):
        #for i in range(width):
        #    uu=tf.squeeze(delui[i])
        #    vv=tf.squeeze(vi[i])
        #    tf.summary.scalar('u'+str(i),uu,family='weights')
        #    tf.summary.scalar('v'+str(i),vv,family='weights')

        self.prob=tf.nn.softmax(self.logits)
        self.predictions = tf.argmax(self.logits, 1)
        self.pred= tf.reshape(self.predictions,[-1,1])
        self.bool_correct= tf.reshape(tf.equal(self.pred,self.y01),[-1,1])
        self.accuracy=tf.reduce_mean(tf.cast(self.bool_correct,tf.float32))
        tf.summary.scalar('accuracy',self.accuracy)

    def calc_loss(self): ##LOSSES##
        self.classification_loss=tf.losses.softmax_cross_entropy(
            logits=self.logits, onehot_labels=self.oh_y)
        self.net_loss=self.classification_loss

        tf.summary.scalar('loss/classification',self.classification_loss)
        tf.summary.scalar('loss/net',self.net_loss)

        #self.global_step=tf.train.get_or_create_global_step()
        self.global_step=tf.Variable(0,name='target_step',trainable=False,dtype=tf.int64)
        #self.net_updates=self.net_opt.minimize(self.net_loss,var_list=self.net_vars,global_step=self.global_step)
        self.net_updates=self.net_opt.minimize(self.net_loss,global_step=self.global_step)

        update_ops=[self.net_updates]
        update_op=tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            self.train_tensor = tf.identity(self.classification_loss)


class Experiment(object):
    def __init__(self,config,data_fn):
        self.config=config
        self.model_dir=self.config.model_dir
        self.summary_dir=self.model_dir #for now
        self.checkpoint_dir=os.path.join(self.model_dir,'checkpoints')
        self.model_name=os.path.join(self.checkpoint_dir,'Model')
        self.summary_dir=os.path.join(self.model_dir,'summaries')
        self.data_fn=data_fn
        self.data=data_fn()
        data=self.data


        ##NETWORK##
        self.model=Model(config,data)
        self.model.calc_loss()

        ##HOUSE KEEPING##
        self.global_step=self.model.global_step
        self.summary_op=tf.summary.merge_all()
        self.saver=tf.train.Saver(keep_checkpoint_every_n_hours=1.0)#!

        t0=time.time()
        self.get_session()#self.sess
        print '{}sec to get session'.format(time.time()-t0)

        if 'mnist' in config.dataset:#others are validated later
            self.val_data=self.data_fn(subset='valid')
            self.val_npX,self.val_npY=self.sess.run([self.val_data['input'],self.val_data['label']])
            self.val_npY01=((self.val_npY+1.)/2.).astype('int')

        self.npX,self.npY=self.sess.run([self.data['input'],self.data['label']])
        ##Load model params if given
        if config.load_path:
            if tf.gfile.IsDirectory(config.load_path):
                checkpoint_path = tf.train.latest_checkpoint(config.load_path)
                if checkpoint_path is None:
                    alt_load_path=os.path.join(config.load_path,'checkpoints')
                    checkpoint_path = tf.train.latest_checkpoint(alt_load_path)
            else:
                checkpoint_path = config.load_path#given checkpoint directly
            print('Attempting to load model:',checkpoint_path)
            self.saver.restore(self.sess,checkpoint_path)

            pnt_str='Loaded variables at Step:{}'
            step=self.sess.run(self.global_step)
            pnt_str=pnt_str.format(step)
            print pnt_str

        else:#not loading
            #save data
            np.save(os.path.join(config.record_dir,'dataX.npy'),self.npX)
            np.save(os.path.join(config.record_dir,'dataY.npy'),self.npY)
            #np.savetxt(os.path.join(config.record_dir,'dataX'),self.npX)#only 2D
            #np.savetxt(os.path.join(config.record_dir,'dataY'),self.npY)

    def get_session(self):
        self.summary_writer = tf.summary.FileWriter(self.summary_dir)

        print('experiment.model_dir:',self.model_dir)
        gpu_options = tf.GPUOptions(allow_growth=True,
                                   )
                                  #per_process_gpu_memory_fraction=0.5)
                                  #per_process_gpu_memory_fraction=0.333)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)

        self.sess = tf.Session(config=sess_config)
        init=tf.global_variables_initializer()
        self.sess.run(init)

    def train_loop(self,num_iter=None):
        ''' it trains the model '''
        num_iter=num_iter or self.config.num_iter

        sess=self.sess
        model=self.model
        train_tensor=model.train_tensor

        wt_output={'step':tf.reshape(self.global_step,[1]) }
        wt_output.update(self.model.weights)
        wt_fetch={k:tf.expand_dims(v,axis=0) for k,v in wt_output.items()}
        self.wt_fetch=wt_fetch
        self.weight_watcher=ArrayDict()
        self.heatmap_watcher=ArrayDict()
        self.ww=self.weight_watcher#convenience
        self.hmw=self.heatmap_watcher
        #self.ww2=ArrayDict()


        if not 'mnist' in self.config.dataset:
            ###Heatmap Boundaries###Probably move this sw else later
            size=self.npX.max(axis=0)-self.npX.min(axis=0)
            lower=self.npX.min(axis=0)-0.1*size
            upper=self.npX.max(axis=0)+0.1*size
            delta=0.05*np.min(size)
            x1=np.arange(lower[0],upper[0],delta)#assuming 2D data
            x2=np.arange(lower[1],upper[1],delta)
            #x1=np.arange(-2.5,2.51,delta)
            #x1=x2=np.arange(-2.5,2.51,delta)
            #X1,X2=np.meshgrid(x,y)
            gridX=np.stack(np.meshgrid(x1,x2),axis=-1)#LxLx2
            self.gridX=gridX
            gX=np.reshape(gridX,[-1,2])#assuming 2D data
            self.gX=gX #DEBUG

        #Setup just for tf.Constant data
        #Train loop
        print('Entering train loop..')
        print 'ni is',num_iter
        for counter in trange(num_iter):
            losses_tuple=self.sess.run(train_tensor)

            if (counter+1) % self.config.save_every_n_steps==0:#dont log at start
                t0=time.time()
                self.saver.save(sess,self.model_name, global_step=self.global_step)
                #print 'savetime',time.time()-t0
                self.write_records()#saves weights all timepoints#Overwrites each time

            #ax.contourf(gridX[:,:,0],gridX[:,:,1],gridP[:,:,1],cmap=plt.cm.bwr_r)

            if counter % self.config.log_every_n_steps == 0:#log at the start
                t0=time.time()
                step,summ=sess.run([model.global_step,self.summary_op])

                if not 'mnist' in self.config.dataset:
                    gP=sess.run(model.prob,feed_dict={model.phx:gX})
                    gridP=gP.reshape(self.gridX.shape)
                    self.hmw.concat({'Prob':np.expand_dims(gridP,axis=0)})
                    self.gP=gP#DEBUG
                    self.gridP=gridP
                ###debug then move elsewhere
                if 'mnist' in config.dataset:#others are validated later
                    val_pred=sess.run(model.pred,feed_dict={model.phx:self.val_npX})
                    self.val_pred=val_pred
                    assert( (val_pred.shape==self.val_npY01.shape) )
                    val_acc=np.mean(val_pred==self.val_npY01)
                #    print 'validation accuracy: ',val_acc
                    val_sum=tf.Summary()
                    val_sum.value.add(tag='validation accuracy',
                                      simple_value=val_acc)
                    self.summary_writer.add_summary(val_sum,step)

                self.summary_writer.add_summary(summ,step)
                self.summary_writer.flush()

                #self.ww2.concat(sess.run(self.model.weights))
                self.weight_watcher.concat(sess.run(wt_fetch))

        #save at end
        self.saver.save(self.sess,self.model_name, global_step=self.global_step)
        self.write_records()
        lo,ac=sess.run([model.net_loss,model.accuracy])
        print 'final loss ',lo, '  train acc: ',ac

        if 'mnist' in config.dataset:#others are validated later
            print 'valid acc: ',val_acc



        #Calc some validation stuff



    def write_records(self):
        #Record the weights
        pfx='wwatch_'
        ext='.npy'
        for k,v in self.ww.items():
            fname=os.path.join(self.config.record_dir,pfx+k+ext)
            np.save(fname,v)
            #np.savetxt(fname,v)

        #Heatmap stuff
        pfx='hmwatch_'
        fname=os.path.join(self.config.record_dir,pfx+'Prob'+ext)
        np.save(fname,  self.hmw['Prob']  )
        fname=os.path.join(self.config.record_dir,pfx+'gridX'+ext)

        if not 'mnist' in self.config.dataset:
            np.save(fname,  self.gridX  )

        if 'mnist' in self.config.dataset:
            #print 'saving validation predictions, shape',self.val_pred.shape
            val_fname=os.path.join(self.config.record_dir,'val_predictions'+ext)
            np.save(val_fname,self.val_pred)

def main():
    #config.dataset='quad_data'#supply as arg 
    #config.nonlinearity='relu'

    tf.reset_default_graph()

    config,_=get_config()
    prepare_dirs_and_logger(config)
    save_config(config)

    print 'model_dir:',config.model_dir

    data_fn=get_toy_data(config.dataset)

    experiment=Experiment(config,data_fn)

    return experiment
#def __init__(self,config,batch):

if __name__=='__main__':

    #tf.reset_default_graph()
    #config,_=get_config()
    #prepare_dirs_and_logger(config)
    #save_config(config)
    #print 'model_dir:',config.model_dir
    #data=get_toy_data(config.dataset)()
    #experiment=Experiment(config,data)
    #breakhere


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


#if __name__=='__main__':
#    tf.reset_default_graph()
#
#    xdim=1
#    halfN=30
#    #Data simple experiment 1025_214221
#    #xpos=np.random.rand(halfN,xdim)+3
#    #xneg=np.random.rand(halfN,xdim)-3
#    #npX=np.vstack([xpos,xneg])
#    #npY=(npX>0.).astype(np.int64)
#
#    #Data complicated experiment 
#    xpos1=np.random.rand(halfN,xdim)+3
#    xpos2=np.random.rand(halfN,xdim)-1
#    xneg1=np.random.rand(halfN,xdim)-3
#    xneg2=np.random.rand(halfN,xdim)+1
#    npX=np.vstack([xpos1,xpos2,xneg1,xneg2])
#    npY=(npX>0.).astype(np.int64)
#    npY=np.vstack(np.ones((2*halfN,1)),-np.ones((2*halfN,1)))
#
#
#
#    X=tf.constant(npX,dtype=tf.float32)
#    Y=tf.constant(npY)
#
#    data={'input':X,'label':Y}
#
#    config,_=get_config()
#    model=Model(config,data)
#
#    exp=Experiment(config,data)
#
#    V=tf.global_variables()

