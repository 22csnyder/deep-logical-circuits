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
        width=config.width1
        if not width:
            raise ValueError('no width was passed')

        #initial=tf.truncated_normal_initializer(mean=0.,stddev=0.5,seed=None)#float32
        if not config.init_stdev1:
            initial=tf.truncated_normal_initializer(mean=0.,stddev=0.05)#float32
        else:
            initial=tf.truncated_normal_initializer(mean=0.,stddev=config.init_stdev1)#float32

        ##The actual model##
        with tf.variable_scope('weights',initializer=initial):

            ### Shallow ####
            self.W1=tf.get_variable('W1',[self.x.shape[-1],width])
            self.b1=tf.get_variable('b1',[width],initializer=tf.zeros_initializer())
            self.u=tf.get_variable('u',[width,2])
            self.W2=self.u
            self.b2=tf.get_variable('b2',[self.u.shape[-1]],initializer=tf.zeros_initializer())
            #self.weights=[self.W1,self.u]
            self.weights={'W1':self.W1,
                          'W2':self.W2,
                          'b1':self.b1,
                          'b2':self.b2,
                          'u':self.u}
            self.h1=tf.matmul(self.phx,self.W1)+self.b1
            self.a1=rho(self.h1)
            self.logits=tf.matmul(self.a1,self.u)+self.b2

            ##### DEEP  ####
            #print 'WARNING Deep Network used! (Not all parts of code may function!!)'
            #print 'WARN. DEBUG'
            #self.W1=tf.get_variable('W1',[self.x.shape[-1],width])
            #self.b1=tf.get_variable('b1',[width],initializer=tf.zeros_initializer())
            #self.W2=tf.get_variable('W2',[width,width+2])
            #self.b2=tf.get_variable('b2',[width+2],initializer=tf.zeros_initializer())
            #self.W3=tf.get_variable('W3',[width+2,width+5])
            #self.b3=tf.get_variable('b3',[width+5],initializer=tf.zeros_initializer())
            #self.u=tf.get_variable('u',[width+5,2])
            #self.bu=tf.get_variable('bu',[self.u.shape[-1]],initializer=tf.zeros_initializer())
            ##self.weights=[self.W1,self.u]

            #self.a1=rho(tf.matmul(self.phx,self.W1)+self.b1)
            #self.a2=rho(tf.matmul(self.a1,self.W2)+self.b2)
            #self.a3=rho(tf.matmul(self.a2,self.W3)+self.b3)
            #self.logits=tf.matmul(self.a3,self.u)+self.bu
            ###################


            self.delu=self.u[:,1]-self.u[:,0]
            self.weights={'W1':self.W1,
                          'W2':self.W2,
                          'b1':self.b1,
                          'b2':self.b2,
                          'u':self.u}
            self.weights['delu']=self.delu

        ###Some summaries###
        delui=tf.split(self.delu,self.delu.shape[0])
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

#class Model2(Model):
#    def __init__(self,config,batch):
#        super(Model2,self).__init__(self,config,batch)

class Experiment(object):
    def __init__(self,config,data):
        self.config=config
        self.model_dir=self.config.model_dir
        self.summary_dir=self.model_dir #for now
        self.checkpoint_dir=os.path.join(self.model_dir,'checkpoints')
        self.model_name=os.path.join(self.checkpoint_dir,'Model')
        self.summary_dir=os.path.join(self.model_dir,'summaries')
        self.data=data

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
        self.gX=gX #debug

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


            #ax.contourf(gridX[:,:,0],gridX[:,:,1],gridP[:,:,1],cmap=plt.cm.bwr_r)


            if counter % self.config.log_every_n_steps == 0:#log at the start
                t0=time.time()
                step,summ=sess.run([model.global_step,self.summary_op])
                gP=sess.run(model.prob,feed_dict={model.phx:gX})
                gridP=gP.reshape(self.gridX.shape)
                self.hmw.concat({'Prob':np.expand_dims(gridP,axis=0)})
                self.gP=gP#debug
                self.gridP=gridP

                self.summary_writer.add_summary(summ,step)
                self.summary_writer.flush()

                #self.ww2.concat(sess.run(self.model.weights))
                self.weight_watcher.concat(sess.run(wt_fetch))

        #save at end
        self.saver.save(self.sess,self.model_name, global_step=self.global_step)
        self.write_records()
        lo,ac=sess.run([model.net_loss,model.accuracy])
        print 'final loss ',lo, '  acc: ',ac


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
        np.save(fname,  self.gridX  )

#def __init__(self,config,batch):
if __name__=='__main__':
    tf.reset_default_graph()

    xdim=1
    halfN=30
    #Data simple experiment 1025_214221
    #xpos=np.random.rand(halfN,xdim)+3
    #xneg=np.random.rand(halfN,xdim)-3
    #npX=np.vstack([xpos,xneg])
    #npY=(npX>0.).astype(np.int64)

    #Data complicated experiment 
    xpos1=np.random.rand(halfN,xdim)+3
    xpos2=np.random.rand(halfN,xdim)-1
    xneg1=np.random.rand(halfN,xdim)-3
    xneg2=np.random.rand(halfN,xdim)+1
    npX=np.vstack([xpos1,xpos2,xneg1,xneg2])
    npY=(npX>0.).astype(np.int64)
    npY=np.vstack(np.ones((2*halfN,1)),-np.ones((2*halfN,1)))



    X=tf.constant(npX,dtype=tf.float32)
    Y=tf.constant(npY)

    data={'input':X,'label':Y}

    config,_=get_config()
    model=Model(config,data)

    exp=Experiment(config,data)

    V=tf.global_variables()


