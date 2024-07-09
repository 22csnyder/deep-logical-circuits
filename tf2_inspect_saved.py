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

#If we want to use
#@tf.function()
#   In order to accumulate results from a dynamically unrolled loop,
#   you'll want to use tf.TensorArray.
#https://www.tensorflow.org/beta/tutorials/eager/tf_function

#Or! we could use map to get the embedded values/states on a dataset level
    #that could be cool
    #Then we could use tf.d.D.reduce to stack them

def unpack(data,model):
    Out={}

    #Assume some stuff followed by fc layers
    is_fc=[isinstance(L,keras.layers.Dense) for L in model.layers]
    fc_start=min([i for i,dense in enumerate(is_fc) if dense])

    emb_layers=model.layers[:fc_start]
    fc_layers=model.layers[fc_start:]

    Ys=[]
    Embed=[]
    States=[]
    for x,y in data:
        Ys.append(y)
        states=[]
        #tf.print(y)
        for L in emb_layers:
            x=L(x)
        Embed.append(x)
        for L in fc_layers:
            x=L(x)
            if L.activation==keras.activations.relu:
                states.append(tf.sign(x))
            elif L.activation==keras.activations.sigmoid:
                states.append(tf.round(x))
            else:
                raise ValueError('fc nonlinearity was not keras.a.relu or\
                                 k.a.sigmoid. Instead got:',L.activation)
        States.append(states)

    Embed=np.concatenate(Embed,axis=0)
    States=[np.concatenate(S,axis=0) for S in zip(*States)]
    Out['Y']=np.concatenate(Ys,axis=0)#0dim array
    #Out['Y']=np.stack(y,axis=0)
    Out['Embed']=Embed
    Out['States']=States
    Out['fc_layers']=fc_layers
    #return Embed,States,fc_layers
    return Out


if __name__=='__main__':
    #ckpt='./logs/Model_0724_155644_tf2_cpu_test/checkpoints/Model_ckpt.h5'




    ##Base Model##
#        keras.layers.Conv2D(64,3,activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        #keras.layers.Conv2D(128,3,2,activation='relu'),
#        keras.layers.Conv2D(128,3,activation='relu'),
#        keras.layers.MaxPool2D( (2,2) ),
#        keras.layers.Conv2D(64,3,activation='relu'),
#        keras.layers.Flatten(),
#        keras.layers.Dense(64,activation='relu'),
#        keras.layers.Dense(64,activation='relu'),


    #format:log_dir# n_supp vectors prob_agree net_acc


    ###FrogShip Cifar Models###[*] indicates used in paper

    #model -1 * Dense(64)
    #log_dir='./logs/Model_0804_141829_frogship_M2plus-1layers' #1991 .971 .968[*]
    #log_dir='./logs/Model_0804_151838_frogship_M2plus-1layers'  #2446 .970 .970

    #model  0 * Dense(64)
    #log_dir='./logs/Model_0803_155517_frogship_debug'         #2368 .971 .963
    #log_dir='./logs/Model_0804_151908_frogship_M2plus+0layers'#1217 .976 .964[*]

    #model +1 * Dense(64)
    #log_dir='./logs/Model_0804_163250_frogship_M2plus+1layers' #998 .988 .969
    #log_dir='./logs/Model_0804_163337_frogship_M2plus+1layers'  #727 .990 .967[*]


    #model +2 * Dense(64)
    #log_dir='./logs/Model_0803_170758_frogship_M2plus2layers'  #576 .992 .976[*]
    #log_dir='./logs/Model_0804_185041_frogship_M2plus+2layers'#379 .988 .97


    #model +3 * Dense(64)
    #log_dir='./logs/Model_0804_134351_frogship_M2plus3layers'   #866 .993 .972[*]
    #log_dir='./logs/Model_0804_185059_frogship_M2plus+3layers' #783 .992 .971


    #model +4 * Dense(64)
    #log_dir='./logs/Model_0803_164409_frogship_M2plus4layers'  #259 .995 .974[*]
    #log_dir='./logs/Model_0804_202228_frogship_M2plus+4layers'#143 .991 .966

    #model +5 * Dense(64)
    #log_dir='./logs/Model_0804_140704_frogship_M2plus5layers'  #737 .985 .970
    #log_dir='./logs/Model_0804_202532_frogship_M2plus+5layers'# Busted 0.5acc
    #log_dir='./logs/Model_0804_213951_frogship_M2plus+5layers' #184 .994 .971[*]
    #log_dir='./logs/Model_0804_215156_frogship_M2plus+5layers' #358 .990 .971
    #log_dir='./logs/Model_0804_220343_frogship_M2plus+5layers' #689 .970 .95#Bad.97trnacc
    #log_dir=''#TODO
    #log_dir=''#TODO


    #model +6 * Dense(64)
        ###+6 didnt work at all with the same learning rate
    #log_dir='./logs/Model_0804_143013_frogship_M2plus+6layers'#
    #log_dir='./logs/Model_0804_222442_frogship_M2plus+6layers'#1023 .984 .964#Bad.93trnacc
    #log_dir='./logs/Model_0804_223005_frogship_M2plus+6layers'#468 .987 .970

    ###smaller learning rate makes svm harder to fit??
    #log_dir='./logs/Model_0804_225520_frogship_M2plus+6layers_lr5e-4'#SVfail
    #log_dir='./logs/Model_0804_231638_frogship_M2plus+6layers_lr5e-4'#SVfail
    #log_dir='./logs/Model_0804_235541_frogship_M2plus+6layers_lr5e-4'#SVfail  #10epoch

    #log_dir=''#
    #log_dir=''#


#--------------------------------------------------#
    descrip,id_str=log_dir.split('_')[-1],str(file2number(log_dir))

    ckpt=os.path.join(log_dir,'checkpoints/Model_ckpt.h5')
    #checkpoint_dir=os.path.join(log_dir,'checkpoints')
    #model_file=os.path.join(checkpoint_dir,'Model_ckpt.h5')
    pactensor_dir=os.path.join(log_dir,'pac-tensor')
    if not os.path.exists(pactensor_dir):
        os.makedirs(pactensor_dir)
    results_fp=os.path.join(pactensor_dir,'results.json')

    #assert(not os.path.exists(results_fp))#not implemented 
    results={}



    datasets,info=load_frogs_vs_ships()
    db_bat=peak(datasets['train'].batch(20))
    train_data,test_data=tuple_splits(datasets)
    #train_data=train_data.shuffle(200)#doenst seem to affect

    #print("Warn. Not using all data")
    #train_data=train_data.take(1000)
    #test_data.take(200)

    train_data=train_data.batch(100)
    test_data=test_data.batch(100)


    model=keras.models.load_model(ckpt)


    print('loaded model:\n',ckpt)

    #Embed,States=get_embed(train_data,model)
    #train_Embed,train_States,fc_layers=get_embed(train_data,model)
    #test_Embed,test_States,fc_layers=get_embed(test_data,model)

    train_unpack=unpack(train_data,model)
    val_unpack=unpack(test_data,model)


    UP=train_unpack
    trnY=UP['Y']
    trn_netpred=np.round(UP['States'][-1].flatten( ))
    trn_netcorrect=(trn_netpred==trnY)

    vUP=val_unpack
    valY=vUP['Y']
    val_netpred=np.round(vUP['States'][-1].flatten())
    val_netcorrect=(val_netpred==valY)

    dplaces=4
    results['trn net acc']=np.round(np.mean(trn_netcorrect),dplaces)
    results['val net acc']=np.round(np.mean(val_netcorrect),dplaces)


    Biases=[L.bias for L in UP['fc_layers']]

    def phi(X,States):
        #Here States includes output
        return [X]+[np.array(b*S) for b,S in zip(Biases[:-1],States[:-1])]
    print('forming gram...')


    #Gram method#
    #phiX=phi(UP['Embed'],UP['States'])
    #phiX1,phiX2=phiX,phiX
    #lyr_kernels=[np.dot(s1,s2.T) for s1,s2 in zip(phiX1,phiX2)]
    #prod_kernels=[np.multiply.reduce(lyr_kernels[:i]) for i in range(len(lyr_kernels))]
    #gram=sum(prod_kernels)
    def get_gram(phiX1,phiX2):
        #phiX=phi(UP['Embed'],UP['States'])
        #phiX1,phiX2=phiX,phiX
        lyr_kernels=[np.dot(s1,s2.T) for s1,s2 in zip(phiX1,phiX2)]
        prod_kernels=[np.multiply.reduce(lyr_kernels[:i]) for i in range(len(lyr_kernels))]
        gram=sum(prod_kernels)
        return gram

    PhiX   =phi( UP['Embed'], UP['States'])
    ValPhiX=phi(vUP['Embed'],vUP['States'])

    gram    =get_gram(PhiX   ,PhiX)#for training
    val_gram=get_gram(ValPhiX,PhiX)#for prediction

    print('shapes:',gram.shape,'and',val_gram.shape)

    print('fitting model..')

    svm_models=[]
    ker_svc=SVC(kernel='precomputed',
               C=1e-5,
               tol=1e-5,
               shrinking=False,
               max_iter=500000,
              )
    #ker_svc.name='svc-precomputed'
    ker_svc.name='ker-svc'
    svm_models.append(ker_svc)

    ker_svc.fit(gram,trnY)
    ker_svc.trn_acc=np.round(ker_svc.score(gram,trnY),dplaces)
    ker_svc.trn_correct=(ker_svc.predict(gram)==trnY)
    ker_svc.wh_wrong=np.where(~ ker_svc.trn_correct)[0]



    #print('  ..finished ker-svc')

### Code works but model not used ###
#    lib_lin=LinearSVC(loss='hinge',
#                  C=1e-5,
#                  tol=1e-8,
##                  dual=True,
##                  max_iter=20000,
#                  #penalty='l1',
#                 )
#    #lib_lin.name='LinearSVC'
#    lib_lin.name='lib-lin'
#
#    lib_lin.fit(gram,trnY)
#    lib_lin.trn_acc=np.round(lib_lin.score(gram,trnY),dplaces)
#    lib_lin.trn_correct=(lib_lin.predict(gram)==trnY)
#    lib_lin.wh_wrong=np.where(~ lib_lin.trn_correct)[0]
#    svm_models.append(lib_lin)
#
#    print(lib_lin.name)
#    print('\t trn acc:',lib_lin.trn_acc)
#    #print('\t n_supp :',lib_lin.n_support_)


    print('comparing val..')

    ker_svc.val_pred   =ker_svc.predict(val_gram)
    ker_svc.val_correct=(ker_svc.val_pred==valY)
    ker_svc.val_acc    =np.round(np.mean(ker_svc.val_correct),dplaces)

    val_agree = (ker_svc.val_correct == val_netcorrect )

    results['svc trn acc']      =ker_svc.trn_acc
    results['svc val acc']      =ker_svc.val_acc
    results['svc num sv' ]      =ker_svc.n_support_.sum()
    results['svc frac sv']=(ker_svc.n_support_.sum())/float(gram.shape[0])
    results['val svc-net agree']=np.round(np.mean(val_agree),dplaces)
    #cmpr with val_netcorrect

    print('net model')
    print('\t trn acc: ', results['trn net acc'])
    print('\t val acc: ', results['val net acc'])

    print(ker_svc.name)
    print('\t trn acc:',ker_svc.trn_acc)
    print('\t val acc:',ker_svc.val_acc)
    print('\t num sv :',ker_svc.n_support_,'=', ker_svc.n_support_.sum())
    print('\t frac sv:',results['svc frac sv'])

    print('Val score same on average:',results['val svc-net agree'])
    #print('\t net_score: ',np.round(np.mean(     val_netcorrect),dplaces))
    #print('\t svc score: ',np.round(np.mean(ker_svc.val_correct),dplaces))



    with open(results_fp, 'w') as fp:
        json.dump(results, fp, indent=4, sort_keys=True)



#        clf.fit(gram,Y)
#        print '..fit model',clf.name
#        clf.acc=np.round(clf.score(gram,Y),dplaces)
#        clf.correct=(clf.predict(gram)==Y)
#        clf.wh_wrong=np.where(~ clf.correct)[0]


    #print('fit successfully')






#------------
#from sklearn import svm
#from sklearn.svm import LinearSVC
#from sklearn.linear_model import SGDClassifier
#    weights=[[w[trn_iter],b[trn_iter]] for w,b in step_dweights]#d*2*wtshape
#    Wweights,Bweights=zip(*weights)
#
#
#    get_states=get_state_fn(weights)
#    get_gram=get_gram_fn(Bweights,get_states)
#    X1,X2=X,X
#    gram=get_gram(X1,X2)
#
#
#
#    clf_precom=svm.SVC(kernel='precomputed',
#               #C=1e-2,
#               C=1e-5,
#               #C=1e5,
#               tol=1e-5,
#               shrinking=False,
#               #cache_size=300,#MB
#               max_iter=500000,
#              )
#    clf_precom.name='svc-precomputed'
#
#
#    for clf in models[1:]:#skip NN
#        clf.fit(gram,Y)
#        print '..fit model',clf.name
#        clf.acc=np.round(clf.score(gram,Y),dplaces)
#        clf.correct=(clf.predict(gram)==Y)
#        clf.wh_wrong=np.where(~ clf.correct)[0]

 #--------------
    #test_Embed,test_States,fc_layers=get_embed(test_data,model)
    #train_StatesY=train_States[-1]
    #test_StatesY=test_States[-1]
    #train_Pred=model.predict(train_data)
    #test_Pred=model.predict(test_data)

#    #Debug
#    train_S=train_data.take(15)
#    #train_S=train_data.take(-1)
#    pred_S=np.round(model.predict(train_S))
#    emb_S,sta_S,fc_layers=get_embed(train_S,model)
#    staY_S=sta_S[-1]
#    print( np.mean(pred_S==staY_S) )

