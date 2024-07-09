from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import os
import json

def str2bool(v):
    return v is True or v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

data_arg=add_argument_group('Data')
misc_arg=add_argument_group('Misc')
model_arg=add_argument_group('Model')
train_arg = add_argument_group('Training')
exp_arg = add_argument_group('Experiment')


#data_arg.add_argument('--dataset',type=str,choices=['mnist','cifar10'],default='mnist', help='''what dataset to run on''')

#data_arg.add_argument('--dataset',type=str, help='''what dataset to run on''')

#data_arg.add_argument('--num_train_samples',type=int,
#                      help='''how many training samples to use''')
model_arg.add_argument('--batch_size',type=int,default=100)#32)

model_arg.add_argument('--arch',type=int,choices=[1,11,12,2,21,22,3])
#model_arg.add_argument('--pca',type=int,
#                      help='''Add a linear layer of size config.pca between the
#                       input and the network architecture.''')

model_arg.add_argument('--load_path',type=str,default='')
#model_arg.add_argument('--network',type=str,choices=['lenet','fcmnist','vgg'],
#                      help='''Which model to use''')

#model_arg.add_argument('--nonlinearity',type=str,choices=['lrelu','relu','tanh','abs'],
#model_arg.add_argument('--nonlinearity',type=str, default='relu',
#                       help='''the primary nonlinearity to use between
#                       layers''')

#model_arg.add_argument('--width1',type=int,
#                       help='''width of first hidden layer''')

model_arg.add_argument('--init_stdev1',type=float,
                       help='''stddev of 1st layer multiplicative weights at
                       initialization''')

misc_arg.add_argument('--load_config',type=str2bool,default=False,
                      help='''whether to adopt the values from the previous
                      model by default''')
misc_arg.add_argument('--is_train',type=str2bool,
                      help='''if we are training, what are we training?''')
#misc_arg.add_argument('--is_eval',type=str2bool,help='''are we evaluating''')

misc_arg.add_argument('--log_dir',type=str,default='./logs',
                     help='''where logs of model are kept''')
misc_arg.add_argument('--descrip',type=str,default='',
                     help='''a small string added to the end of the
                     model folder for future reference''')

misc_arg.add_argument('--prefix',type=str,default='Model',
                     help='''a small string added to the beginning of the
                     model folder for future reference''')


###Kind of hard to pick reasonable numbers for these across models
#train_arg.add_argument('--num_iter',type=int,default=3000,
#                       help='''number of steps to run the experiment for unless
#                       stopped by some other termination condition''')

#train_arg.add_argument('--no_bias',type=str2bool,default=False,
#                       help='''Use Wx instead of Wx+b in last map''')

#train_arg.add_argument('--log_every_n_steps'  ,type=int,default = 10,
#                       help='''how frequently to report to console''')
#train_arg.add_argument('--save_every_n_steps'  ,type=int,default = 1000,
#                       help='''how frequently to report save model''')

#train_arg.add_argument('--validation_every_n_steps' ,type=int,default = 2000,
#                       help='''how frequently to do evaluation''')

#Adam seems to help fit data much easier
#train_arg.add_argument('--optimizer',type=str,default='adam',choices=['grad_descent','adam'])
#train_arg.add_argument('--optimizer',type=str,choices=['grad_descent','adam'])

#train_arg.add_argument('--learning_rate',type=float,default=0.0005)

#train_arg.add_argument('--dropout_keep_prob',type=float,default=1.0,
#                      help='''what fraction of inputs to keep''')

#exp_arg.add_argument('--experiment',type=str,choices=['skeleton','sign','supvec'],
#                     help='''which experiment file to run''')

def load_config(config):
    #loads config from previous model
    cf_path=os.path.join(config.load_path,'params.json')
    print('Attempting to load params from: ',cf_path)
    with open(cf_path,'r') as f:
        load_config=json.load(f)
    return load_config


def get_config():
    config, unparsed = parser.parse_known_args()

    #dataset_dir=os.path.join('data',config.dataset)
    #setattr(config, 'dataset_dir', dataset_dir )

    #if config.is_eval and not config.load_path:
    #    raise ValueError('must load path to evaluate')

    if len(unparsed)>0:
        print('WARNING, there were unparsed arguments:',unparsed)

    dont_adopt=['model_dir','model_name','descrip','is_train','is_eval','load_path','dataset','load_config']

    if config.load_config:
        print('WARNING: experimental loading of previous model config')
        prev_config=load_config(config)
        old_keys,update_keys=[],[]
        for k in prev_config.keys():
            if k not in dont_adopt:
                if k not in config.__dict__.keys():
                        old_keys.append(k)
                elif config.__dict__[k]!=prev_config[k]:
                        update_keys.append(k)
        if len(old_keys)>0:
            old_dict={k:prev_config[k] for k in old_keys}
            print( "WARN: keys in previous model not in current model:",old_dict)
            #config.__dict__.update(old_dict) #not unless old code is used#not implemented
        if len(update_keys)>0:
            update_dict={k:prev_config[k] for k in update_keys}
            print("WARN: overwriting config with value from prev model:",update_dict)
            config.__dict__.update(update_dict)

    return config, unparsed


if __name__=='__main__':
    config,unparsed=get_config()


