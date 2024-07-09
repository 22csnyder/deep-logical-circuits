import os
import sys
import argparse
from subprocess import call

parser=argparse.ArgumentParser()
parser.add_argument('--port',type=int,default=6006,
                    help='''the port to host the tensorboard on''')
parser.add_argument('--logdir',type=str,default='',
                    help='''what data to host''')

def file2number(fname):
    nums=[s for s in fname.split('_') if s.isdigit()]
    if len(nums)==0:
        nums=['0']
    number=int(''.join(nums))
    return number

if __name__=='__main__':
    cmd_args=parser.parse_args()
    port=cmd_args.port

    root='./logs'

    if not cmd_args.logdir:
        logs=os.listdir(root)
        logs.sort(key=lambda x:file2number(x))


        logdir=os.path.join(root,logs[-1])
    else:
        logdir=cmd_args.logdir



    print 'running tensorboard on logdir:',logdir

    call(['tensorboard', '--logdir',logdir,'--port',str(port)])

