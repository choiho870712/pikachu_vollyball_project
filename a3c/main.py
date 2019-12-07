from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from envs import dmlab_env
from model import ActorCritic
from test import test
from train import train

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=8,
                    help='how many training processes to use (default: 8)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=600,
                    help='maximum length of an episode (default: 600)')
parser.add_argument('--env-name', default='sk',
                    help='environment to train on (default: sk)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum (default=False)')
parser.add_argument('--save-path', default=None,
                    help='model save path (default: None)')
parser.add_argument('--log', default=None,
                    help='log save path (default: None)')
parser.add_argument('--pre-model', default=None,
                    help='pretrained a3c (default: None)')
parser.add_argument('--cnt', type=int, default=7500000,
                    help='max count of steps (default: 7.5 million)')
parser.add_argument('--save-freq', type=int, default=500000,
                    help='save frequent (default: 0.5 million)')


if __name__ == '__main__':
    # add for ubuntu cpu
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    mp.set_start_method("spawn")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    ## TODO ##
    env = dmlab_env(env_id=args.env_name)
    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    ## TODO END ##

    if args.pre_model!=None:
        shared_model.load_state_dict(torch.load(args.pre_model))
        print('pretrained model %s loaded' %(args.pre_model))
    
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    processes = []

    counter = mp.Value('i', 0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, counter))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, counter, lock, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
