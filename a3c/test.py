import time
import os
import sys
from os.path import join
from collections import deque

import torch
import torch.nn.functional as F

from envs import dmlab_env
from model import ActorCritic

def test(rank, args, shared_model, counter):
    SAVE_FREQ = args.save_freq
    torch.manual_seed(args.seed + rank)

    ## TODO ##
    env = dmlab_env(env_id=args.env_name)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    ## TODO END ##

    model.eval()

    state = env.reset(seed=args.seed + rank)
    state = torch.from_numpy(state).float()
    reward_sum = 0
    done = True
    
    if args.save_path == None:
        model_dir = join('saved_model/', args.env_name)
    else:
        model_dir = args.save_path

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if args.log == None:
        log_path = join('log/', args.env_name+'.csv')
    else:
        log_path = args.log
    print('Time (HMS), num steps, FPS, episode reward, episode length', file=open(log_path, "a"))


    start_time = time.time()
    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    episode = 0
    save_cnt = -1
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            ## TODO ## Match your hidden layer size in model.py.
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            ## TODO END ##

        value, logit, (hx, cx) = model((
            state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=1)
        action = prob.max(1, keepdim=True)[1].data.numpy()

        state, reward, done = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True
            
        if done and reward_sum == 0:
            reward_sum = -1.0
        
        if done:
            #Time (HMS), num steps, FPS, episode reward, episode length
            print("{}, {}, {:.0f}, {}, {}".format(
                time.strftime("%H:%M:%S",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length), file=open(log_path, "a"))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()

            if save_cnt != counter.value//SAVE_FREQ:
                save_cnt = counter.value//SAVE_FREQ
                torch.save(model.state_dict(),join(model_dir,'model_%03d.pth'%save_cnt))
            if counter.value > args.cnt:
                sys.exit()

            ## TODO ## test every 60 sec
            time.sleep(60)
            ## TODO END ##

        state = torch.from_numpy(state).float()