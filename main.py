from env import Env
from actor import Actor, Trajectory
import numpy as np
from pynput.keyboard import Key, Controller
from collections import namedtuple
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keyboard = Controller()
env = Env()
actor = Actor(env.state_space_num, env.action_space_num, device)
trajectory = Trajectory(1000)
i_episode = 0
TARGET_UPDATE = 1
pushed = False
while True:
    flag = env.flag()

    if flag == 0 or flag == 1 : # start game
        env.release_key() # release all key
        state = env.state()

    elif flag == 2 : # if in game, action
        print(state)
        action = actor.select_action(state) # Select an action
        next_state, reward = env.step(action) # step
        trajectory.push(state, action) # Store the state and action in trajectory
        state = next_state
        pushed = False

    elif flag == 3 : # if win or loss, Store the trajectory in replay memory
        if not pushed :
            env.release_key() # release all key
            actor.push_trajectory(trajectory, reward)
            pushed = True
            print("rest time, reward = %d"%reward)

    elif flag == 4 : # optimize model when rest time
        if not pushed :
            env.reset() # reset key and score
            actor.push_trajectory(trajectory, reward)
            i_episode += 1
            pushed = True
            print("gameset, reward = %d"%reward)

    elif flag == 10 : # begin screen
        pass

    # Update the target network, copying all weights and biases in DQN
    actor.optimize_model()
    if i_episode % TARGET_UPDATE == 0:
        actor.update_model()
        i_episode = 0