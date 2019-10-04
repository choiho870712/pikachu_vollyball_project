from env import Env
from actor import Actor, Trajectory
import numpy as np
from pynput.keyboard import Key, Controller
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keyboard = Controller()
env = Env()
actor = Actor(env.state_space_num, env.action_space_num, device)
trajectory = Trajectory(10000)
i_episode = 0
TARGET_UPDATE = 1
pushed = False
while True:
    flag = env.flag()

    if flag == 0 or flag == 1 : # start game
        env.release_key() # release all key

    elif flag == 2 : # if in game, action
        state = env.state()
        action = actor.select_action(state) # Select an action
        env.step(action) # step
        trajectory.push(state, action) # Store the state and action in trajectory
        pushed = False

    elif flag == 3 : # if win or loss, Store the trajectory in replay memory
        if not pushed :
            env.release_key() # release all key
            actor.push_trajectory(trajectory)
            pushed = True
            print("rest time")

    elif flag == 4 : # optimize model when rest time
        if not pushed :
            env.release_key() # reset key
            actor.push_trajectory(trajectory)
            i_episode += 1
            pushed = True
            print("gameset")

    elif flag == 10 : # begin screen
        pass # need auto next game

    # Update the target network, copying all weights and biases in DQN
    actor.optimize_model()
    if i_episode % TARGET_UPDATE == 0:
        actor.update_model()
        i_episode = 0