from env import Env

env = Env()
observation = env.get_observation_space()
while True :
    if env.is_running() :
        action = env.get_random_action()
        observation, reward, done = env.step(action)

        if reward == 1 :
            print("win")
        elif reward == -1 :
            print("lose")
    else :
        env.reset()