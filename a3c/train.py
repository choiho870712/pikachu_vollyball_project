import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
#from torch.autograd import Variable

from envs import dmlab_env
from model import ActorCritic


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    ## TODO ##
    env = dmlab_env(env_id=args.env_name, t_type=args.type, rank=rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    ## TODO END ##

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset(seed = args.seed + rank)
    state = torch.from_numpy(state).float()
    done = True

    episode_length = 0
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            ## TODO ## Match your hidden layer size in model.py.
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
            ## TODO END ##
        else:
            cx = cx.data
            hx = hx.data

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit,dim=1)
            log_prob = F.log_softmax(logit,dim=1)
            #print('log_prob size:', log_prob.size())
            
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, action)

            state, reward, done = env.step(action.numpy()[0, 0])
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()#seed=args.seed)
                if reward == 0:
                    reward = -1

            state = torch.from_numpy(state).float()
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.data

        values.append(R)
        policy_loss = 0
        value_loss = 0
        
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
        
        if counter.value > args.cnt:
            sys.exit()
