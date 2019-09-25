# https://thispointer.com/python-check-if-a-process-is-running-by-name-and-find-its-process-id-pid/
import psutil

class psutil_api :

    def findProcessIdByName(processName):
        listOfProcessObjects = []

        #Iterate over the all the running process
        for proc in psutil.process_iter():
           try:
               pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
               # Check if process name contains the given name string.
               if processName.lower() in pinfo['name'].lower() :
                   listOfProcessObjects.append(pinfo)
           except (psutil.NoSuchProcess, psutil.AccessDenied , psutil.ZombieProcess) :
               pass
    
        return listOfProcessObjects[0]['pid']


# https://code.activestate.com/recipes/576362-list-system-process-and-process-information-on-win/
# https://docs.microsoft.com/zh-tw/windows/win32/api/tlhelp32/nf-tlhelp32-module32first
from ctypes import c_long , c_int , c_uint , c_char , c_ubyte , c_char_p , c_void_p , c_ulong
from ctypes import windll
from ctypes import byref

class win32_api :

    def __init__(self, pid) :
        self.OpenProcess = windll.kernel32.OpenProcess
        self.CloseHandle = windll.kernel32.CloseHandle
        self.ReadProcessMemory = windll.kernel32.ReadProcessMemory
        self.buffer = c_char_p(b"0")
        self.bufferSize = len(self.buffer.value)
        self.bytesRead = c_ulong(0)
        self.processHandle = self.OpenProcess(0xFFF, False, pid)

    def memoryRead(self, address) :
        self.ReadProcessMemory(self.processHandle, address, self.buffer, self.bufferSize, byref(self.bytesRead))
        return int.from_bytes(self.buffer.value, byteorder='little')


from pynput.keyboard import Key, Controller
import numpy as np

class Env :

    def __init__(self) :
        self.win32_api_handler = win32_api(psutil_api.findProcessIdByName("volleyball"))
        head_address = self.head_address()
        self.player2_score_address = head_address + 0x0e1c
        self.player1_score_address = head_address + 0x0e20
        self.player2_y_address = head_address + 0x0ee8
        self.player2_x_address = head_address + 0x0eec
        self.player1_y_address = head_address + 0x5310
        self.player1_x_address = head_address + 0x5314
        self.ball_y_address = head_address + 0x64e0
        self.ball_x_address = head_address + 0x64e4
        self.flag_address = head_address + 0x0e28
        self.keyboard = Controller()
        self.reset()
        print("done init")

    def head_address(self) :
        addr = 0x2000ee8
        while addr < 0x3000000 :
            value = self.win32_api_handler.memoryRead(addr)
            if value == 36 :
                return addr - 0x0ee8

            addr += 0x10000

        print("head_address error")
        return 0x400000

    def reset(self):
        self.keyboard.release(Key.up)
        self.keyboard.release(Key.down)
        self.keyboard.release(Key.left)
        self.keyboard.release(Key.right)
        self.keyboard.release(Key.enter)
        self.player1_score = 0
        self.player2_score = 0

    def state(self) :
        s = []
        s.append(self.win32_api_handler.memoryRead(self.player2_y_address))
        s.append(self.win32_api_handler.memoryRead(self.player2_x_address))
        s.append(self.win32_api_handler.memoryRead(self.player1_y_address))
        s[-1] += self.win32_api_handler.memoryRead(self.player1_y_address+1) << 8
        s.append(self.win32_api_handler.memoryRead(self.player1_x_address))
        s.append(self.win32_api_handler.memoryRead(self.ball_y_address))
        s[-1] += self.win32_api_handler.memoryRead(self.ball_y_address+1) << 8
        s.append(self.win32_api_handler.memoryRead(self.ball_x_address))
        return np.array(s)

    def reward(self) :
        p1_score = self.win32_api_handler.memoryRead(self.player1_score_address)
        p2_score = self.win32_api_handler.memoryRead(self.player2_score_address)

        # caculate reward
        if p1_score != self.player1_score :
            self.player1_score = p1_score
            return 1
        elif p2_score != self.player2_score :
            self.player2_score = p2_score
            return -1
        else :
            return 0

    def flag(self) :
        return self.win32_api_handler.memoryRead(self.flag_address)

    def step(self, action) :
        # up
        if action[0] > 0.5:
            self.keyboard.press(Key.up)
        else :
            self.keyboard.release(Key.up)

        # down
        if action[1] > 0.5:
            self.keyboard.press(Key.down)
        else :
            self.keyboard.release(Key.down)

        # left
        if action[2] > 0.5:
            self.keyboard.press(Key.left)
        else :
            self.keyboard.release(Key.left)

        # right
        if action[3] > 0.5:
            self.keyboard.press(Key.right)
        else :
            self.keyboard.release(Key.right)

        # enter
        if action[4] > 0.5:
            self.keyboard.press(Key.enter)
        else :
            self.keyboard.release(Key.enter)

    def get_screen(self) :
        return self.state(), self.reward(), self.flag()

Transition_trajectory = namedtuple('Transition_trajectory', ('state', 'action'))
class Trajectory(object) :

    def __init__(self, capacity) :
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[self.position] = Transition_trajectory(*args)
            self.position = self.position + 1

    def pop(self) :
        if self.position > 0 :
            self.position = self.position - 1
            return self.memory[self.position]
        else :
            return None

    def clear(self) :
        del self.memory
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Actor(object) :

    def __init__(self, screen_height, screen_width, n_actions, device) :
        self.steps_done = 0
        self.screen_height = screen_height
        self.screen_width = screen_width
        self.n_actions = n_actions
        self.device = device
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.BATCH_SIZE = 128
        self.GAMMA = 0.999
        self.policy_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net = DQN(self.screen_height, self.screen_width, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.replay_memory = ReplayMemory(10000)

    def update_model(self) :
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push_trajectory(self, trajectory, reward) :
        sa_1 = trajectory.pop()
        sa_2 = trajectory.pop()
        discount = 0.99
        while ( sa_1 != None and sa_2 != None ) :
            torch_reward = torch.tensor([reward], device=self.device, dtype=torch.float)
            self.replay_memory.push(sa_2.state, sa_2.action, sa_1.state, torch_reward)
            reward = reward * discount
            if reward > -0.1 and reward < 0 :
                reward = 0.1
                discount = 1.01
            elif reward > 1 :
                discount = 1
            sa_1 = sa_2
            sa_2 = trajectory.pop()

        trajectory.clear()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 0.01
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.replay_memory) < self.BATCH_SIZE:
            return
        transitions = self.replay_memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


env = Env()
state, reward, flag = env.get_screen()

actor = Actor(np.shape(state)[2], np.shape(state)[3], 6, device)
trajectory = Trajectory(1000)

keyboard = Controller()

i_episode = 0
TARGET_UPDATE = 2
while "Screen capturing":
    # get screen
    state, reward, gaming_status = env.get_screen()

    # show state
    cv2.imshow("show", cv2.resize(state.cpu().squeeze(0).squeeze(0).numpy(), (200, 200)))
    cv2.waitKey(50)

    # if in game, action
    if gaming_status == "gaming" :
        isPressed = False

        # Select and perform an action
        action = actor.select_action(state)
        env.step(action)

        # Store the state and action in trajectory
        trajectory.push(state, action)

    # if win or loss, Store the trajectory in replay memory
    elif reward != 0 :
        actor.push_trajectory(trajectory, reward)
        i_episode += 1

    # optimize model when rest time
    elif gaming_status == "gameset" :
        env.reset() # release all key
        actor.optimize_model()

    elif gaming_status == "waiting_for_start" :
        keyboard.press(Key.enter)
        time.sleep(0.5)
        keyboard.release(Key.enter)

    elif gaming_status == "next_stage" :
        env.reset() # release all key

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        actor.update_model()
