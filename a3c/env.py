# read memory with ptrace
# https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/330612-linux-wine-reading-memory-ptrace.html 
from ptrace.debugger.debugger import PtraceDebugger
from ptrace.debugger.process import PtraceProcess

class memory_reader :

    def __init__(self, pid) :
        self.tracer = PtraceProcess(PtraceDebugger(), pid, True)

    def read_bytes(self, address, size = 1) :
        return int.from_bytes(self.tracer.readBytes(address, size), byteorder='little')

    def search_address(self, low_address, high_address, target_value1, target_value2) :
        addr = low_address
        value = self.read_bytes(addr, 4)
        while addr < high_address :
            addr += 0x4
            pre_value = value
            value = self.read_bytes(addr, 4)
            if pre_value == target_value1 and value == target_value2 :
                return addr - 0x4

        return 0

import time
import control as c
import numpy as np
import subprocess
from pynput.keyboard import Key, Controller

class Env :

    def __init__(self) :
        self.keyboard = Controller()
        self.open_new_game()
        self.time_stamp = time.time()
        self.score1 = 0
        self.score2 = 0
        self.is_my_ball = False
        self.is_reset = True
        self.cur_state = np.array([ 36, 244, 396, 244, 56, 0 ])
        self.pre_state = self.cur_state
        self.action_id = {0: c.release,1: c.left,2: c.right,3: c.up,4: c.down,5: c.p}
        self.action_space = np.arange(len(self.action_id))
        print("envrionment is ready!")

    def open_new_game(self) :
        process = subprocess.Popen(["wine","volleyball.exe"])
        print("game opened")
        self.auto_start_new_game()
        self.reader = memory_reader(process.pid)
        self.caculate_addresses()

    def auto_start_new_game(self) :
        press_count = 0
        time.sleep(2)
        while press_count < 4 : 
            self.keyboard.press(Key.enter)
            time.sleep(0.2)
            self.keyboard.release(Key.enter)
            time.sleep(1)
            press_count += 1

        print("game auto started")

    def caculate_addresses(self) :
        # check memory value with scanmem
        # read memory with ptrace
        # https://www.unknowncheats.me/forum/other-software/151529-scanmem-value-editing-installation-tutorial-linux-2-a.html
        # https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection
        # player2.y   // init = 36  // range = 32  ~ 184
        # player2.x   // init = 244 // range = 108 ~ 244
        # player1.y   // init = 396 // range = 248 ~ 400
        # player1.x   // init = 244 // range = 108 ~ 244
        # ball.y      // init = 56  // range = 20  ~ 430
        # ball.x      // init = 0   // range = 0   ~ 252
        # flag        // 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
        self.base = 0x5a0000
        self.player1_y_address = self.reader.search_address(self.base, self.base + 0xffff, 396, 244)
        self.player1_x_address = self.player1_y_address + 0x4
        self.player2_y_address = self.reader.search_address(self.base, self.base + 0xffff, 36, 244)
        self.player2_x_address = self.player2_y_address + 0x4
        self.ball_y_address = self.reader.search_address(self.base + 0x6000, self.base + 0xffff, 56, 0)
        self.ball_x_address = self.ball_y_address + 0x4
        self.flag_address = self.player2_y_address - 0xc0    
        self.player1_score_address = self.flag_address - 0x8
        self.player2_score_address = self.player1_score_address - 0x4 

    def get_random_action(self):
        return np.random.choice(self.action_space, 1)[0]  

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        if self.is_running():
            return self.observations()
        else:
            return self.reset()

    def observations(self):
        # state = [sub_state, vector, key_map]
        # sub_state = [p2.y, p2.x, p1.y, p1.x, b.y, b.x]
        # vector = sub_state - sub_state_before_0.1s
        # key_map = [left, right, up, down]
        if self.is_running():
            while time.time() - self.time_stamp < 0.1 :
                pass
            self.pre_state = self.cur_state
            self.cur_state = self.sub_state()
            self.time_stamp = time.time()
            self.is_reset = False
        else :
            if self.is_my_ball :
                self.cur_state = np.array([ 36, 244, 396, 244, 376, 0 ])
            else :
                self.cur_state = np.array([ 36, 244, 396, 244, 56, 0 ])
            self.pre_state = self.cur_state

        vector = self.cur_state - self.pre_state
        state = np.concatenate((self.pre_state, vector, c.get_key_map()), axis=None)
        return self.normalization(state)

    def normalization(self, state) :
        normal_state = []
        normal_state.append((state[0] - 108) / 76)
        normal_state.append((state[1] - 176) / 68)
        normal_state.append((state[2] - 324) / 76)
        normal_state.append((state[3] - 176) / 68)
        normal_state.append((state[4] - 216) / 216)
        normal_state.append((state[5] - 126) / 126)
        normal_state.append(state[6] / 24)
        normal_state.append(state[7] / 45)
        normal_state.append(state[8] / 24)
        normal_state.append(state[9] / 45)
        normal_state.append(state[10]/ 60)
        normal_state.append(state[11]/ 90)
        normal_state.append(state[12] * 2 - 1)
        normal_state.append(state[13] * 2 - 1)
        normal_state.append(state[14] * 2 - 1)
        normal_state.append(state[15] * 2 - 1)
        return np.array(normal_state)

    def is_running(self):
        if self.flag() == 2 :
            return True
        else :
            return False

    def reset(self):
        # set "self.is_reset = False" in observations() when "self.is_running() == True" 
        if not self.is_reset :
            flag = self.flag()
            if flag == 3:
                c.release()
                self.is_reset = True

            elif flag == 4:
                c.release()
                print("final score is %d:%d"%(self.score2, self.score1))
                self.score1 = 0
                self.score2 = 0
                self.is_my_ball = False
                self.is_reset = True

                # sometimes page fault will let the game shut down
                # killing winedbg can close the game
                # then restart the environment by try except
                time.sleep(1)
                subprocess.run(["killall", "-9", "winedbg"])
                try :
                    time.sleep(1)                  # wait the game close if winedbg is killed
                    self.flag()                    # try to get the flag, check the game is working or not
                    self.auto_start_new_game()     # if the game is not closed, auto start
                except :                           # else reopen the game
                    self.open_new_game()

        return self.observations()

    def step(self, action):
        if action not in self.action_space:
            raise ValueError('Invalid action: ', action)

        self.action_id[action]() # convert to real action representation
        observation = self.observations()
        reward = self.score()
        done = not self.is_running()
        
        return observation, reward, done

    def sub_state(self) :
        # sub_state = [p2.y, p2.x, p1.y, p1.x, b.y, b.x]
        s = []
        s.append(self.reader.read_bytes(self.player2_y_address))
        s.append(self.reader.read_bytes(self.player2_x_address))
        s.append(self.reader.read_bytes(self.player1_y_address,2))
        s.append(self.reader.read_bytes(self.player1_x_address))
        s.append(self.reader.read_bytes(self.ball_y_address))
        s.append(self.reader.read_bytes(self.ball_x_address))
        return np.array(s)

    def flag(self) :
        # 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
        return self.reader.read_bytes(self.flag_address)

    def score(self) :
        score1 = self.reader.read_bytes(self.player1_score_address)
        score2 = self.reader.read_bytes(self.player2_score_address)

        if self.score1 != score1 :
            self.score1 = score1
            self.is_my_ball = True
            return 1
        elif self.score2 != score2 :
            self.score2 = score2
            self.is_my_ball = False
            return -1
        else :
            return 0
