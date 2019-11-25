# find process id with psutil
# https://thispointer.com/python-check-if-a-process-is-running-by-name-and-find-its-process-id-pid/
import psutil

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

# read memory with ptrace
# https://www.ownedcore.com/forums/world-of-warcraft/world-of-warcraft-bots-programs/wow-memory-editing/330612-linux-wine-reading-memory-ptrace.html 
from ptrace.debugger.debugger import PtraceDebugger
from ptrace.debugger.process import PtraceProcess
from struct import unpack

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

class Env :

    def __init__(self, pid) :
        # self.reader = memory_reader(findProcessIdByName('volleyball'))
        self.reader = memory_reader(pid)
        self.caculate_addresses()
        self.action_space = {0: c.release,1: c.left,2: c.right,3: c.up,4: c.down,5: c.p}
        self.state_space_num = 16
        self.action_space_num = 6
        self.cur_state = np.array([ 36, 244, 396, 244, 56, 0 ])
        self.pre_state = self.cur_state
        self.time_stamp = time.time()
        self.score1 = 0
        self.score2 = 0
        self.my_ball = False
        print("envrionment is ready!")

    def __del__(self) :
        print("envrionment deleted")

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
        self.player1_y_address = 0
        while self.player1_y_address == 0 :
            self.player1_y_address = self.reader.search_address(0x5a0000, 0x5b0000, 396, 244)
        self.player1_x_address = self.player1_y_address + 0x4
        self.base = self.player1_y_address - (self.player1_y_address & 0xffff)
        self.player2_y_address = self.reader.search_address(self.base, self.base + 0xffff, 36, 244)
        self.player2_x_address = self.player2_y_address + 0x4
        self.ball_y_address = self.reader.search_address(self.base + 0x6000, self.base + 0xffff, 56, 0)
        self.ball_x_address = self.ball_y_address + 0x4
        # self.flag_address = self.reader.search_address(self.base + 0xe00, self.base + 0xf00, 0, 15) - 0x4
        self.flag_address = self.player2_y_address - 0xc0    
        self.player1_score_address = self.flag_address - 0x8
        self.player2_score_address = self.player1_score_address - 0x4   

    def init(self) :
        c.release()
        if self.my_ball == True :
            self.cur_state = np.array([ 36, 244, 396, 244, 376, 0 ])
        else :
            self.cur_state = np.array([ 36, 244, 396, 244, 56, 0 ])
        self.pre_state = self.cur_state

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

    def state(self) :
        # state = [sub_state, vector, key_map]
        # sub_state = [p2.y, p2.x, p1.y, p1.x, b.y, b.x]
        # vector = sub_state - sub_state_before_0.1s
        # key_map = [left, right, up, down]
        if time.time() - self.time_stamp > 0.1 :
            self.pre_state = self.cur_state
            self.cur_state = self.sub_state()
            vector = self.cur_state - self.pre_state
            self.time_stamp = time.time()
            return np.concatenate((self.pre_state, vector, c.get_key_map()), axis=None), True
        else :
            return np.zeros(16), False

    def flag(self) :
        # 0 : a game start,  1 : ball start, 2 : playing ball, 3 : ball end, 4 : game set, 10 : begin screen
        return self.reader.read_bytes(self.flag_address)

    def step(self, action) :
        # action = [release, left, right, up, down, power]
        self.action_space[action]()

    def score(self) :
        score1 = self.reader.read_bytes(self.player1_score_address)
        score2 = self.reader.read_bytes(self.player2_score_address)

        if self.score1 != score1 :
            self.score1 = score1
            print("win")
            self.my_ball = True
            return 1
        elif self.score2 != score2 :
            self.score2 = score2
            print("lose")
            self.my_ball = False
            return -1
        else :
            return 0

    def final_score(self) : 
        score = self.score1 - self.score2
        print("final score is %d:%d"%(self.score2, self.score1))
        self.score1 = 0
        self.score2 = 0
        return score

# test environment
if __name__ == "__main__":
    env = Env()
    while True :
        # # test flag
        # print(env.flag())

        # test state
        state, got_state = env.state()
        if got_state :
            print(state)

