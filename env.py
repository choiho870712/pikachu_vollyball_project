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

import control as c
import numpy as np
import time

class Env :

    def __init__(self) :
        self.win32_api_handler = win32_api(findProcessIdByName("volleyball"))
        self.caculate_addresses()
        self.action_space = {0: c.release,1: c.left,2: c.right,3: c.up,4: c.down,5: c.p}
        self.state_space_num = 16
        self.action_space_num = 6
        self.reset()
        print("envrionment is ready!")
        print(self.state())

    def caculate_addresses(self) :
        # find player1's (x,y)
        addr = 0x2000000
        max_addr = 0x5000000
        value = self.win32_api_handler.memoryRead(addr)
        while addr < max_addr :
            addr += 0x4
            pre_value = value
            value = self.win32_api_handler.memoryRead(addr)
            if pre_value == 140 and value == 244 :
                self.player1_y_address = addr - 0x4
                self.player1_x_address = addr
                break

        # find player2's (x,y) , score , flag
        addr -= addr & 0xffff # mask
        max_addr = addr + 0x10000
        value = self.win32_api_handler.memoryRead(addr)
        while addr < max_addr :
            addr += 0x4
            pre_value = value
            value = self.win32_api_handler.memoryRead(addr)
            if pre_value == 36 and value == 244 :
                self.player2_x_address = addr
                self.player2_y_address = self.player2_x_address - 0x4
                self.flag_address = self.player2_y_address - 0xC0
                self.player1_score_address = self.flag_address - 0x8
                self.player2_score_address = self.player1_score_address - 0x4
                break

        # find ball's (x,y)
        addr = self.player1_x_address
        value = self.win32_api_handler.memoryRead(addr)
        while addr < max_addr :
            addr += 0x4
            pre_value = value
            value = self.win32_api_handler.memoryRead(addr)
            if pre_value == 56 and value == 0 :
                self.ball_y_address = addr - 0x4
                self.ball_x_address = addr
                break

    def reset(self) :
        c.release()
        self.player1_score = 0
        self.player2_score = 0

    def release_key(self) :
        c.release()

    def state(self) :
        # get players' and ball's coordinate
        s = []
        s.append(self.win32_api_handler.memoryRead(self.player2_y_address))
        s.append(self.win32_api_handler.memoryRead(self.player2_x_address))
        s.append(self.win32_api_handler.memoryRead(self.player1_y_address))
        s[-1] += self.win32_api_handler.memoryRead(self.player1_y_address+1) << 8
        s.append(self.win32_api_handler.memoryRead(self.player1_x_address))
        s.append(self.win32_api_handler.memoryRead(self.ball_y_address))
        s[-1] += self.win32_api_handler.memoryRead(self.ball_y_address+1) << 8
        s.append(self.win32_api_handler.memoryRead(self.ball_x_address))

        time.sleep(0.1) # wait for a while

        # get players' and ball's vector
        s.append(self.win32_api_handler.memoryRead(self.player2_y_address) - s[0])
        s.append(self.win32_api_handler.memoryRead(self.player2_x_address) - s[1])
        s.append(self.win32_api_handler.memoryRead(self.player1_y_address))
        s[-1] += (self.win32_api_handler.memoryRead(self.player1_y_address+1) << 8) - s[2]
        s.append(self.win32_api_handler.memoryRead(self.player1_x_address) - s[3])
        s.append(self.win32_api_handler.memoryRead(self.ball_y_address))
        s[-1] += (self.win32_api_handler.memoryRead(self.ball_y_address+1) << 8) - s[4]
        s.append(self.win32_api_handler.memoryRead(self.ball_x_address) - s[5])

        # get current action 
        return np.concatenate((np.array(s), c.get_key_map()), axis=None)

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

    def action(self, action) :
        print(self.action_space[action])
        self.action_space[action]()

    def step(self, action) :
        self.action(action)
        return self.state(), self.reward()

