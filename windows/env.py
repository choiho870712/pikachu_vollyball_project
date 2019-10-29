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

    def read_word(self, address) :
        value = 0
        self.ReadProcessMemory(self.processHandle, address + 0x3, self.buffer, self.bufferSize, byref(self.bytesRead))
        value |= int.from_bytes(self.buffer.value, byteorder='little')
        self.ReadProcessMemory(self.processHandle, address + 0x2, self.buffer, self.bufferSize, byref(self.bytesRead))
        value <<= 8
        value |= int.from_bytes(self.buffer.value, byteorder='little')
        self.ReadProcessMemory(self.processHandle, address + 0x1, self.buffer, self.bufferSize, byref(self.bytesRead))
        value <<= 8
        value |= int.from_bytes(self.buffer.value, byteorder='little')
        self.ReadProcessMemory(self.processHandle, address, self.buffer, self.bufferSize, byref(self.bytesRead))
        value <<= 8
        value |= int.from_bytes(self.buffer.value, byteorder='little')
        return value

    def read_byte(self, address) :
        self.ReadProcessMemory(self.processHandle, address, self.buffer, self.bufferSize, byref(self.bytesRead))
        return int.from_bytes(self.buffer.value, byteorder='little')

    def search_addresses(self, low_address, high_address, target_value1, target_value2) :
        addr = low_address
        value = self.read_word(addr)
        while addr <= high_address :
            addr += 0x4
            pre_value = value
            value = self.read_word(addr)
            if pre_value == target_value1 and value == target_value2 :
                return addr - 0x4

        return 0

    def print_addresses_value(self, low_address, high_address) :
        s = []
        addr = low_address
        while addr <= high_address :
            s.append(self.read_word(addr))
            addr += 0x4

        print(s)

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
        self.cur_state = np.array(self.sub_state())
        self.pre_state = self.cur_state
        self.time_stamp = time.time()
        print("envrionment is ready!")
        print(self.cur_state)
        print("flag = %d"%self.flag())

    def caculate_addresses(self) :
        self.player1_y_address = self.win32_api_handler.search_addresses(0x2000000, 0x5000000, 396, 244)
        self.player1_x_address = self.player1_y_address + 0x4
        self.base = self.player1_y_address - (self.player1_y_address & 0xffff)
        self.player2_y_address = self.win32_api_handler.search_addresses(self.base, self.base + 0xffff, 36, 244)
        self.player2_x_address = self.player2_y_address + 0x4
        self.ball_y_address = self.win32_api_handler.search_addresses(self.base + 0x6000, self.base + 0xffff, 56, 0)
        self.ball_x_address = self.ball_y_address + 0x4
        # self.flag_address = self.win32_api_handler.search_addresses(self.base + 0xe00, self.base + 0xf00, 0, 15) - 0x4
        self.flag_address = self.player2_y_address - 0xc0

    def release_key(self) :
        c.release()

    def sub_state(self) :
        s = []
        s.append(self.win32_api_handler.read_byte(self.player2_y_address))
        s.append(self.win32_api_handler.read_byte(self.player2_x_address))
        s.append(self.win32_api_handler.read_word(self.player1_y_address))
        s.append(self.win32_api_handler.read_byte(self.player1_x_address))
        s.append(self.win32_api_handler.read_word(self.ball_y_address))
        s.append(self.win32_api_handler.read_byte(self.ball_x_address))
        return np.array(s)

    def state(self) :
        if time.time() - self.time_stamp > 0.1 :
            self.pre_state = self.cur_state
            self.cur_state = self.sub_state()
            vector = self.cur_state - self.pre_state
            self.time_stamp = time.time()
            return np.concatenate((self.pre_state, vector, c.get_key_map()), axis=None), True
        else :
            return np.zeros(16), False

    def flag(self) :
        return self.win32_api_handler.read_byte(self.flag_address)

    def step(self, action) :
        self.action_space[action]()

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

        # # chech memory
        # env.win32_api_handler.print_addresses_value(env.base + 0xd00, env.base+0xfff)
