from pynput.keyboard import Key, Controller
import numpy as np

keyboard = Controller()
key_map = [0, 0, 0, 0] # left, right, up, down

def release():
    if key_map[0] != 0 :
        keyboard.release(Key.left)
        key_map[0] = 0

    if key_map[1] != 0 :
        keyboard.release(Key.right)
        key_map[1] = 0

    if key_map[2] != 0 :
        keyboard.release(Key.up)
        key_map[2] = 0

    if key_map[3] != 0 :
        keyboard.release(Key.down)
        key_map[3] = 0

    keyboard.release(Key.enter)

def left():
    if key_map[1] != 0 :
        keyboard.release(Key.right)
        key_map[1] = 0

    if key_map[0] != 1 :
        keyboard.press(Key.left)
        key_map[0] = 1

def right():
    if key_map[0] != 0 :
        keyboard.release(Key.left)
        key_map[0] = 0

    if key_map[1] != 1 :
        keyboard.press(Key.right)
        key_map[1] = 1

def up():
    if key_map[3] != 0 :
        keyboard.release(Key.down)
        key_map[3] = 0

    if key_map[2] != 1 :
        keyboard.press(Key.up)
        key_map[2] = 1

def down():
    if key_map[2] != 0 :
        keyboard.release(Key.up)
        key_map[2] = 0

    if key_map[3] != 1 :
        keyboard.press(Key.down)
        key_map[3] = 1

def p():
    keyboard.press(Key.enter)

def get_key_map() :
    keyboard.release(Key.enter)
    return np.array(key_map)
