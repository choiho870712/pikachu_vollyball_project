from pynput.keyboard import Key, Controller
import time
import numpy as np

keyboard = Controller()
is_pressed = [0, 0, 0, 0] # left, right, up, down

def release():
    keyboard.release(Key.left)
    keyboard.release(Key.right)
    keyboard.release(Key.up)
    keyboard.release(Key.down)
    keyboard.release(Key.enter)
    is_pressed[0] = 0
    is_pressed[1] = 0
    is_pressed[2] = 0
    is_pressed[3] = 0

def left():
    keyboard.release(Key.right)
    keyboard.press(Key.left)
    is_pressed[0] = 1
    is_pressed[1] = 0

def right():
    keyboard.release(Key.left)
    keyboard.press(Key.right)
    is_pressed[0] = 0
    is_pressed[1] = 1

def up():
    keyboard.release(Key.down)
    keyboard.press(Key.up)
    is_pressed[2] = 1
    is_pressed[3] = 0

def down():
    keyboard.release(Key.up)
    keyboard.press(Key.down)
    is_pressed[2] = 0
    is_pressed[3] = 1

def p():
    keyboard.press(Key.enter)
    time.sleep(0.1)
    keyboard.release(Key.enter)

def get_key_map() :
    return np.array(is_pressed)