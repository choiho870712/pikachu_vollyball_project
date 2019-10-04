from pynput.keyboard import Key, Controller
import time
import numpy as np

keyboard = Controller()
is_pressed = [0, 0, 0, 0] # left, right, up, down

def release():
    if is_pressed[0] != 0 :
        keyboard.release(Key.left)
        is_pressed[0] = 0

    if is_pressed[1] != 0 :
        keyboard.release(Key.right)
        is_pressed[1] = 0

    if is_pressed[2] != 0 :
        keyboard.release(Key.up)
        is_pressed[2] = 0

    if is_pressed[3] != 0 :
        keyboard.release(Key.down)
        is_pressed[3] = 0

    keyboard.release(Key.enter)

def left():
    if is_pressed[1] != 0 :
        keyboard.release(Key.right)
        is_pressed[1] = 0

    if is_pressed[0] != 1 :
        keyboard.press(Key.left)
        is_pressed[0] = 1

def right():
    if is_pressed[0] != 0 :
        keyboard.release(Key.left)
        is_pressed[0] = 0

    if is_pressed[1] != 1 :
        keyboard.press(Key.right)
        is_pressed[1] = 1

def up():
    if is_pressed[3] != 0 :
        keyboard.release(Key.down)
        is_pressed[3] = 0

    if is_pressed[2] != 1 :
        keyboard.press(Key.up)
        is_pressed[2] = 1

def down():
    if is_pressed[2] != 0 :
        keyboard.release(Key.up)
        is_pressed[2] = 0

    if is_pressed[3] != 1 :
        keyboard.press(Key.down)
        is_pressed[3] = 1

def p():
    keyboard.press(Key.enter)

def get_key_map() :
    keyboard.release(Key.enter)
    return np.array(is_pressed)