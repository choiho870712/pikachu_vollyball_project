
# coding: utf-8

# In[ ]:


# cmd: cd  C:\ProgramData\Anaconda3
# "pip install pynput" on cmd
import pynput

from pynput.keyboard import Key, Controller
import time



keyboard = Controller()

# Press and release space
#keyboard.press(Key.space)
#keyboard.release(Key.space)

def Up():
    keyboard.press( Key.up )
    keyboard.release( Key.up )
    
def Down():
    keyboard.press( Key.down )
    keyboard.release( Key.down )
    
def Left():
    keyboard.press( Key.left )
    keyboard.release( Key.left )

def Right():
    keyboard.press( Key.right )
    keyboard.release( Key.right )

def Enter():
    keyboard.press( 'e' )
    keyboard.release( 'e' )
    keyboard.press( Key.enter )
    keyboard.release( Key.enter )
    
def Controller( num ):
    if num == 0:
        Up()
    
    if num == 1:
        Down()

    if num == 2:
        Left()
    
    if num == 3:
        Right()
    
    if num == 4:
        Enter()
        

