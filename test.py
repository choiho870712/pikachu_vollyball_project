import cv2
from mss import mss
import numpy as np
import time

class env() :
    def __init__(self) :
        self.bbox = {'top': 51, 'left': 0, 'width': 520, 'height': 275}
        self.sct = mss()
        self.lower_red = np.array([0, 200, 120])
        self.upper_red = np.array([10, 255, 150])
        self.lower_yellow = np.array([20, 120, 100])
        self.upper_yellow = np.array([30, 255, 255])
        self.gaming_status = "gameset"
        self.time_stamp = 0

    def get_screen(self) :
        def preprocessing(img) :
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_red = cv2.inRange(img_hsv, self.lower_red, self.upper_red)
            img_yellow = cv2.inRange(img_hsv, self.lower_yellow, self.upper_yellow)
            return img_red + img_yellow

        def get_reward(img) :
            mask_boom = img[-1:, :]
            reward = 0
            if self.gaming_status == "gaming" :
                if np.sum(mask_boom[ :, 225: ]) > np.sum(mask_boom[ :, :225 ]) :
                    reward = -1
                elif np.sum(mask_boom[ :, 225: ]) < np.sum(mask_boom[ :, :225 ]) :
                    reward = 1

            return reward

        def set_status(img) :
            if self.gaming_status == "gaming" :
                if np.sum(img[ 20:120 , : ]) > 3200000 : # restart( find gameset flag )
                    self.gaming_status = "gameset"
                else :
                    self.gaming_status = "next_stage_delay"
                    self.time_stamp = time.time()
            elif self.gaming_status == "gameset" and np.sum(img) > 9000000 : # wait for start( find start page )
                self.gaming_status = "waiting_for_start"
            elif self.gaming_status == "waiting_for_start" and np.sum(img) == 0 : # black screen
                self.gaming_status = "start_delay"
                self.time_stamp = time.time()
            elif self.gaming_status == "start_delay" and time.time() - self.time_stamp > 4 : # gaming delay end
                self.gaming_status = "gaming"
            elif self.gaming_status == "next_stage_delay" and time.time() - self.time_stamp > 2 : # next delay end
                self.gaming_status = "gaming"

        img = cv2.resize(np.array(self.sct.grab(self.bbox)), (450, 350))
        mask = preprocessing(img)
        reward = get_reward(mask)
        set_status(mask)
            
        return mask, reward, self.gaming_status
    
env = env()

while "Screen capturing":
    img, reward, gaming_status = env.get_screen()

    cv2.imshow('screenShot', img)
    print(reward, gaming_status)

    # Press "q" to quit
    if cv2.waitKey(10) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break