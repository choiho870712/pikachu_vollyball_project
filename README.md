# ptrace_scope setting

```
sudo bash
echo 0 > /proc/sys/kernel/yama/ptrace_scope
```

# run learner
assume there are 2 actors
```
xhost +
python learner --actor-num 2
Learner: Model saved in  log/191101175440/model.pt
```

# get log file permission
```
sudo chattr log/
sudo lsattr log/
sudo chmod -R 777 log/
```

# run actor
copy learner's model id and run 1st actor
```
Xvfb :1 &
export DISPLAY=:1
DISPLAY=:1 python actor.py --simnum 0 --load-model 191101175440
```
copy learner's model id and run 2nd actor
```
Xvfb :2 &
export DISPLAY=:2
DISPLAY=:2 python actor.py --simnum 1 --load-model 191101175440
```

# test
```
python actor.py --load-model 191101175440
```
