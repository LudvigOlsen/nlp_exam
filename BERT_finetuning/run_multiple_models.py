#!/usr/bin/python
from subprocess import call

for fold in reversed(range(8)):
    for rep in range(3):

        print("doing: fold {}, rep {}".format(fold+1, rep))        
        call(["python", "train.py", "--fold", str(fold+1)])