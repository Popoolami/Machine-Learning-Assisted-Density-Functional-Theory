import numpy as np
import subprocess
from itertools import islice
import os
import re
from os.path import exists

home = '/work/p/popoola/ANN/Si-Onetimerun'
#read input data from QE file - second implementation
with open(home+"/QE-input", "r") as f:
    for line in f:
        #read number of atoms here
        if re.search("nat", line):
            word1 = line
            for word in word1.split():
                if word.isdigit():
                    nAtom = int(word)

if exists(home+"/CFG-all"):
    os.remove(home+"/CFG-all")

with open(home+"/CFG-all", 'w') as f1:
    j = 1
    for filename in os.listdir(home+"/CFGS/"):
        f = open(home+"/CFGS/CFG."+str(j), 'r')
        f1.write("".join(islice(f,nAtom+1)))#islice(iterable, stop)
        f.close()
        j+=1
