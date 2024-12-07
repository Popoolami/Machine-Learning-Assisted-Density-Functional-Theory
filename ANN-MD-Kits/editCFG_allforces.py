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

if exists(home+"/CFG-allforces"):
    os.remove(home+"/CFG-allforces")

with open(home+"/CFG-allforces", 'w') as f1:
    j = 1
    for filename in os.listdir(home+"/CFGforces/"):
        f = open(home+"/CFGforces/CFGForce."+str(j), 'r')
        f1.write("Forces acting on atoms (cartesian axes, Ry/au): \n"+"".join(islice(f,nAtom)))#islice(iterable, stop)
        f.close()
        j+=1
