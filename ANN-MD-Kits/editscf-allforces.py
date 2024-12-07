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

if exists(home+"/scf-allforces"):
    os.remove(home+"/scf-allforces")

with open(home+"/scf-allforces", 'w') as f1:
    j = 1
    for filename in os.listdir(home+"/scf-CFGforces/"):
        f = open(home+"/scf-CFGforces/scfForce."+str(j), 'r')
        f1.write("Forces acting on atoms (cartesian axes, Ry/au): \n"+"".join(islice(f,nAtom)))#islice(iterable, stop)
        f.close()
        j+=1

f3 = open(home+"/CFG-allforces", 'a')
with open(home+"/scf-allforces", 'r') as f2:
    for line in f2:
        f3.write(line)
f3.close()
    
