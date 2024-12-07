import numpy as np
import subprocess
from itertools import islice
import re


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

#read the just created file and print each config. to a different file

with open(home+"/CFG-all", "r") as f1:
    i=1
    for line in f1:
        outf1=open(home+"/CFG."+str(i), "w")
        outf1.write("ATOMIC_POSITIONS (angstrom) \n"+"".join(islice(f1,nAtom))) #islice(iterable, stop)
        outf1.close()
        i+=1

#read the just created file and print each config. to a different file

with open(home+"/CFG-allforces", "r") as f2:
    i=1
    for line in f2:
        outf2=open(home+"/CFGForce."+str(i), "w")
        outf2.write("Forces acting on atoms (cartesian axes, Ry/au): \n"+"".join(islice(f2,nAtom))) #islice(iterable, stop)
        outf2.close()
        i+=1

