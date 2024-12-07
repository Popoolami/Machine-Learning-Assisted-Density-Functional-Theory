import numpy as np
import subprocess
from itertools import islice
import re
import os

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

#get all the lines for ATOMIC_POSITIONS and output them into a single file
child = subprocess.Popen('grep -i "ATOMIC_POSITIONS (angstrom)" -A'+str(nAtom)+' QE-output|grep -v -- -- > CFG-all',stdout=subprocess.PIPE,shell=True)
output = child.communicate()[0]

#Delete the last structure
with open("CFG-all", "r") as f1:
    lines = f1.readlines()
os.remove(home+"/CFG-all")
with open("CFG-all", "w") as f2:
    f2.writelines(lines[:-(nAtom+1)])
  
#read the just created file and print each config. to a different file
with open("CFG-all", "r") as f:
    i=1
    for line in f:
        outf=open("CFG."+str(i), "w")
        outf.write("ATOMIC_POSITIONS (angstrom) \n"+"".join(islice(f,nAtom))) #islice(iterable, stop)
        outf.close()
        i+=1
