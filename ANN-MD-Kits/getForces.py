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

#get all the lines for ATOMIC_POSITIONS and output them into a single file
child = subprocess.Popen('grep -i "Forces acting on atoms (cartesian axes, Ry/au):" -A '+str(nAtom+1)+' QE-output|grep -v -- -- > CFG-allforces',stdout=subprocess.PIPE,shell=True)
output = child.communicate()[0]

nAtom_list = []
for i in range(nAtom+2):
    nAtom_list.append(i)

#delete the data for the first force on atoms
lines = []
with open(home+"/CFG-allforces", 'r') as fp:
    lines = fp.readlines()
with open(home+"/CFG-allforces", 'w') as fp:
    for number, line in enumerate(lines):
        if number not in nAtom_list:  #put indices of lines to be deleted in this list
            fp.write(line)

#read the just created file and print each config. to a different file
with open(home+"/CFG-allforces", "r") as f:
    i=1
    for line in f:
        outf=open(home+"/CFGForce."+str(i), "w")
        outf.write(home+"Forces acting on atoms (cartesian axes, Ry/au): \n"+"".join(islice(f,nAtom+1))) #islice(iterable, stop)
        outf.close()
        i+=1
