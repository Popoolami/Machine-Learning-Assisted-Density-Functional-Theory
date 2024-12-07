#%%Title
"""
Created on Tue Aug 24 20:11:53 2021
Hybrid ANN-MD code executed in python
@author: Popoola
"""
#%% Block 1 - Import modules
import numpy as np
import pickle 
import pandas as pd
import re
from itertools import islice
from os.path import exists
import os
np.set_printoptions(linewidth=np.inf)
import subprocess
import os.path
import time

#%% read data from this directory 
home = '/work/p/popoola/ANN/Si-Onetimerun'
home_train=home+'/TrainOut/'

#%%Process initialization  
folders = subprocess.Popen(['./getforces_positions'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = folders.communicate()[0]

p = subprocess.Popen('python editCFG_all.py',stdout=subprocess.PIPE,shell=True)
output = p.communicate()[0]

f = subprocess.Popen('python editCFG_allforces.py',stdout=subprocess.PIPE,shell=True)
output = f.communicate()[0]

CFGs = subprocess.Popen(['./createCFGs-for-retraining'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = CFGs.communicate()[0]

Training = subprocess.Popen('python ANN.py',stdout=subprocess.PIPE,shell=True)
output = Training.communicate()[0]
    
#%% set up parameters
# Units conversions taken from QE
ELECTRONMASS_SI  = 9.1093837015E-31    # Kg
AMU_SI           = 1.66053906660E-27
BOHR_RADIUS_SI   = 0.529177210903e-10
BOHR_RADIUS_Angs = BOHR_RADIUS_SI*1.e+10
AMU_AU           = AMU_SI / ELECTRONMASS_SI
AMU_RY           = AMU_AU / 2.0
HARTREE_SI       = 4.3597447222071e-18
RYDBERG_SI       = HARTREE_SI/2.0
K_BOLTZMANN_SI   = 1.380649e-23
ry_to_kelvin = RYDBERG_SI / K_BOLTZMANN_SI

if exists(home+"/structure.csv"):
    os.remove(home+"/structure.csv")
if exists(home+"/masses.csv"):
    os.remove(home+"/masses.csv")

#read input from QE input file
with open(home+"/QE-input", "r") as f:
    for line in f:
        #read number of atoms here
        if re.search("nat", line):
            word1 = line
            for word in word1.split():
                if word.isdigit():
                    nAtom = int(word)
        #read number of elements here
        if re.search("ntyp", line):
            word2 = line
            for word in word2.split():
                if word.isdigit():
                    nElement = int(word)
        #read the timestep here
        if re.search("dt", line):
            word3 = line
            for word in word3.split():
                if word.isdigit():
                    deltaT = float(word)
        if "ATOMIC_POSITIONS" in line:
            outf=open(home+"/structure.csv", "w")
            outf.write("".join(islice(f,nAtom))) #islice(iterable, stop)
            outf.close()
        if "ATOMIC_SPECIES" in line:
            outff=open(home+"/masses.csv", "w")
            outff.write("".join(islice(f,nElement))) #islice(iterable, stop)
            outff.close()

#read the n_samples from the number of files im the current CFGS directory
n_files = 0
for filename in os.listdir(home+"/CFGforces/"):
    n_files += 1

nAtom_per_element = np.genfromtxt(home+'/structure.csv',skip_header=False,dtype='unicode')
element_list = nAtom_per_element[:,0].tolist()
nAtom_per_element = pd.DataFrame(nAtom_per_element)
nAtom_per_element.columns = ['Elements', 'x', 'y', 'z']
nAtom_per_element = nAtom_per_element.pivot_table(columns=['Elements'], aggfunc='size')
nAtom_per_element = nAtom_per_element.tolist()

masses_input = np.genfromtxt(home+'/masses.csv',skip_header=False,dtype='unicode')

if masses_input.ndim == 1:
    mass_per_element = []
    masses_input = masses_input[1]
    masses_input = masses_input.astype('float32')
    mass_per_element.append(masses_input)
    
else:
    masses_input = masses_input[:,1]
    masses_input = masses_input.astype('float32')
    mass_per_element = masses_input.tolist()

mass_list = []
idx=0
for i in mass_per_element:
    mp = nAtom_per_element[idx]
    for j in range(int(mp)):
        mass_list.append(i*AMU_RY)
    idx = idx + 1


n_dgf= 3*nAtom # number of associated degrees of freedom 
n_samples = n_files 
deltaT2=deltaT**2
deltaT_1=1./deltaT

#read MD data
data_input = np.genfromtxt(home+'/data_file.in',skip_header=False,dtype='unicode')
data_input = data_input[:,2].astype("int")
stepLimit = data_input[0]

# select integrators from input file
data_input = np.genfromtxt(home+'/data_file.in',skip_header=False,dtype='unicode')
data_input = data_input[:,2].astype("int")
verlet = data_input[1]; leap_frog = data_input[2]; pred_corr = data_input[3]

if verlet == 1: 
    verlet=True
    leap_frog=False
    pred_corr=False
if leap_frog == 1:
    verlet=False
    leap_frog=True
    pred_corr=False
if pred_corr == 1:
    verlet=False
    leap_frog=False
    pred_corr=True  
    
# coefficients for predictor corrector integrator
cr1=0.12500 
cr2=0.41666666666666 
cr3=-0.0416666666666 
cv1=0.29166666666666 
cv2=0.2500 
cv3=-0.0416666666666 

#scale_temp = (1.4382e8)/(3*nAtom)
scale_temp=2.0*ry_to_kelvin/(3*nAtom)
#%% setup job
r=np.zeros((3,nAtom))
rv=np.zeros((3,nAtom))
ra=np.zeros((3,nAtom))

# for storage and comparison
F_MD = np.zeros((stepLimit, n_dgf))          #MD forces
r_MD = np.zeros((stepLimit, n_dgf))          #MD positions
rv_MD = np.zeros((stepLimit, n_dgf))          #MD velocities
T_MD = np.zeros((stepLimit))             #MD temperature
r_MD_scaled = np.zeros((stepLimit, n_dgf))   #scaled MD positions
                 
#%%Block 4 - Read DFT positions and forces
# reading DFT positions in cartesian coordinates
r_DFT = np.zeros((0,n_dgf)) # 3 because x,y,z
def build_cfg(r_DFT,file_in,n_samples):
    for isample in range(1,n_samples+1):
        r_temp = np.genfromtxt(file_in %isample,skip_header=True,dtype='unicode') #skip footer to remove -- below atomic positions
        r_temp = np.transpose(r_temp[:,1:]) # order x1 x2 array suitable for MD
        r_temp = r_temp.astype('float64') #Assigns float
        r_row_cart = r_temp #np.matmul(A, r_temp) #Fract to cart coord conversion
        r_row_cart = r_row_cart.reshape((1,n_dgf),order='F')  # order x1 y1 z1 x2 y2 z2
        r_DFT = np.vstack([r_DFT,r_row_cart])
    return r_DFT
r_DFT = build_cfg(r_DFT, home+'/CFGS/CFG.%d', n_samples)
r_DFT = r_DFT/BOHR_RADIUS_Angs # conversion from Angstrom to Bohr

# reading DFT forces
F_dft = np.zeros((0,n_dgf)) # 3 because x,y,z and Forces directly scaled to acceleration without mass
def build_cfg_for(F_dft,file_in,n_samples):
    for isample in range(1,n_samples+1):
        F_temp = np.genfromtxt(file_in %isample,skip_header=True,dtype='unicode') #skip footer to remove -- below atomic positions
        F_temp = np.transpose(F_temp) 
        F_temp=F_temp.astype('float64')
        F_temp = F_temp.reshape((1,n_dgf),order='F')  # order fx1 fy1 fz1 fx2 fy2 fz2                                        
        F_dft = np.vstack([F_dft,F_temp])
    return F_dft     
F_dft = build_cfg_for(F_dft, home+'/CFGforces/CFGForce.%d', n_samples)

#%% Calculate DFT velocities - use backward difference to get the last term
rv_DFT = np.zeros((n_samples, n_dgf))
for i in range(0, n_samples-1):
    if i <= n_samples-2:
        rv_DFT[i, :] = (r_DFT[i+1,:] - r_DFT[i,:])/(deltaT)    # forward difference to obtain all terms except the last one
    if i == n_samples-1:
        rv_DFT[i, :] = (r_DFT[i,:] - r_DFT[i-1,:])/(deltaT)    # backward difference to obtain the last term 

#%%Temperature from DFT data
T_dft = np.genfromtxt(home+'/temperature.dat', skip_header = False, dtype='unicode')
T_dft = T_dft.astype('float64')

#%% Block 6 - Read saved scalers and regressors for Forces
# read saved input scalar, output scaler and regression from a previously trained atomic positions
file_input_scaler=home_train+'/scaler_input_100x0_P.sav' # name of input scaler file
file_output_scaler=home_train+'/scaler_output_100x0_P.sav' # name of output scaler file
file_regr=home_train+'/regression_P_100x0.sav' # regression file
scaler_posit=pickle.load(open(file_input_scaler, 'rb')) # read scaler for positions
scaler_F=pickle.load(open(file_output_scaler, 'rb')) # scaler for P
regr_F=pickle.load(open(file_regr, 'rb')) # regression file_regr

#%%  Initialize  from DFT, Step=0
r = r_DFT[0,:].reshape((3,nAtom), order='F') # initialize positions
rv=np.zeros((3,nAtom)) #rv_DFT[0,:].reshape((3,nAtom), order='F') # initialize velocities
F_init = F_dft[0, :].reshape((3, nAtom), order='F')
#Initialize from DFT last step
#r = r_DFT[n_samples-1,:].reshape((3,nAtom), order='F') # initialize position from last md step
#rv = rv_DFT[n_samples-1,:].reshape((3,nAtom), order='F') # initialize velocity from last md step
#F_init = F_dft[n_samples-1,:].reshape((3,nAtom), order='F') # initialize positions
for (mass, k) in zip(mass_list, range(nAtom)):
    ra[:,k] = F_init[:, k]/mass
    
if (verlet): 
    r_old=r-rv*deltaT+0.5*ra*deltaT2
    r_new=r+rv*deltaT+0.5*ra*deltaT2
   
if (leap_frog):
    rv=rv-ra*deltaT/2.0 # factor 1/2 because for leap frog veloc and accel computed at diff time steps.

if (pred_corr): 
    rv0=np.zeros((3,nAtom))
    ra=np.zeros((3,nAtom))
    ra1=np.zeros((3,nAtom))
    ra2=np.zeros((3,nAtom))
 
iInit=0 # first step
#%%Open files to save results
if exists(home+"/r_MD"):
    os.remove(home+"/r_MD")
if exists(home+"/rv_MD"):
    os.remove(home+"/rv_MD")
if exists(home+"/F_MD"):
    os.remove(home+"/F_MD")
if exists(home+"/T_MD"):
    os.remove(home+"/T_MD")
if exists(home+"/r_MD_save"):
    os.remove(home+"/r_MD_save")
if exists(home+"/F_MD_save"):
    os.remove(home+"/F_MD_save")

#%% Dynamics
step_size = 100
batch_size = 1000 
for m in range(iInit,stepLimit):  
    
    ###Continue Classical MD    
    if (pred_corr):
        #predictor step
        r0=r; rv0=rv
        r=r+deltaT*rv+deltaT2*(cr1*ra+cr2*ra1+cr3*ra2)
        rv=deltaT_1*(r-r0)+deltaT*(cv1*ra+cv2*ra1+cv3*ra2)
        ra2=ra1; ra1=ra

    r_row=r.reshape((1,n_dgf), order='F')
    r_row_sc = scaler_posit.transform(r_row)        # scaled coordinates
    r_MD_scaled[m,:] = r_row_sc                     # Just to save scaled positions
    F_row_sc = regr_F.predict(r_row_sc)             # forces prediction
    F_MD_usc = scaler_F.inverse_transform(F_row_sc)
    F_MD[m, :] = F_MD_usc
    with open(home+"/F_MD", "a") as file3:
        file3.write(str(F_MD[m, :]).replace('[','').replace(']','')+"\n")
    if m>=n_samples and m%step_size == 0:
        F_save = F_MD[m,:].reshape((nAtom, 3))
        with open(home+"/F_MD_save", 'a') as file6:
            file6.write("Forces acting on atoms (cartesian axes, Ry/au): \n" + str(F_save).replace('[','').replace(']','')+"\n")
    F_loop = F_MD_usc.reshape((3,nAtom),order='F')
    
    # For different masses of different elements
    for (mass, k) in zip(mass_list, range(nAtom)):
        ra[:,k] = F_loop[:, k]/mass

    # this will need to be replaced by outputting into a file    
    r_MD[m, :] = r.reshape((1,n_dgf),order='F')# store positions
    with open(home+"/r_MD", "a") as file1:
        file1.write(str(r_MD[m, :]).replace('[','').replace(']','')+"\n")

    if m>=n_samples and m%step_size == 0:
        r_save = r_MD[m,:].reshape((nAtom, 3))*BOHR_RADIUS_Angs
        with open(home+"/r_MD_save", 'a') as file5, open(home+'/CFG-all', 'a') as saveCFG:
            file5.write('ATOMIC_POSITIONS (angstrom) \n')
            saveCFG.write('ATOMIC_POSITIONS (angstrom) \n')
            for (element,k) in  zip(element_list,range(nAtom)):
                file5.write(str(element)+'\t'+str(r_save[k,:]).replace('[','').replace(']','')+"\n")
                saveCFG.write(str(element)+'\t'+str(r_save[k,:]).replace('[','').replace(']','')+"\n")

    rv_MD[m, :] = rv.reshape((1,n_dgf),order='F')# store velocity
    with open(home+"/rv_MD", "a") as file2:
        file2.write(str(rv_MD[m, :]).replace('[','').replace(']','')+"\n")

    # Temperature for different masses
    T = 0.0
    rv_loop = rv_MD[m, :].reshape((3,nAtom),order='F')
    for (mass, k) in zip(mass_list, range(nAtom)):
        T = T + scale_temp*mass*np.sum(rv_loop[:, k]**2) 
    T_MD[m] = T
    with open(home+"/T_MD", "a") as file4:
        file4.write(str(T_MD[m]).replace('[','').replace(']','')+"\n")

    if (pred_corr):
    # corrector step
        r=r0+deltaT*rv0+deltaT2*(cr1*ra+cr2*ra1+cr3*ra2)
        rv=deltaT_1*(r-r0)+deltaT*(cv1*ra+cv2*ra1+cv3*ra2)
        ra2=ra1; ra1=ra
    if (leap_frog):
        rv = rv + deltaT * ra    #velocity
        # subtract total momentum
        moment=np.sum(rv,axis=1)/nAtom
        for i in range(0,3):
            rv[i,:]=rv[i,:]-moment[i]
        r = r + deltaT * rv      #position
    if (verlet):    # default verlet
        r_new = 2.0e0*r - r_old + deltaT2*ra
        rv=0.5*(r_new-r_old)/deltaT
        r_old=r; r=r_new
    #  subtract displacement of center of mass, following QE        
        delta=np.sum((r_new-r),axis=1)/nAtom
        for i in range(0,3):
            r_new[i,:]=r_new[i,:]-delta[i]

    #ANN retraining
    if m>n_samples+10 and m%batch_size == 0:
        
        #Create structures for scf calculations
        with open(home+"/r_MD_save", "r") as f:
            i=1
            for line in f:
                outf=open(home+"/scf."+str(i), "w")
                with open(home+"/pwscf", "r") as s:
                    for line in s:
                       outf.write(line) 
                outf.write("ATOMIC_POSITIONS (angstrom) \n"+"".join(islice(f,nAtom))) #islice(iterable, stop)
                outf.close()
                i+=1
        
        #submit job
        scf = subprocess.call(['sbatch batch'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while not os.path.exists(home+'/forces'):
            time.sleep(10)
        time.sleep(300)
    
        A = subprocess.Popen(['./getSCFforces'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = A.communicate()[0]

        B = subprocess.Popen('python editscf-allforces.py',stdout=subprocess.PIPE,shell=True )
        output = B.communicate()[0]

        C = subprocess.Popen(['./createCFGs-for-retraining'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = C.communicate()[0]

        D = subprocess.Popen('python ANN.py',stdout=subprocess.PIPE,shell=True)
        output = D.communicate()[0]
    
        #Reload the model after retraining
        file_input_scaler = home_train+'/scaler_input_100x0_P.sav' # name of input scaler file
        file_output_scaler = home_train+'/scaler_output_100x0_P.sav' # name of output scaler file
        file_regr = home_train+'/regression_P_100x0.sav' # regression file
        scaler_posit = pickle.load(open(file_input_scaler, 'rb')) # read scaler for positions
        scaler_F = pickle.load(open(file_output_scaler, 'rb')) # scaler for P
        regr_F = pickle.load(open(file_regr, 'rb')) # regression file_regr


        os.remove(home+"/r_MD_save")
        os.remove(home+"/scf-allforces")
###The END
