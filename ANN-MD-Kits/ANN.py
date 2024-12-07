#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:55:28 2020
The code takes CFG files  and force vector from a run with n_MD_step and fits it with ANN
08/25/21 IP added batch training
09/29/21 IP forces are read from different files 
         IP the data straucture for training (x1 y1 z1 x2 y2 z2 ...) with rows giving different datasets (MD step)
@author: Inna
"""
#%% Block 1 - Import modules
import numpy as np
import pickle 
import re
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from os.path import exists
import os
#%% Block 2 - read data from directories
home = '/work/p/popoola/ANN/Si-Onetimerun' # indicate path
home_train=home+'/TrainOut/'    #We need a file named TrainOut

#%% Block 3 - read data for positions and forces
BOHR_RADIUS_SI   = 0.529177210903e-10
BOHR_RADIUS_Angs = BOHR_RADIUS_SI*1.e+10

#read input data from QE file - second implementation
with open(home+"/QE-input", "r") as f:
    for line in f:
        #read number of atoms here
        if re.search("nat", line):
            word1 = line
            for word in word1.split():
                if word.isdigit():
                    nAtom = int(word)

#read the n_samples from the number of files im the current CFGS directory
n_files = 0
for filename in os.listdir(home+"/CFGforces/"):
    n_files += 1

n_dgf= 3*nAtom # number of associated degrees of freedom 
n_samples = n_files

# # reading DFT positions in cartesian coordinates
r_row = np.zeros((0,n_dgf)) # 3 because x,y,z
def build_cfg(r_row,file_in,n_samples):
    for isample in range(1,n_samples+1):
        r_temp = np.genfromtxt(file_in %isample,skip_header=True,dtype='unicode') #skip footer to remove -- below atomic positions
        r_temp = np.transpose(r_temp[:,1:]) # order x1 x2 array suitable for MD
        r_temp = r_temp.astype('float64') #Assigns float
        r_row_cart = r_temp.reshape((1,n_dgf),order='F')  # order x1 y1 z1 x2 y2 z2
        r_row = np.vstack([r_row,r_row_cart])
    return r_row
r_row = build_cfg(r_row, home+'/CFGS/CFG.%d', n_samples)
r_row = r_row/BOHR_RADIUS_Angs

F_row=np.zeros((0,n_dgf)) # 3 because x,y,z and Forces directly scaled to acceleration without mass
def build_cfg_for(F_row,file_in,n_samples):
    for isample in range(1,n_samples+1):
        F_temp = np.genfromtxt(file_in %isample,skip_header=True,dtype='unicode') #skip footer to remove -- below atomic positions
        F_temp = np.transpose(F_temp) 
        F_temp = F_temp.reshape((1,n_dgf),order='F')  # order fx1 fy1 fz1 fx2 fy2 fz2                                        
        F_row = np.vstack([F_row,F_temp])
    return(F_row)    
(F_row)=build_cfg_for(F_row, home+'/CFGforces/CFGForce.%d', n_samples)
F_row = F_row.astype('float64')

#%% Block 4 - scale input data  
scaler_input = MinMaxScaler()
scaler_input.fit(r_row)
file=home_train+'scaler_input_100x0_P.sav' # name of  file to store scaler for input variables
pickle.dump(scaler_input,open(file,'wb'))
r_row_train = scaler_input.transform(r_row) # scaled positions

#%% Block 5 - scale output data
scaler_output = MinMaxScaler()
scaler_output.fit(F_row)
file=home_train+'scaler_output_100x0_P.sav' # name of  file to store scaler for output variables
pickle.dump(scaler_output,open(file,'wb'))
F_row_train = scaler_output.transform(F_row)

#%% Block 6 - train
test_size=0.01  #works well for silicon
X_train, X_test, y_train, y_test = train_test_split(r_row_train[:n_samples, :], F_row_train[:n_samples,:],random_state=1,test_size=test_size) #I changed P to F here
n_batch=min(20,int(n_samples*(1-test_size))) #batch size for training
n_iter_no_change=10 # Maximum number of epochs to not meet tol improvement. Only effective when solver=’sgd’ or ‘adam’
max_iter=100000 # maximum iteration for the fit
tol=0.00000001 # tolerance
n_neur=100 # number of neurons
regr_file=home_train+'regression_P_100x0.sav' # file to output regression. This file was also found in TrainOut folder
regr = MLPRegressor(random_state=1, hidden_layer_sizes=(n_neur,),activation='relu',max_iter=max_iter,tol=tol,solver='sgd', batch_size=min(n_batch, n_samples),learning_rate='adaptive', learning_rate_init=0.001) # training 

# partial training in batches
regr.partial_fit(X_train, y_train) # initialize regression
score_prev=regr.score(X_train, y_train) # get score
tol_cur=1.0 # initial tolerance
ite=0 # initialize iteration counter
while tol_cur>tol and ite< max_iter:
    ite=ite+1
    regr.partial_fit(X_train, y_train) # training in batches 
    if (ite % n_iter_no_change ==0): 
        score_cur=regr.score(X_train, y_train)
        tol_cur=score_cur-score_prev
        score_prev=score_cur

with open(home+"/score-sheet", "a") as s:
    s.write('Score on training and testing '+ str(regr.score(X_train, y_train))+'\t' + str(regr.score(X_test, y_test)) +'\n')
    s.write('Converged in '+str(regr.n_iter_) +' iterations')

pickle.dump(regr, open(regr_file, 'wb')) # save regression

#%% Block 8 - Predictions - The forces are being predicted from atomic positions 
# load saved scalers and regressor from home_train directory
file_input_scaler = home_train+'/scaler_input_100x0_P.sav' # name of input scaler file
file_output_scaler = home_train+'/scaler_output_100x0_P.sav' # name of output scaler file
file_regr = home_train+'/regression_P_100x0.sav' # regression file

scaler_posit = pickle.load(open(file_input_scaler, 'rb')) # read scaler for positions
scaler_P = pickle.load(open(file_output_scaler, 'rb')) # scaler for P
regr_P = pickle.load(open(file_regr, 'rb')) # regression file_regr

# %%% Block 9 - Make prediction and save predicted forces and positions
r_row_sc = scaler_input.transform(r_row) # scale atomic position
F_pred_sc = regr.predict(r_row_sc) # PREDICTION using regr from partial fit
F_pred = scaler_output.inverse_transform(F_pred_sc)
if exists(home+"/ANN-forces"):
    os.remove(home+"/ANN-forces")
f1 = open(home+"/ANN-forces", "w")
    
if exists(home+"/DFT-forces"):
    os.remove(home+"/DFT-forces")
f2 = open(home+"/DFT-forces", "w")

for i in range(n_samples):
    F_pred_save = F_pred[i,:]
    f1.write(str(F_pred_save).replace('[','').replace(']','')+"\n")
    F_DFT_save = F_row[i,:]
    f2.write(str(F_DFT_save).replace('[','').replace(']','')+"\n")
# %% Block 10 - Prediction for the new batch of data, currently randomly picked  from the array
# IP: I really do not know what I am doing here
#test_size=0.1
#iStart=2000
#iEnd=iStart+n_batch
#X_train_new, y_train_new = r_row_train[iStart:iEnd, :], F_row_train[iStart:iEnd,:]#new data for training
#X_train=np.vstack((X_train,X_train_new)); y_train=np.vstack((y_train,y_train_new)) # entire training set
#regr.partial_fit(X_train_new, y_train_new) # initialize partial regression
#score_prev=regr.score(X_train_new, y_train_new) # get score
#tol_cur=1.0 # initial tolerance
#ite=0 # initialize iteration counter
#while tol_cur>tol and ite< max_iter:
#    ite=ite+1
#    regr.partial_fit(X_train_new, y_train_new) # training in batches 
#     if (ite % n_iter_no_change ==0): 
#        score_cur=regr.score(X_train_new, y_train_new)
#        tol_cur=score_cur-score_prev
#        score_prev=score_cur

#%%percentage error
sumError = np.zeros((1, n_dgf))
for i in range(n_samples):
    sumError = sumError + (F_pred[i,:] - F_row[i,:])**2
RMSE = np.sqrt(sumError/n_samples)
RMSE = RMSE.reshape(nAtom, 3)
RMSE_per_atom = 0.
for j in range(nAtom):
    for i in range(3):
        RMSE_per_atom += RMSE[j,i]/nAtom
if exists(home+"/RMSE"):
    os.remove(home+"/RMSE")
with open(home+"/RMSE", "a") as ff:
    ff.write('The root mean square error is' + str(RMSE_per_atom))
#%% Close the files
f1.close()
f2.close()
