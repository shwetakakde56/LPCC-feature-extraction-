from scipy.io import wavfile
import os
import numpy as np
import warnings
warnings.simplefilter('ignore')
import scipy
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
import pandas as pd

extensions = ('.wav')
X = []
Y = []
featu = []
num_ceps = 13
lifter = 0
normalize = True


for subdir, dirs, files in os.walk(r'Dataset\\train\\'):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:

            audio = os.path.join(subdir, file)
            (rate,sig) = wavfile.read(audio)
            # mfcc_feat = mfcc(sig,rate,nfft=1024)
            # mfc = np.mean(mfcc_feat,axis=0)
            
            # compute lpcs
            lpccs = lpc(sig=sig, fs=rate, num_ceps=num_ceps)
            lpccs = np.mean(lpccs,axis=0)
            
#            mfc = mfc.reshape(1,-1)
            X.append(lpccs)
            folder = subdir[subdir.rfind('\\') + 1:]

            if folder=="H_F_N_i":
                Y.append(1)
            if folder=="H_M_N_i":
                Y.append(2)
            if folder=="P_F_N_i":
                Y.append(3)
            if folder=="P_M_H_i":
                Y.append(4)
            if folder=="P_M_L_i":
                Y.append(5)   
            if folder=="P_M_lhl_i":
                Y.append(6)  
            if folder=="P_M_N_a":
                Y.append(7)
            if folder=="P_M_N_i":
                Y.append(8)  

X = np.array(X)
Y = np.array(Y)

df = pd.DataFrame(X)
filepath ="lpccfeat.xlsx"
df.to_excel(filepath)

df1 = pd.DataFrame(Y)
filepath ="lpcclabel.xlsx"
df1.to_excel(filepath)
print('done') 