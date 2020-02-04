import os
import math
import numpy as np
import pandas as pd

df = pd.read_excel (r'/Users/ed/Documents/Python/PCA.xlsx')


XXT = df.dot(df.T)
print(XXT)

cov_mat2 = XXT

eig_vals, eig_vecs = np.linalg.eig(XXT)
print(eig_vals, eig_vecs)

opti_xT = eig_vecs[:,1]

opti_x = (np.matrix(opti_xT)).T
c = (df.T).dot(opti_x)
cT = np.matrix(c).T
df_approx = (opti_x).dot(cT)
print(c)
print(df_approx)

#Computing the error 
dif = df - df_approx
print(dif)
x = np.square(dif)

sumx = x.sum()
sumy = sumx.sum()

error = 0.5*np.sqrt(sumy)
