# simple-linear-regression-by-mathemaics

#@debasish saha

#model 1

import  matplotlib.pyplot as plt

import numpy as np

x=np.array([95,85,80,70,60]) #independent

y=np.array([85,95,70,65,70]) #dependent

n=np.size(x)

m_x,m_y=np.mean(x),np.mean(y)

ss_xy=np.sum(y*x)-n* m_x *m_y

ss_xx=np.sum(x*x)-n* m_x *m_x

b0_1=ss_xy/ss_xx #slope

b0_0=m_y-b0_1*m_x #intercept

def predict(x):

    y_pred=b0_0+b0_1*x
    
    return y_pred

print('Intercept: ',b0_0)

print('Slope: ',b0_1)

#By mathematics

import math 

sigma_x=math.sqrt(n*np.sum(x*x)-np.sum(x)*np.sum(x))

sigma_y=math.sqrt(n*np.sum(y*y)-np.sum(y)*np.sum(y))

print(sigma_x)

print(sigma_y)

cov_xy=(n*np.sum(x*y))-np.sum(x)*np.sum(y)

print('covariance: ',cov)

cor_coeff=cov_xy/(sigma_x*sigma_y)

print('Correlation coefficient: ',cor_coeff)

sse=sum((y-y_pred)*(y-y_pred))

sst=sum((y-m_y)*(y-m_y))

r2_cal=1-(sse/sst)

print(r2_cal)

print(y_pred)

plt.scatter(x,y)

plt.plot(x,y_pred,color='k',marker='o')

#also correlation coefficient and r2 value by inbuild method

from sklearn.metrics import r2_score

print(r2_score(y,y_pred))

r2=r2_score(y,y_pred)

r=r2**0.5

print(r)

#call the function for predict

predict(84)


