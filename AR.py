# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:31:38 2018

@author: 15065
"""

###This is a program for the simulation of ARX model with real-word data
import csv_read
import numpy as np
import matplotlib.pyplot as plt
import os
os.system('cls')
    
"""
LS updating rule

"""
def LS(x_new,P,d,theta,phi,error):
    error_temp = (np.dot(phi,theta)-x_new)**2
    error.append(error_temp[0,0])
    a = 1.0/(1 + np.dot(np.dot(phi,P),phi.transpose()))   
    theta = theta + a*np.dot(P,phi.transpose())*(x_new-np.dot(phi,theta))
    P = P - a*np.dot(np.dot(np.dot(P,phi.transpose()),phi),P)
    return P,theta,error

"""
Analytic LS solution

"""

def Analytic_LS(x_new,phi,R,Q):
    R = R + np.dot(phi.transpose(),phi)
    Q = Q + x_new*phi.transpose()
    theta = np.dot(np.mat(R).I,Q)
    
    return R,Q,theta
       
"""
main function with Analytic solution

"""
def main_functionA(d,x,step_0,step,theta_0,R0,Q0):
    n1 = len(theta_0); n2 = np.shape(R0)[0]; n3 = np.shape(Q0)[0]
    error = []
    prediction = []
    #P = 0.2*np.eye(d)
    R = 0.2*np.eye(d)
    Q = np.zeros((d,1))
    R[0:n2,0:n2] = R0
    Q[0:n3,:] = Q0
    theta = np.zeros((d,1))
    if n1 > 0:
        for i in range(n1):
            theta[i,0] = theta_0[i,0]
    for i in range(step_0,step):
        phi = np.zeros((1,d))
        for j in range(d):
            phi[0,j] = x[i-1-j]
        error_temp = (np.dot(phi,theta)-x[i])**2
        error.append(error_temp[0,0])
        R,Q,theta = Analytic_LS(x[i],phi,R,Q)
        predict = np.dot(phi[0,:-1],theta[1:,0])[0,0] + x[i]*theta[0,0]
        prediction.append(predict)
    return error,theta,prediction,R,Q


"""
main function with Recursive solution

"""
def main_functionR(d,x,step_0,step,theta_0,P0):
    n1 = len(theta_0); n2 = np.shape(P0)[0]; 
    error = []
    prediction = []
    P = 0.2*np.eye(d)
    P[0:n2,0:n2] = P0
    theta = np.zeros((d,1))
    if n1 > 0:
        for i in range(n1):
            theta[i,0] = theta_0[i,0]
    for i in range(step_0,step):
        phi = np.zeros((1,d))
        for j in range(d):
            phi[0,j] = x[i-1-j]
        P,theta,error = LS(x[i],P,d,theta,phi,error)
        predict = np.dot(phi[0,:-1],theta[1:,0]) + x[i]*theta[0,0]
        prediction.append(predict)
    return error,theta,prediction,P
"""
average cumulative error

"""
def average_error(error):
    error_sum = [0]; error_average = []
    for i in range(20,len(error)):
        temp = error_sum[-1]+error[i]
        error_sum.append(temp)
        error_average.append(error_sum[-1]/(i+1))
    return error_average

##load original time series data
x_data_o = csv_read.results
#x_data = np.log(x_data_o)
x_data = x_data_o

x_l = len(x_data)


##do differencing to the original data
#x_data = [(x_data_o[1]+x_data_o[0])/2]
#for i in range(1,len(x_data_o)-1):
#    x_data.append((x_data_o[i+1]+x_data_o[i]+x_data_o[i-1])/3)
#
#x_data.append((x_data_o[-1]+x_data_o[-2])/2)

##Solution with Analytic solution


#num = 5
#theta0 = []
#R = 0.2*np.eye(4)
#Q = np.zeros((4,1))    
#error_b = locals()
#theta = locals()
#prediction = locals()
#batch = len(x_data)//num  ##batch size
#error_b['error_1'],theta['theta_1'],prediction['prediction_1'],R,Q = main_functionA(4,x_data,4,batch,theta0,R,Q)
#for i in range(2,num+1):
#    error_b['error_%d'%i],theta['theta_%d'%i],prediction['prediction_%d'%i],R,Q = main_functionA(4*i,x_data,batch*(i-1),batch*i,theta['theta_%d'%(i-1)],R,Q)

###Solution with Recursive solution
num = 4   ##changing times
error_b = locals()
theta = locals()
theta0 = []
prediction = locals()
batch = len(x_data)//num  ##batch size
P = 0.2*np.eye(2)
error_b['error_1'],theta['theta_1'],prediction['prediction_1'],P = main_functionR(2,x_data,2,batch,theta0,P)
for i in range(2,num+1):
    error_b['error_%d'%i],theta['theta_%d'%i],prediction['prediction_%d'%i],P = main_functionR(2*i,x_data,batch*(i-1),batch*i,theta['theta_%d'%(i-1)],P)


##收集MSE
error = []; pr_value = []
for i in range(1,num+1):
    error.extend(error_b['error_%d'%i])
for i in range(1,num+1):
    pr_value.extend(prediction['prediction_%d'%i])


error_average = average_error(error)

"""
ARX(p) model prediction error

"""
d = 12
theta_arx = np.zeros((d,1))
R = 0.2*np.eye(d)
Q = np.zeros((d,1))
error_arx = []
for i in range(d,x_l):
    phi = np.zeros((1,d))
    for j in range(d):
        phi[0,j] = x_data[i-1-j]
    error_temp = (np.dot(phi,theta_arx)-x_data[i])**2
    R,Q,theta_arx = Analytic_LS(x_data[i],phi,R,Q)
    error_arx.append(error_temp[0,0])

#d = 4
#error_arx,theta_arx = main_function(d,x,u,d,10000,theta_0)

error_arx_average = average_error(error_arx)

l = len(error_average)
t = list(range(l))
plt.plot(t,error_average,'-',label='ARX')
s = list(range(len(error_arx_average)))
plt.plot(s,error_arx_average,'-.',label='ARX(p)')
plt.legend()
#plt.yscale('log')
plt.savefig('ARX',format='eps',dpi=1000)
plt.show()


num1 = list(range(len(pr_value)))
num2 = list(range(len(x_data)))
plt.plot(num1,pr_value,'-',label='original data')
plt.plot(num2,x_data,'-',label='prediction data')
plt.legend()
#plt.yscale('log')
plt.savefig('data_x',format='eps',dpi=1000)
plt.show()
#t = list(range(len(error_arx)))
#plt.plot(y,error,'-')
#plt.show()
t = list(range(len(error[10:])))
fig = plt.figure()
ax_1 = fig.add_subplot(211)
ax_1.plot(t,error[10:],'-',label='ARX')
plt.legend(loc='upper right')
#plt.yscale('log')
#plt.savefig('ARX error',format='eps',dpi=1000)
s = list(range(len(error_arx[10:])))
ax_2 = fig.add_subplot(212)
ax_2.plot(s,error_arx[10:],'-',label='ARX(p)')
plt.legend(loc='upper right')
#plt.yscale('log')
plt.savefig('ARX(p) error',format='eps',dpi=1000)



