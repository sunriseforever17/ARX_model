# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:09:38 2018

@author: 15065
"""
###This is a program for the simulation of ARX model with synthetic data
import numpy as np
import matplotlib.pyplot as plt



##仿真人工数据
gamma = 0.8  ##parameter decay rate
###generate the true parameter
alpha = [0.8,-1.0,0.3,-0.4,0.1,0.0,0.05,-0.08,-0.04,0.03,-0.005,0.002,0,0.01,-0.002,0,0.004,-0.004,0,0,0.00005];
beta = [0.5,-0.2,0.1,0.1,0.05,-0.05,0.1,0.003,-0.3,0,0.1,0,0.001,0.002,0,0.0004,-0.0004,0,0,0.00005];
##generate input u
u=[]
for i in range(2000):
    if i%5==0:
        u.append(gamma+0.02*i)
    else:
        u.append(gamma+0.03*i)
for i in range(2000,10000):
    u.append(gamma+0.02*i)
##generate the signal x
x = [0]  
for i in range(1,20):
    x_new = 0
    for j in range(i):
        x_new = x_new + x[i-1-j]*alpha[j]+u[i-1-j]*beta[j]+np.random.uniform(-0.1,0.1)
    x.append(x_new)   
for i in range(20,10000):
    x_new = 0
    for j in range(20):
        x_new = x_new + x[i-j-1]*alpha[j]+u[i-j-1]*beta[j]+np.random.uniform(-0.1,0.1)
    x.append(x_new)
    
"""
LS updating rule

"""
def LS(x_new,P,d,theta,phi,error):
    error_temp = (np.dot(phi,theta)-x_new)**2
    error.append(error_temp[0][0])
    a = 1.0/(1+np.dot(np.dot(phi,P),phi.transpose()))   
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
main function

"""
def main_function(d,x,u,step_0,step,theta_0):
    n = int(len(theta_0)/2)
    error = []
    P = 0.2*np.eye(2*d)
    theta = np.zeros((2*d,1))
    if n>0:
        for i in range(n):
            theta[i] = theta_0[i]
            theta[d+i] = theta_0[n+i]
    for i in range(step_0,step):
        phi = np.zeros((1,2*d))
        for j in range(d):
            phi[0][j] = x[i-1-j]
            phi[0][d+j] = u[i-1-j]
        P,theta,error = LS(x[i],P,d,theta,phi,error)
    return error,theta

"""
average cumulative error

"""
def average_error(error):
    error_sum = [0]; error_average = []
    for i in range(len(error)):
        temp = error_sum[-1]+error[i]
        error_sum.append(temp)
        error_average.append(error_sum[-1]/(i+1))
    return error_average
    
theta_0 = []
error = []  ##prediction error
error_1,theta_1 = main_function(1,x,u,1,10,theta_0) 
error_2,theta_2 = main_function(4,x,u,10,100,theta_1) 
error_3,theta_3 = main_function(9,x,u,100,1000,theta_2) 
error_4,theta_4 = main_function(16,x,u,1000,10000,theta_3)   
error=error_1
error.extend(error_2)
error.extend(error_3)
error.extend(error_4)  

error_average = average_error(error)

"""
ARX(p) model prediction error

"""
d = 4
theta_arx = np.zeros((d*2,1))
R = 0.2*np.ones((d*2,d*2))
Q = np.zeros((d*2,1))
error_arx = []
for i in range(100,10000):
    phi = np.zeros((1,d*2))
    for j in range(d):
        phi[0][j] = x[i-1-j]
        phi[0][d+j] = u[i-1-j]
    R,Q,theta_arx = Analytic_LS(x[i],phi,R,Q)
    error_temp = (np.dot(phi,theta_arx)-x[i])**2
    error_arx.append(error_temp[0,0])

#d = 4
#error_arx,theta_arx = main_function(d,x,u,d,10000,theta_0)
error_arx_average = average_error(error_arx)

###The true parameter
#theta = np.zeros((d*2,1))
#for i in range(4):
#    theta[i,0] = alpha[i]
#    theta[i+d,0] = beta[i]
#
#delta_1 = np.linalg.norm(theta_4-theta)
l = len(error_arx)
t = list(range(l))
plt.plot(t,error_average[0:l],'-',label='ARX')
plt.plot(t,error_arx_average,'-.',label='ARX(p)')
plt.legend()
plt.yscale('log')
plt.savefig('ARX',format='eps',dpi=1000)
plt.show()


num = list(range(len(x)))
plt.plot(num,x,'-')
plt.legend()
plt.yscale('log')
plt.savefig('data_x',format='eps',dpi=1000)
plt.show()
#t = list(range(len(error_arx)))
#plt.plot(y,error,'-')
#plt.show()
t = list(range(10,l))
fig = plt.figure()
ax_1 = fig.add_subplot(211)
ax_1.plot(t,error[10:l],'-',label='ARX')
plt.legend(loc='upper right')
plt.yscale('log')
#plt.savefig('ARX error',format='eps',dpi=1000)
ax_2 = fig.add_subplot(212)
ax_2.plot(t,error_arx[10:l],'-',label='ARX(p)')
plt.legend(loc='upper right')
plt.yscale('log')
plt.savefig('ARX(p) error',format='eps',dpi=1000)



