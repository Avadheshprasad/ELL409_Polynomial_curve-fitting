#!/usr/bin/env python
# coding: utf-8

# In[36]:

import numpy as ny
from numpy import genfromtxt
import matplotlib.pyplot as pl
# data_pts = genfromtxt('gaussian.csv', delimiter=',')

# data_pts = genfromtxt('saru.csv', delimiter=',')
col_x = ny.genfromtxt('train.csv', delimiter=',',usecols=0,encoding=None,dtype=None,skip_header=1)
col_t= ny.genfromtxt('train.csv', delimiter=',',usecols=1,encoding=None,dtype=None,skip_header=1)
col_x1= ny.genfromtxt('test.csv', delimiter=',',usecols=0,encoding=None,dtype=None,skip_header=1)
n=len(col_x)

dx=ny.zeros(n)
dt=ny.zeros(n)
for i in range(n):
    p=str(col_x[i]).split("/")
    dx[i]=(int(p[0])+int(p[2])*12)
    dt[i]=float(col_t[i])
# # print(col_x)
# print(dx)

n1=len(col_x1)
dx1=ny.zeros(n1)
for i in range(n1):
    p1=str(col_x1[i]).split("/")
#     print(p1)
    dx1[i]=(int(p1[0])+int(p1[2])*12)

# print(dx1)  
def d_matrix(X,M):
    N=len(X)
    p=ny.matrix(ny.zeros((N,M), dtype = float))    
    for i in range(N):
        p[i,0]=1
        for j in range(M//2):
            p[i,2*j+1]=ny.sin((j+1)*57.6*X[i]/110)
            p[i,2*j+2]=ny.cos((j+1)*57.6*X[i]/110)
    return p

def moore_inv(p,lamb,t):
    m=len(p.T)
    z=lamb*ny.identity(m)+ny.matmul(p.T,p)
    y=ny.linalg.pinv(z)
    final= ny.matmul(y,p.T)
    w=ny.dot(final,ny.transpose(t))
    return w


def value(w,z):
    m=len(w.T)
    dm=d_matrix(z,m)
    W=w.reshape((m,1))
    res=ny.matmul(dm,W)
    return res

def SSE(w,x,t):
    y=value(w,x)
    h=(ny.subtract(t,y.T))
#     print(h)
    square_error=ny.square(h)
    res= ny.sum(square_error)
    return res/2

def Erms(w,x,t):
    y=value(w,x)
    l=len(t)
    square_error=ny.square((ny.subtract(t,y.T)))
    res= (ny.sum(square_error))/l
    return res**0.5

# c=100
# for l in range(1):   
#     p=d_matrix(dx[:c],3)
#     # print(p)
#     w=moore_inv(p,l,dt[:c])
#     #     print(w)
#     print(ny.ravel(value(w,dx1)))
# #     print(SSE(w,dx[c:],dt[c:]),l)


# x=dx[:90]
# t=dt[:90]
# x1=dx[:110]
# t1=dt[:110]
# phi=d_matrix(x,3)
# w=moore_inv(phi,0,t)
# a=value(w,x1)

# sorted_x, sorted_y = zip(*sorted(zip(x1,ny.ravel(a))))

# sort_x, sort_y = zip(*sorted(zip(x1,t1)))
# y=ny.subtract(sorted_y,sort_y)
# print("mean=",ny.mean(y))
# print("variance=",ny.var(y))
# # # print(sorted_y)
# # pl.plot(sorted_x,sorted_y,label = 'curve fitting ')
# pl.plot(sort_x,sort_y,label = 'Actual Curve')
# pl.xlabel('y-values')
# pl.ylabel('x-values')
# pl.title('Curve diagram')

# pl.legend()
# pl.show()



# p=d_matrix(dx,3)
# # print(p)
# w=moore_inv(p,0,dt)
# print(w)
# # print(ny.ravel(value(w,dx1)))
# # #     print(SSE(w,dx[c:],dt[c:]),l)



# x=dx[:100]
# t=dt[:100]
# x1=dx[:110]
# t1=dt[:110]
# phi=d_matrix(x,3)
# w=moore_inv(phi,0,t)
# a=value(w,x1)

# sorted_x, sorted_y = zip(*sorted(zip(x1,ny.ravel(a))))

# sort_x, sort_y = zip(*sorted(zip(x1,t1)))
# y=ny.absolute(ny.subtract(sorted_y,sort_y))
# print("mean=",ny.mean(y))
# print("variance=",ny.var(y))
              
# pl.plot(sorted_x,y,label = 'Noise' )
# # pl.plot(sort_x,sort_y,label = 'Actual Curve')
# pl.xlabel('x-values')
# pl.ylabel('y-values')
# pl.title('Noise vs x')

# pl.legend()
# pl.show(   )         
# # print(sorted_y)
# pl.plot(sorted_x,sorted_y,label = 'curve fitting ')
# pl.plot(sort_x,sort_y,label = 'Actual Curve')
# pl.xlabel('y-values')
# pl.ylabel('x-values')
# pl.title('Curve diagram')

# pl.legend()
# pl.show()

# n=80
# N=100
# m=25
# lambd=0



# def error_matrix(n,N,m,lamb):
#     x=dx[:n]
#     t=dt[:n]
#     x1=dx[n:N]
#     t1=dt[n:N]

#     ET=ny.zeros((m,3))
#     for i in range(m):
#         phi=d_matrix(x,2*i+1)
#         w=moore_inv(phi,lamb,t)
#         ET[i][0]=2*i+1
#     #     traing_error
#         ET[i][1]=Erms(w,x,t)
#     #     testing_error
#         ET[i][2]=Erms(w,x1,t1)  
#     return ET

# ET=error_matrix(10,20,8,0)
# ET=error_matrix(80,100,20,0)
# pl.plot(ET[0:,0:1],ET[0:,1:2],label = 'Training')
# pl.plot(ET[0:,0:1],ET[0:,2:3],label = 'Testing')
# pl.xlabel('Degree(m)')
# pl.ylabel('Erms')
# pl.title('Erms vs Degree(m)')

# pl.legend()
# pl.show()

