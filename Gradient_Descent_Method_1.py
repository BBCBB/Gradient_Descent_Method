# We want to minimize 0.5*X'*H*X+C'*X , which is a QP / H is S++
#Problem generator
import sys
import random
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
opt = input("please determine if you want to adopt a fixed step length or by Armijo Line Search?\n 0 / 1 respectively:")
if opt=='0':
    selected=float(input("enter a fixed step length: "))
n=2

H = [[0 for i in range(n)] for i in range(n)] 
D = [[0 for i in range(n)] for i in range(n)]
C = [[0 for i in range(1)] for j in range(n)]
for i in range(n):
    for j in range(n):
        H[i][j]=random.uniform(0,1)
        if i==j:
            D[i][j]=random.uniform(0,1)+1
        if j==0:    
            C[i][j]=np.random.normal(0,1)
       
H=np.array(H)
D=np.array(D)
C=np.array(C)
#generating a symmetric matrix
H=H+H.transpose()

evalue,evector=eig(H)
H=evector.dot(D).dot(evector.transpose())
                                       
epsilon=sys.float_info.epsilon*100000000

def step_length (H,C,x):                    # step length calculator
    if opt=='0':
        t=selected
        return t
    alpha=0.25
    beta=0.5
    maxit=200
    t=10
    for i in range(maxit):
        g=H.dot(x)+C 
        ft=0.5*(x-t*g).transpose().dot(H).dot((x-t*g))+C.transpose().dot((x-t*g))
        fx=0.5*x.transpose().dot(H).dot(x)+C.transpose().dot(x)
        if ft <= fx-alpha*t*(g.transpose().dot(g)):
            t_armijo=t
            return t_armijo
        else:
            t=beta*t

x=np.array([[0],[0]])                          # initial guess        
x1h=x[0].tolist()                              # to save the history
x2h=x[1].tolist()                              # to save the history      
maxiter=300
for l in range(1,maxiter+1):
    step= step_length (H,C,x)
    g=H.dot(x)+C                               # the gradient of the cost function
    x=x-step*g                                 # updating the value of the decision variable
    x1h.append(x[0].tolist()[0])
    x2h.append(x[1].tolist()[0])
    if np.linalg.norm(g)<=epsilon:
        opt_val=0.5*x.transpose().dot(H).dot(x)+C.transpose().dot(x)
        opt_var=x
        Iterations=i
        break
    elif l==maxiter:
        print('The maximum iterations allowed is not enough to obtain a solution under the determined tolerance')
        break
    else:
        continue


def f(a,b):                                 # defining a function to help plotting the contour lines
    t=np.array([[a],[b]])
    y=0.5*t.transpose().dot(H).dot(t)+C.transpose().dot(t)
    return  y.tolist()[0][0]

if i!=maxiter:
    lb=int(x[0].tolist()[0])-1.5
    ub=int(x[1].tolist()[0])+1.5
    x1 = np.linspace(lb, ub, 25)
    x2 = np.linspace(lb, ub, 25)
    # z=np.zeros((len(x1), len(x2)))
    z = [[0 for i in range(len(x1))] for i in range(len(x2))] 
    c1=0
    c2=0
    for i in x1:
        for j in x2:
            # t = np.array([i,j])
            z[c1][c2] = f(i,j)
            c2=c2+1
        c1=c1+1
        c2=0 
    z=np.array(z).ravel().reshape((len(x1),len(x2)))
    plt.figure(figsize=(8,6))
    plt.contourf(x1, x2, z, levels=10)
    plt.plot(x1h, x2h, 'r*')
    plt.colorbar()
