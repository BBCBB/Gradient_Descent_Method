# We want to minimize 0.5*X'*H*X+C'*X , which is a QP / H is S++
#Problem generator
import sys
import random
import numpy as np
from numpy.linalg import eig

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
#generating a symmetrix matrix
H=H+H.transpose()
evalue,evector=eig(H)
# H=np.matmul(evector,D,evector.transpose())   # a random positive definite matrix
H=evector.dot(D).dot(evector.transpose())


maxiter=500
x=np.array([[0],[0]])                                        # initial guess
epsilon=sys.float_info.epsilon*100
step=0.1                                      
for i in range(1,maxiter+1):
    g=H.dot(x)+C                               # the gradient of the cost function
    x=x-step*g                                 # updating the value of the decision variable
    
    if np.linalg.norm(g)<=epsilon:
        opt_val=0.5*x.transpose().dot(H).dot(x)+C.transpose().dot(x)
        opt_var=x
        Iterations=i
        break
    elif i==maxiter:
        print('The maximum iterations allowed is not enough to obtain a solution under the determined tolerance')
        break
    else:
        continue