import argparse  
from code_part1 import foo
import numpy as ny
from numpy import genfromtxt
# import matplotlib.pyplot as pl


def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=5, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=float, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="", type=str, help = "Read content from the file")
    return parser.parse_args()
    

if __name__ == '__main__':
    args = setup()
    # foo.demo(args)

method=args.method
lambd=args.lamb
polynomial=args.polynomial
b_sz=args.batch_size


file_name=args.X

# file_name='non_gaussian.csv'
data_x = ny.genfromtxt(file_name, delimiter=',',usecols=0,encoding=None,dtype=None,skip_header=0)
data_t= ny.genfromtxt(file_name, delimiter=',',usecols=1,encoding=None,dtype=None,skip_header=0)


def d_matrix(X,M):
    N=len(X)
    p=ny.matrix(ny.zeros((N, M), dtype = float))    
    for i in range(N):
        for j in range(M):
            p[i,j]=(X[i])**j
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

def SGD(X,t,M,Batch_sz,eta,lamb):
    dm = d_matrix(X,M)
    N=len(X)
    d=ny.arange(N)
    ny.random.shuffle(d)
    weights = ny.zeros((M,1))
    for i in range(50000):
        G = 0
        for k in range(Batch_sz):
            G += -ny.transpose(dm[d[k],:])*(t[d[k]]-dm[d[k],:]*weights)
        G+=lamb*weights
        weights -= eta*G
    return(ny.ravel(weights))


if(method=="pinv"):
    phi=d_matrix(data_x,int(polynomial)+1)
    w=moore_inv(phi,lambd,data_t)
    print(f"weights={ny.ravel(w)}")

if(method=="gd"):
    w=SGD(data_x,data_t,int(polynomial)+1,b_sz,0.0001,lambd)
    print(f"weights={w}")


#########################################################3
# def error_vs_m(n,N,m,lamb):
#     x=data_x[:n]
#     t=data_t[:n]
#     x1=data_x[n:N]
#     t1=data_t[n:N]

#     ET=ny.zeros((m,3))
#     for i in range(m):
#         phi=d_matrix(x,i+1)
#         w=moore_inv(phi,lamb,t)
#         ET[i][0]=i
#     #     traing_error
#         ET[i][1]=Erms(w,x,t)
#     #     testing_error
#         ET[i][2]=Erms(w,x1,t1)  
#     return ET

# ET=error_vs_m(10,20,8,0)
# pl.plot(ET[0:,0:1],ET[0:,1:2],label = 'Training-Error')
# pl.plot(ET[0:,0:1],ET[0:,2:3],label = 'Testing-Error')


# ET=error_vs_m(80,100,20,0)
# pl.plot(ET[6:,0:1],ET[6:,1:2],label = 'Training')
# pl.plot(ET[6:,0:1],ET[6:,2:3],label = 'Testing')
# pl.xlabel('polynomial-Degree(m)')
# pl.ylabel('Error(RMS)')
# pl.title('Dataset-100pts : Erms vs m ')

# pl.legend()
# pl.show()
#####################################################################


# def error_vs_m_sgd(n,N,m,lamb):
#     x=data_x[:n]
#     t=data_t[:n]
#     x1=data_x[n:N]
#     t1=data_t[n:N]

#     ET=ny.zeros((m,3))
#     for i in range(m):
#         w=SGD(x,t,i+1,5,0.00001,0.01)
#         ET[i][0]=i
#     #     traing_error
#         ET[i][1]=Erms(w,x,t)
#     #     testing_error
#         ET[i][2]=Erms(w,x1,t1)  
#     return ET

# ET=error_vs_m_sgd(80,100,10,0)
# print(ET)
# # ET=error_vs_m_sgd(10,20,8,0)
# pl.plot(ET[0:,0:1],ET[0:,1:2],label = 'Training-Error')
# pl.plot(ET[0:,0:1],ET[0:,2:3],label = 'Testing-Error')
# pl.xlabel('polynomial-Degree(m)')
# pl.ylabel('SGD-Erms ')
# pl.title('Dataset-100pts :Batch-size=5 : Erms vs m')

# pl.legend()
# pl.show()

######################################################################

# def error_vs_batchsize(n,N,m,bs):
#     x=data_x[:n]
#     t=data_t[:n]
#     x1=data_x[n:N]
#     t1=data_t[n:N]

#     ET=ny.zeros((bs,3))
#     for i in range(bs):
#         w=SGD(x,t,m+1,bs,0.00001,0.01)
#         ET[i][0]=i
#     #     traing_error
#         ET[i][1]=Erms(w,x,t)
#     #     testing_error
#         ET[i][2]=Erms(w,x1,t1)  
#     return ET

# ET=error_vs_batchsize(80,100,2,30)
# print(ET)
# # ET=error_matrix(10,20,8,0)
# pl.plot(ET[0:,0:1],ET[0:,1:2],label = 'Training-Error')
# # pl.plot(ET[0:,0:1],ET[0:,2:3],label = 'Testing-Error')
# pl.xlabel('Batch-size')
# pl.ylabel('SGD-Erms')
# pl.title('Dataset-100pts : m=4: Erms vs Batch_size')

# pl.legend()
# pl.show()

########################################################
##noise curve


# x=data_x[:80]
# t=data_t[:80]
# x1=data_x[:100]
# t1=data_t[:100]
# phi=d_matrix(x,23)
# w=moore_inv(phi,0,t)
# a=value(w,x1)

# sorted_x, sorted_y = zip(*sorted(zip(x1,ny.ravel(a))))

# sort_x, sort_y = zip(*sorted(zip(x1,t1)))
# y=ny.subtract(sorted_y,sort_y)
# print("mean=",ny.mean(y))
# print("variance=",ny.var(y))
# # print(sorted_y)
# pl.plot(sorted_x,sorted_y,label = 'Polynomial curve fitting , overfitting')
# pl.plot(sort_x,sort_y,label = 'Actual Curve')
# pl.xlabel('y-values')
# pl.ylabel('x-values')
# pl.title('Curve diagram')

# pl.legend()
# pl.show()
####################################################
# def error_vs_lambda(n,N,m,lamb):
#     x=data_x[:n]
#     t=data_t[:n]
#     x1=data_x[n:N]
#     t1=data_t[n:N]

#     ET=ny.zeros((lamb,3))
#     for i in range(lamb):
#         phi=d_matrix(x,m+1)
#         w=moore_inv(phi,i/100000,t)
#         ET[i][0]=i/100000
#     #     traing_error
#         ET[i][1]=Erms(w,x,t)
#     #     testing_error
#         ET[i][2]=Erms(w,x1,t1)  
#     return ET

# ET=error_vs_lambda(80,100,11,1000)
# # pl.plot(ET[0:,0:1],ET[0:,1:2],label = 'Training-Error')
# pl.plot(ET[0:,0:1],ET[0:,2:3],label = 'Testing-Error')


# # ET=error_matrix(80,100,20,0)
# # pl.plot(ET[6:,0:1],ET[6:,1:2],label = 'Training')
# # pl.plot(ET[6:,0:1],ET[6:,2:3],label = 'Testing')
# pl.xlabel('lambda')
# pl.ylabel('Error(RMS)')
# pl.title('Dataset-100pts:m=11 : Erms vs lambda ')

# pl.legend()
# pl.show()

##########################################################
# def error_vs_validation(n,N,m,lamb):
#     ET=ny.zeros((n,3))
#     for i in range(n):
#         x=data_x[:i]
#         t=data_t[:i]
#         x1=data_x[i:N]
#         t1=data_t[i:N]
        
#         phi=d_matrix(x,m+1)
#         w=moore_inv(phi,0,t)
#         ET[i][0]=i
#     #     traing_error
#         ET[i][1]=Erms(w,x,t)
#     #     testing_error
#         ET[i][2]=Erms(w,x1,t1)  
#     return ET

# ET=error_vs_validation(88,100,13,0)
# pl.plot(ET[30:,0:1],ET[30:,1:2],label = 'Training-Error')
# pl.plot(ET[30:,0:1],ET[30:,2:3],label = 'Testing-Error')


# # ET=error_matrix(80,100,20,0)
# # pl.plot(ET[6:,0:1],ET[6:,1:2],label = 'Training')
# # pl.plot(ET[6:,0:1],ET[6:,2:3],label = 'Testing')
# pl.xlabel('validation set size n')
# pl.ylabel('Error(RMS)')
# pl.title('Dataset-80pts:m=13 : Erms vs Validation set ')

# pl.legend()
# pl.show()