from cProfile import label
from syslog import LOG_DAEMON
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

"""
Xt=X.transpose()
Xtx=np.matmul(Xt,X)
Xtx_inv=np.linalg.inv(Xtx)
print(Xt.dot(y))
theta=Xtx_inv.dot(Xt.dot(y))
print("Theta ",theta)
"""

def get_info(X,y,calc_y_hat,):

    info=[]

    calc_theta=lambda X,y: np.linalg.inv(np.matmul(X.transpose(),X)).dot(X.transpose()).dot(y)
    theta=calc_theta(X,y)
    for i in theta:
        print(round(i,2))
    info.append(theta)

    store_y_hat=[]
    x=[1,2,3,4,5]
    for i in x:
        store_y_hat.append(round(calc_y_hat(i),2))

    print("Yhat vals ",store_y_hat)
    info.append(store_y_hat)

    calc_error=lambda y, y_hat: (y-y_hat)**2
    store_errors=[]
    for i in range(len(y)):
        store_errors.append(round(calc_error(y[i],store_y_hat[i]),2))
    print("Errors ",store_errors)
    #info.append(store_errors)

    total_error=round(sum(store_errors)/2,2)
    print("total Error: ", total_error)
    info.append(total_error)

    return info





def plot(x,info_vec):

    
    #plt.plot(x,y1, label="true line",  marker='o', markerfacecolor='purple', markersize=10)

    plt.xlabel('x - axis')
    plt.ylabel('y - axis')

    plt.legend()
    plt.show()






X1=np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
X2=np.array([[1,1,1],[1,2,4],[1,3,9],[1,4,16],[1,5,25]])
X3=np.array([[1,1,1,1,1,1],[1,2,4,8,16,32],[1,3,9,27,81,243],[1,4,16,64,256,1024],[1,5,25,125,625,3125]])

y1=np.array([1,3,2,3,5])
y2=np.array([1,5,9,15,25])

calc_y_hat_1=lambda x: 0.6-(0.2*x)+(x**2)
calc_y_hat_2=lambda x: 435.78+(304.1*x)-(51.94*x**2)-(13.9*x**3)+(5.37*x**4)-(0.28*x**5)
calc_y_hat_3=lambda x: 0.4+0.8*x
calc_y_hat_4=lambda x: x
calc_y_hat_5=lambda x: 1+2*x
calc_y_hat_6=lambda x: -6.4 +5.8*x

p1=get_info(X1,y2,calc_y_hat_6)
p2=get_info(X2,y2,calc_y_hat_1)
p3=get_info(X3,y2,calc_y_hat_2)

plot()



