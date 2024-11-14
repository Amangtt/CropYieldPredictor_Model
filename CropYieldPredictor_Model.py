import numpy as np
import math,copy
import matplotlib.pyplot as plt

X_train = np.array([
    [800, 20, 7, 200, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Amhara, Maize
    [600, 22, 6, 150, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Oromia, Wheat
    [700, 19, 8, 100, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Tigray, Barley
    [1000, 25, 9, 300, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], # SNNPR, Coffee
    [400, 30, 5, 50, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],   # Somali, Sorghum
    [500, 28, 6, 70, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],   # Benishangul, Millet
    [900, 18, 8, 250, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Addis Ababa, Vegetables
    [300, 35, 4, 20, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],   # Afar, Teff
    [500, 26, 7, 80, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],   # Dire Dawa, Pulses
    [1200, 27, 10, 400, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # Gambela, Cassava
])
y_train = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 6.0, 1.8, 3.2, 4.8])
w_int = np.random.rand(X_train.shape[1]) * 0.0001  
b_int = 0
def cost(x,y,w,b):
    m=x.shape[0]
    cost=0.
    for i in range(m):
        f_wb= np.dot(w,x[i])+b
        cost+= (f_wb- y[i])**2
    cost= cost/(2*m)
    return cost

def comp_grad(x,y,w,b):
    m, n = x.shape
    d_dw = np.zeros((n,))
    d_db = 0.
    for i in range(m):
        f_wb = (np.dot(x[i], w) + b) - y[i]
        for j in range(n):
            d_dw[j] += f_wb * x[i, j]
        d_db += f_wb

    d_dw = d_dw / m
    d_db = d_db / m
    return d_dw, d_db
def grad_dec(x, y, w_in, b_in, alpha, iter):
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(iter):
        d_dw, d_db = comp_grad(x, y, w, b)

        
        w = w - alpha * d_dw
        b = b - alpha * d_db

        if i < 10000:
            j_history.append(cost(x, y, w, b))
        if i % math.ceil(iter / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")

    return w, b, j_history
initial_w = w_int
initial_b = b_int
alpha = 0.000001  
itrr = 10000    


f_w, f_b, j_history = grad_dec(X_train, y_train, initial_w, initial_b, alpha, itrr )


print(f"b, w found by gradient descent: {f_b}, {f_w}")


m, _ = X_train.shape
predictions = np.dot(X_train, f_w) + f_b
plt.scatter(range(m),y_train, marker='x', c='r')
plt.scatter(range(m),predictions, marker='o', c='b')
plt.show()