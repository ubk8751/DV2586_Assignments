import numpy as np

def f_dx(x,y):
    return 2*x - 3 * np.sin(3*x) + y + 2

def f_dy(x,y):
    return x + 2*y

def f(x,y):
    return np.square(x) + np.square(y) + x * (y + 2) + np.cos(3 * x)
    
def gradDecent(x,y, n, lr, lst):
    for i in range(n):
        x = x - lr*f_dx(x,y)
        y = y - lr*f_dy(x,y)
        z = f(x, y)
        lst.append([i, z])
    return lst
    

if __name__ == "__main__":
    startValX = 3
    startValY = 5
    n = 100
    lr = 0.01
    resList = []

    gradDecent(startValX, startValY, n, lr, resList)

    print(resList)