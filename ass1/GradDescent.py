import numpy as np
import matplotlib.pyplot as plt

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
        lst.append(z)
    return lst
    

if __name__ == "__main__":
    startValX = 3
    startValY = 5
    n = 100
    lr = 0.01
    resList = []

    gradDecent(startValX, startValY, n, lr, resList)
    x_axis = list(range(1, 101))
    fig, ax = plt.subplots()
    ax.plot(x_axis, resList)
    ax.set_xlabel("Steps")
    ax.set_ylabel("f(x,y)-value")
    ax.set_yticks(np.arange(-5, 55,step=5))
    ax.grid(True, which="both")
    ax.axhline(0, color='black', linewidth=.5)
    ax.axhline(-2.3, color='black', linewidth=.5)
    plt.show()
    print(resList[-1])