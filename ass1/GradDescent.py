import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from statistics import median

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

def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"

def create_last_z_val_list(dict):
    lst=[]
    for key in dict:
        lst.append(dict[key][-1])
    lst = np.array(lst)
    return lst

def grouped_df(data, labels):
    df = {}
    for i in range(len(data)):
        df[data[i]] = [labels[i]]
    return df

if __name__ == "__main__":
    # Define the range of x, y values to search over
    x_list = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_list = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Set number of itterations and learning rate
    n = 100
    lr = 0.01
    resdict = {}

    # Find all f(x,y) values for found
    for x in x_list:
        for y in y_list:
            lst = []
            resdict[x,y] = gradDecent(x, y, n, lr, lst)
    
    # Get only the z-values to cluster
    res_list = create_last_z_val_list(resdict)
    km = KMeans(n_clusters=4)
    clusters = km.fit_predict(res_list.reshape(-1,1))
    df = grouped_df(res_list, clusters)
    df = pd.DataFrame.from_dict(df)
    lst0 = []
    lst1 = []
    lst2 = []
    lst3 = []
    lst4 = []
    for key in df:
        if df[key].item() == 0:
            lst0.append([df[key], key])
        if df[key].item() == 1:
            lst1.append([df[key], key])
        if df[key].item() == 2:
            lst2.append([df[key], key])
        if df[key].item() == 3:
            lst3.append([df[key], key])

    # Find the median of each cluster
    lst0_vals = [el[1] for el in lst0]
    lst1_vals = [el[1] for el in lst1]
    lst2_vals = [el[1] for el in lst2]
    lst3_vals = [el[1] for el in lst3]
    m0 = lst0_vals[np.argsort(lst0_vals)[len(lst0_vals)//2]]
    m1 = lst1_vals[np.argsort(lst1_vals)[len(lst1_vals)//2]]
    m2 = lst2_vals[np.argsort(lst2_vals)[len(lst2_vals)//2]]
    m3 = lst3_vals[np.argsort(lst3_vals)[len(lst3_vals)//2]]

    # Plot the clusters and medians
    fig, ax = plt.subplots()
    ax.scatter([el[1] for el in lst0], [el[0] for el in lst0], label=0)
    ax.scatter([el[1] for el in lst1], [el[0] for el in lst1], label=1)
    ax.scatter([el[1] for el in lst2], [el[0] for el in lst2], label=2)
    ax.scatter([el[1] for el in lst3], [el[0] for el in lst3], label=3)
    ax.plot(m0, 0, "o", color="black")
    ax.plot(m1, 1, "o", color="black")
    ax.plot(m2, 2, "o", color="black")
    ax.plot(m3, 3, "o", color="black")
    ax.set_ylabel("Group")
    ax.set_xlabel("Value")
    plt.legend()
    plt.savefig("kmeans.png")

    # Find where each median is in the resdict
    keys = list(resdict.keys())

    idx0 = [float(x) for x in res_list].index(m0)
    idx1 = [float(x) for x in res_list].index(m1)
    idx2 = [float(x) for x in res_list].index(m2)
    idx3 = [float(x) for x in res_list].index(m3)

    xy0 = keys[idx0]
    xy1 = keys[idx1]
    xy2 = keys[idx2]
    xy3 = keys[idx3]

    print(f'Median in group 0 is found at {m0}, with (x,y) = {xy0}')
    print(f'Median in group 1 is found at {m1}, with (x,y) = {xy1}')
    print(f'Median in group 2 is found at {m2}, with (x,y) = {xy2}')
    print(f'Median in group 3 is found at {m3}, with (x,y) = {xy3}')
    
    # Print the gradient descent for the smallest number
    y_axis = resdict[xy0]
    x_axis = list(range(1, 101))
    fig, ax = plt.subplots()
    ax.plot(x_axis, y_axis)
    ax.set_xlabel("Steps")
    ax.set_ylabel("f(x,y)-value")
    ax.set_yticks(np.arange(-5, max(y_axis),step=5))
    ax.grid(True, which="both")
    ax.axhline(0, color='black', linewidth=.5)
    ax.axhline(-1.1301159503969216, color='black', linewidth=.5)
    plt.savefig("gradientdescentx0y8.png")