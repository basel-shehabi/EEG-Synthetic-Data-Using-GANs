import numpy as np

x = np.load('D:\Ciaran Python Data\Transfer-SARzWXT6VsetANd8\motorImagery\S1Processeds.npz')
print(len(x))
print(len(x['x']))
new_x = x['x']
print(new_x.shape)