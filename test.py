# Dummy script used just for testing .npz files and loading. 
import numpy as np

x = np.load('D:\Ciaran Python Data\Transfer-SARzWXT6VsetANd8\motorImagery\Subject1Test1.npz')
new_x = x['x']
print(new_x.shape)