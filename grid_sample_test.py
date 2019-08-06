import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy

w,h=26, 26

x_t, y_t = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
meshgrid = np.concatenate((np.matrix(x_t.flatten()), np.matrix(y_t.flatten()), np.ones((1, w**2))), axis=0)






images = np.load('X_train.npy').reshape((60000,1,100,100))
labels = np.load('y_train.npy')
print(images.shape)
im = images[[0]]
label = labels[0]
print(label)
plt.imshow(im[0,0])
plt.show()


theta = torch.Tensor([[0.28,0, 0.64],[0,0.28, 0.28]])
print(theta[1,2], theta[0,0], theta[0,2], theta[1,1], 1-theta[0,0]/100, 1-theta[1,1]/100)
#old_theta_0 = copy.deepcopy(theta[0,2])
#theta[0,2] = 2 * theta[1,2] - (1 - theta[0,0])
#theta[1,2] = 2 * old_theta_0 - (1 - theta[1,1])

theta[0,2] = 2 * theta[0,2] - (1 - theta[0,0])
theta[1,2] = 2 * theta[1,2] - (1 - theta[1,1])


"""
im = torch.zeros((100,100))
for i in range(100):
    for j in range(100):
        im[i,j] = min(i,j)
im = im.unsqueeze(0).unsqueeze(0)
"""

for i in range(11):
    for j in range(11):
        theta = torch.Tensor([[0.7, 0, (i/10)], [0, 0.7, (j/10)]])
        theta[0,2] = 2 * theta[0,2] - (1 - theta[0,0])
        theta[1,2] = 2 * theta[1,2] - (1 - theta[1,1])
        
        print(theta)
        
        
        sampling_coords = torch.matmul(theta, torch.Tensor(meshgrid))
        sampling_coords = sampling_coords.transpose(0,1).reshape((1, h, w, 2))
        X = nn.functional.grid_sample(torch.Tensor(im).transpose(2,3), sampling_coords).transpose(2,3)
        plt.imshow(X[0,0])
        plt.show()
