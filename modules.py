"""

Modules defined in this file:

        * Attention: mechanism that generates glimpses of an image given glimpse parameters
        * Context: network that takes in a downsampled image and extracts global features
        * Glimpse: network that takes Attenion output and Glimpse params and extracts features
        * Recurrent: Contains two LSTMs
        *   * LSTM1: Takes Glimpse output and hidden1 and is used for Classification and LSTM2 in
        *   * LSTM2: Takes LSTM1 output and hidden2 and is used for Emission
        * Classification: Takes LSTM1 output and is used for image glimpse classification
        * Emission: Takes LSTM2 output and is used to decide next glimpse parameters
        *

"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import copy


class Attention(nn.Module):
    """
        Class that defines the spatial transformation attention mechanism.
    """

    _meshgrid = None
    _N, theta = None, None
    _width, _height = 26, 26

    def __init__(self, glimpse_size):
        super(Attention, self).__init__()
        self.grid_generator()
        self.glimpse_size = glimpse_size

    def forward(self, input_batch, A):
        """
        This function is used to implement the feature sampling after the grid generation.

        :return:
        """
        A = A.reshape((-1, 2, 3))
        (N, _, H, W) = input_batch.shape

        # We init the width and height of each image
        self._N = N
        # self._width, self._height = W, H
        #self.theta = torch.zeros(A.shape, requires_grad=True).float().to(input_batch.device)
        #self.theta = A.detach()
        #self.theta = A
        
        ### CRUCIAL TO MAKING SURE TRANSFORMATION IS DONE CORRECTLY
        #self.theta[:,:,2] = 2 * self.theta[:,:,2] - (1 - self.theta / 100)
        #old_theta_0 = copy.deepcopy(self.theta[:,0,2])
        #self.theta[:,0,2] = 2 * self.theta[:,1,2] - (1 - self.theta[:,0,0])
        #self.theta[:,1,2] = 2 * old_theta_0 - (1 - self.theta[:,1,1])

        # self.theta[:,:,:2] = A[:,:,:2]
        
        
        dA = torch.zeros(A.shape, requires_grad=True).float().to(input_batch.device)
        dA[:,0,2] = A[:,0,2] - (1 - A[:,0,0])
        dA[:,1,2] = A[:,1,2] - (1 - A[:,1,1])
        
        A = A + dA
        
        #A[:,0,2] = 2 * A[:,0,2]
        #A[:,0,2] = A[:,0,2] - (1 - A[:,0,0])
        #A[:,1,2] = 2 * A[:,1,2] 
        #A[:,1,2] = A[:,1,2] - (1 - A[:,1,1])
        
        
        #A[:,0,2] = 2 * A[:,1,2] - (1 - A[:,0,0])
        #A[:,1,2] = 2 * A[:,0,2] - (1 - A[:,1,1])
        
        sampling_coords = torch.matmul(A, torch.Tensor(self._meshgrid).to(input_batch.device))
        sampling_coords = sampling_coords.transpose(1,2).reshape((N, self._height, self._width, 2))
        X = F.grid_sample(input_batch.transpose(2,3), sampling_coords).transpose(2,3)

        # X = torch.empty((N,) + self.glimpse_size)
        # for im_num in range(self._N):
        #     for col in range(self.glimpse_size[0]):
        #         for row in range(self.glimpse_size[0]):
        #             X[im_num][row][col] = input_batch[im_num][0][row][col] * \
        #                                   torch.max(torch.tensor(0).float(), (1 - torch.abs(sampling_coords[im_num][0][row * 100 + col] - row))) * \
        #                                   torch.max(torch.tensor(0).float(),
        #                                             (1 - torch.abs(sampling_coords[im_num][1][row * 100 + col] - row)))

        #A[:,0,2] = (A[:,0,2] + (1 - A[:,0,0]))
        #A[:,0,2] = A[:,0,2] / 2
        #A[:,1,2] = (A[:,1,2] + (1 - A[:,1,1]))
        #A[:,1,2] = A[:,1,2] / 2
        A = A - dA


        return X

    def grid_generator(self):
        """
        This function is used to create the meshgrid coords to be used
        :return:
        """

        x_t, y_t = np.meshgrid(np.linspace(-1, 1, self._width), np.linspace(-1, 1, self._height))
        self._meshgrid = np.concatenate((np.matrix(x_t.flatten()), np.matrix(y_t.flatten()), np.ones((1, self._width ** 2))),
                                        axis=0)


class Context(nn.Module):

    """ 
        Context module.
        param filter_widths: convolution widths by layer
        param filter_depths: convolution depths by layer
    """

    conv_out_size = 512

    def __init__(self, filter_widths, filter_depths, in_size=(1, 1, 12, 12)):
        super(Context, self).__init__()
        self.conv_layers = nn.Sequential()

        for i in range(len(filter_depths)-1):
            
            self.conv_layers.add_module("conv"+str(i+1), nn.Conv2d(in_channels=filter_depths[i],
                                                                   out_channels=filter_depths[i+1],
                                                                   kernel_size=filter_widths[i],
                                                                   padding=0))
            # self.conv_layers.add_module("bn"+str(i+1), nn.BatchNorm2d(filter_depths[i+1]))
        
        # self.conv_out_size = self.conv_output(in_size)
        self.conv_layers.add_module("flatten0", Flatten())
        """
        self.conv_layers.add_module("fc0", nn.Linear(self.conv_out_size, 1024))
        self.conv_layers.add_module("relu", nn.ReLU())
        """
    """
        param x: tensor of size (N, 1, 12, 12)
    """
    def forward(self, x):
        features = self.conv_layers(x)
        # return features.view((2, 1, -1, int(1024/2)))
        #return features.reshape((1, -1, 2, int(1024/2))).permute(2, 0, 1, 3).contiguous()
        return features.unsqueeze(0)
        
    def conv_output(self, shape):
        inp = Variable(torch.rand(*shape))
        out_f = self.forward(inp)
        num_nodes = out_f.data.reshape(1, -1).size(1)
        return num_nodes


class Glimpse(nn.Module):

    """
        Glimpse module.
        param filter_widths: convolution widths by layer
        param filter_depths: convolution depths by layer
        param fcs: list of length k of integers, used to create k-1 layers
        param in_size: tuple indicating the shape of a default 1-batch input
        param A_size: parameter for the number of nodes taken up when A transform is added
    """
    # @TODO: Add stride?
    def __init__(self, filter_widths, filter_depths, fc_size, in_size=(1, 1, 26, 26), A_size=6):
        super(Glimpse, self).__init__()
        self.conv_layers = nn.Sequential()
        
        for i in range(len(filter_widths)-1):

            padding = 0            
            if i in [0, 2, 3, 4]:
                padding = int(filter_widths[i] / 2)
            self.conv_layers.add_module("conv"+str(i+1), nn.Conv2d(in_channels=filter_depths[i],
                                                                   out_channels=filter_depths[i+1],
                                                                   kernel_size=filter_widths[i],
                                                                   padding=padding))

            #self.conv_layers.add_module("bn"+str(i+1), nn.BatchNorm2d(filter_depths[i+1], momentum=0.05))            
            self.conv_layers.add_module("relu"+str(i+1), nn.ReLU())
            
            if (i == 1 or i == 3):
                self.conv_layers.add_module("pool"+str(i), nn.MaxPool2d(2, stride=2))
                
        self.conv_layers.add_module("flatten", Flatten())
        conv_fc_size = self.conv_output(in_size)
        
        # self.conv_layers.add_module("pool", nn.MaxPool2d(self.filter_depths[-1]))
        self.conv_layers.add_module("fc_conv", nn.Linear(conv_fc_size, fc_size[0]))
        #self.conv_layers.add_module("batchnorm_conv", nn.BatchNorm1d(fc_size[0], momentum=0.9))
        self.conv_layers.add_module("relu_conv", nn.ReLU())

        self.a_layer = nn.Sequential()
        self.a_layer.add_module("fc_A", nn.Linear(A_size, fc_size[0]))
        #self.a_layer.add_module("batchnorm_A", nn.BatchNorm1d(fc_size[0], momentum=0.05))
        self.a_layer.add_module('relu_A', nn.ReLU())
        
        """
        self.output_layer = nn.Sequential()
        self.output_layer.add_module("fc", nn.Linear(fc_size[0], fc_size[0]))
        """
        # @TODO: Maybe add relu
        
    def conv_output(self, shape):
        inp = Variable(torch.rand(*shape))
        out_f = self.conv_layers(inp)
        num_nodes = out_f.data.reshape(1, -1).size(1)
        return num_nodes
        
    def forward(self, x, A_in):
        A_in = A_in.reshape((-1, 6))
        product = self.conv_layers(x) * self.a_layer(A_in)
        return product


class Classification(nn.Module):
    """
        Classification module.
        param fcs: list of length k of integers, used to create k-1 layers. All layers but final have nonlinearity relu, with final layer having softmax
    """
    # @TODO: Direct softmax on last layer may be numerically unstable
    # @TODO: Experiment with linear output layer and cross entropy loss during training
    def __init__(self, fcs):
        super(Classification, self).__init__()
        self.fcs = fcs
        self.layers = nn.Sequential()
        for i in range(len(fcs) - 1):
            self.layers.add_module("fc"+str(i+1), nn.Linear(fcs[i], fcs[i+1]))
            if i != len(fcs)-2:
                #self.layers.add_module("batchnorm"+str(i+1), nn.BatchNorm1d(fcs[i+1], momentum=0.05))
                self.layers.add_module("nonlinearity"+str(i+1), nn.ReLU())
            """
            else:
                self.layers.add_module("nonlinearity"+str(i+1), nn.Softmax())
            """
                
    """
        param x: tensor of size (N, hidden1_size)
    """
    def forward(self, x):
        return self.layers(x)
    
        
class Emission(nn.Module):

    """
        param fcs: list of length k of integers, used to create k-1 layers. All layers but final have nonlinearity relu
    """
    def __init__(self, fcs):
        super(Emission, self).__init__()
        self.fcs = fcs
        self.layers = nn.Sequential()
        for i in range(len(fcs)-1):
            self.layers.add_module("fc"+str(i+1), nn.Linear(fcs[i], fcs[i+1]))
            self.layers.add_module("tanh", nn.Tanh())
            if i != len(fcs)-2:
                self.layers.add_module("nonlinearity"+str(i+1), nn.ReLU())
                #self.layers.add_module("batchnorm"+str(i+1), nn.BatchNorm1d(fcs[i+1], momentum=0.9))
    
        self.layers[0].weight.data.zero_()
        self.layers[0].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
        
    """
        param x: tensor of size (N, hidden2_size)
    """
    def forward(self, x):
        layers_out = self.layers(x)
        output = torch.zeros(layers_out.shape).float().to(layers_out.device)
        
        output[:,[1,2,3,5]] = layers_out[:,[1,2,3,5]]
        output[:,0] = torch.clamp(layers_out[:,0], 0.0, 1.0)
        output[:,4] = torch.clamp(layers_out[:,4], 0.0, 1.0)
        
        return output

"""
    Helper class to streamline flattening of convolutional outputs in nn.Sequential
"""
class Flatten(nn.Module):
    
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
