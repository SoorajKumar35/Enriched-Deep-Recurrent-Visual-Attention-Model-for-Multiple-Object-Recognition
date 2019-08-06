import numpy as np
import torch
import torch.nn as nn


class STAM(nn.Module):
    """
    Class that defines the spatial transformation attention mechanism.
    """

    _meshgrid = None
    _trans_params = None
    _width, _height = None, None

    def __init__(self, A, input_batch):
        super(STAM, self).__init__()

        (N, _, H, W) = input_batch.shape

        # We init the width and height of each image
        self._N = N
        self._width, self._height = W, H
        self.theta = A

        self._grid_generator()

        # self._create_fcs_layers()
        # self.theta = self.forward()
        self._sampler(input_batch)

    def forward(self, input_batch):
        """
        This function is the forward function for the Spatial Tranformation attention mechanism
        :return:
        """
        return self.layers(input_batch)

    def _create_fcs_layers(self):
        self.layers = nn.Sequential()
        for i in range(len(self.fcs) - 1):
            self.layers.add_module("fc" + str(i+1), nn.Linear(self.fcs[i], self.fcs[i + 1]))
        self.layers.add_module("fc" + str(len(self.fcs)),
                               nn.Linear(self.fcs[len(self.fcs) - 1],
                               self.fcs[len(self.fcs)]))

    def _sampler(self, input_batch):
        """
        This function is used to implement the feature sampling after the grid generation.

        :return:
        """

        sampling_coords = np.matmul(self.theta, self._meshgrid)
        X = np.empty((self._height, self._width))
        for im_num in range(self._N):
            for col in range(self._width):
                for row in range(self._height):
                    X[im_num][row][col] = input_batch[im_num][row][col] *\
                                  np.max(0, 1 - np.abs(sampling_coords[0][row*100 + col] - row)) *\
                                  np.max(0, 1 - np.abs(sampling_coords[1][row*100 + col] - col))
        return X

    def _grid_generator(self):
        """
        This function is used to create the meshgrid coords to be used
        :return:
        """

        x_t, y_t = np.meshgrid(np.linspace(-1, 1, self._width), np.linspace(-1, 1, self._height))
        self._meshgrid = np.concatenate((np.matrix(x_t.flatten()), np.matrix(y_t.flatten()), np.ones((1, 100**2))),
                                        axis=0)
