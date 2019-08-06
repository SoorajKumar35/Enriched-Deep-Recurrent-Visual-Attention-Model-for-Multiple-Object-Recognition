import torch
import torch.nn as nn
import modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class EDRAM(nn.Module):
    
    """
        Constructs EDRAM network. The network has several subcomponents.
        * -------------------------------------------------
        * Attention: mechanism that generates glimpses of an image given glimpse parameters
        * Context: network that takes in a downsampled image and extracts global features
        * Glimpse: network that takes Attention output and Glimpse config and extracts features
        * Recurrent: Contains two LSTMs
        *   * LSTM1: Takes Glimpse output and hidden1 and is used for Classification and LSTM2 in
        *   * LSTM2: Takes LSTM1 output and hidden2 and is used for Emission
        * Classification: Takes LSTM1 output and is used for image glimpse classification
        * Emission: Takes LSTM2 output and is used to decide next glimpse parameters
        *
        * The code for all of these modules can be found in modules.py
        
        param config: dict containing the important hyperparameters    
    """
        
    def __init__(self, config):

        super(EDRAM, self).__init__()

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.A_size = 6
        self.T = config['T']
        self.C = config['n_classes']
        self.context_downsample = config['context_downsample']
        self.glimpse_size = config['glimpse_size']
        self.hidden1_size = config['hidden1']
        self.hidden2_size = config['hidden2']
        self.context_filter_widths, self.context_filter_depths = config['context_filters']
        self.glimpse_filter_widths, self.glimpse_filter_depths = config['glimpse_filters']
        self.glimpse_fcs = config['glimpse_fcs']
        self.classification_fcs = config['classification_fcs']
        self.emission_fcs = config['emission_fcs']
        self.LA_weights = torch.from_numpy(np.array(config['LA_weights'])).float().to(self.device)
        self.a1, self.a2 = config['loss_term_weights']
        
        # assert(self.glimpse_fcs[-1] == self.hidden1_size)
        # assert(self.glimpse_filter_depths[0] == self.context_filter_depths[0] == 1)
        # assert(self.classification_fcs[-1] == self.C)
        # assert(self.hidden2_size == self.emission_fcs[0])
        # assert(self.hidden1_size == self.classification_fcs[0])
        # assert(len(self.context_filter_widths)+1 == len(self.context_filter_depths))
        # assert(len(self.glimpse_filter_widths)+1 == len(self.glimpse_filter_depths))
        # assert(self.A_size == self.emission_fcs[-1])

        self.Attention = modules.Attention(self.glimpse_size).to(self.device)
        self.Context = modules.Context(self.context_filter_widths, self.context_filter_depths).to(self.device)
        self.Glimpse = modules.Glimpse(self.glimpse_filter_widths, self.glimpse_filter_depths, self.glimpse_fcs).to(self.device)
        self.LSTM1 = nn.LSTM(input_size=self.glimpse_fcs[-1], hidden_size=self.hidden1_size).to(self.device)
        self.LSTM2 = nn.LSTM(input_size=self.hidden1_size, hidden_size=self.hidden2_size).to(self.device)
        self.Classification = modules.Classification(self.classification_fcs)
        
        self.apply(EDRAM.weights_initialize)
        self.Emission = modules.Emission(self.emission_fcs)


        self.CEloss = nn.CrossEntropyLoss()

    """
        forward takes in N images in a tensor of size (N, 1, H, W), and performs the feed forward operation of EDRAM.
        
        param x:  torch float cuda tensor of size (N, 1, H, W). Contains one digit and cluttered noise.
        
        return: locations       - (N, T, |A|) torch float cuda tensor of our T predictions of the location of the digit in all N images.
                classifications - (N, T, C) torch float cuda tensor of our T classifications of the T glimpses our network took
        
    """
    def forward(self, x, iteration=1):
    
        (N, _, H, W) = x.shape
    
        # Takes broad glimpse at downscaled full image, becomes input to context network
        context_in = nn.functional.interpolate(x, self.context_downsample)
        
        # Init hidden states
        hidden1 = self.init_hidden(N)
        h2 = self.Context(context_in)
        c2 = self.init_hidden2_c(N)
        hidden2 = (h2, c2)
        
        #print(hidden1[0].shape, hidden1[1].shape, h2.shape, c2.shape)
        # Init glimpse parameters
        A = self.A_init(N)
        
        # First array returned in our forward function, stores locations of glimpses
        locations = torch.zeros((N, self.T, 6)).float().to(self.device)
        
        # Second array returned in our forward function, stores class probabilities of glimpses
        classifications = torch.zeros((N, self.T, self.C)).to(self.device)
        
        # Loop through T glimpses and classifications
        for i in range(self.T):      
            
            locations[:, i] = A.reshape((N,6))
            # Return array of (N, 1, self.glimpse_size, self.glimpse_size), stores our glimpses
            xt = self.Attention(x, A)
            
            """
            if iteration == 0:
                print(locations.shape)
                print(locations[0])
                plt.imshow(x[0,0].cpu())
                plt.show(block=False)
                plt.imshow(xt[0,0].cpu().detach())
                plt.show(block=False)
                plt.pause(0.001)
            """
        
            # Forwards xt and A through Glimpse, returning (N, glimpse_out) size tensor
            glimpse_out = self.Glimpse(xt, A)
            
            
            # Forwards glimpse_out through LSTM1, LSTM1_out through LSTM2
            LSTM1_out, hidden1 = self.LSTM1(glimpse_out.unsqueeze(0), tuple(hidden1))
            (h1, c1) = hidden1
            LSTM2_out, hidden2 = self.LSTM2(h1, tuple(hidden2))
            (h2, c2) = hidden2
            

            LSTM1_out = LSTM1_out.reshape((LSTM1_out.shape[1], LSTM1_out.shape[2]))
            LSTM2_out = LSTM2_out.reshape((LSTM2_out.shape[1], LSTM2_out.shape[2]))

            # Make class predictions
            #classification_prediction = self.Classification(h1.squeeze(0))
            classification_prediction = self.Classification(LSTM1_out)
            
            # Decide next glimpse parameters
            #A = self.Emission(h2.squeeze(0))
            A = self.Emission(LSTM2_out)
            # Store locations, predictions
            classifications[:, i] = classification_prediction
            
        return locations, classifications
        
    """
        Loss function of our network with respect to ground truth targets
        Formula: L = a1 * Ly + a2 * LA
                    with Ly = -log classifications[i], where i is the ground truth image label
                    and LA = weighted squared distance between locations[i] and ground truth locations[i]
                    
        
        param locations: (N, T, |A|) torch float cuda tensor of our T predictions of the location of the digit in all N images
        param classifications: (N, T, C) torch float cuda tensor of our T classifications of the T glimpses our network took
        param targets: (N, |A|+1) torch float cuda tensor of ground truth locations and labels of digits
        
        return: torch 1-element tensor
    """
    def loss(self, locations, classifications, targets):
        
        N = locations.shape[0]
        gt_classes = targets[:, self.T:].long()
        
        #expanded_targets = gt_classes.unsqueeze(1).expand(N, self.T, self.C).float()

        #print(classifications, expanded_targets)
        #print( classifications.gather(2, expanded_targets) )
        #Ly = torch.m( -torch.log( classifications.gather(2, expanded_targets) ) )
        Ly = self.CEloss(classifications[:,0], gt_classes[:,0])
        
        for i in range(1, self.T):
            Ly = Ly + self.CEloss(classifications[:,i], gt_classes[:,0])
        Ly = Ly / self.T
        
        #Ly = self.CEloss(classifications, gt_classes.unsqueeze(1).expand(N, self.T, self.C))
        
        LA_sqdiff = torch.mean(torch.mean((locations - targets[:, :6].unsqueeze(1)) ** 2, dim=1), dim=0)
        LA = torch.dot(LA_sqdiff, self.LA_weights.float())
        
        Loss = self.a1 * Ly + self.a2 * LA
        
        return Ly, LA
        
    def init_hidden2_c(self, N):
        return torch.zeros((1, N, 512)).to(self.device)

    def init_hidden(self, N):
        return torch.zeros((2, 1, N, 512)).to(self.device)

    def A_init(self, N):
        w = torch.zeros((N, 2, 3)).float().to(self.device)
        w[:] = torch.tensor([[1,0,0],[0,1,0]]).float().to(self.device)
        #[0] = torch.tensor([[0.9, 0.1, 0.1],[0.1, 0.9, 0.1]]).to(self.device)
        return w
        
    def weights_initialize(layer):
        if type(layer) == nn.Conv2d:
            torch.nn.init.uniform(layer.weight, -0.1, 0.1)
        if type(layer) == nn.LSTM:
            layer.weight_hh_l0.data.uniform_(-0.01, 0.01)
        if type(layer) == nn.Linear:
            layer.weight.data.normal_(0.0, 0.001)
            
    def prob_layer(self, logits):
        return nn.functional.softmax(logits)

    def enable_running_stats(layer):
        if type(layer) == nn.BatchNorm1d or type(layer) == nn.BatchNorm2d:
            layer.track_running_stats = True

    def disable_running_stats(layer):
        if type(layer) == nn.BatchNorm1d or type(layer) == nn.BatchNorm2d:
            layer.track_running_stats = False
























