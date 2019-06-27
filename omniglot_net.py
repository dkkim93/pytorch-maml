import torch
import torch.nn as nn
from layers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OmniglotNet(nn.Module):
    '''The base model for few-shot learning on Omniglot'''
    def __init__(self, loss_fn, args):
        super(OmniglotNet, self).__init__()
        self.loss_fn = loss_fn
        self.args = args

        # Define network
        self.add_module('fc1', nn.Linear(1, 64))
        self.add_module('fc2', nn.Linear(64, 64))
        self.add_module('fc3', nn.Linear(64, 1))

        self._init_weights()

    def forward(self, x, weights=None):
        ''' Define what happens to data in the net '''
        if weights is None:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
        else:
            x = relu(linear(x, weights['fc1.weight'], weights['fc1.bias']))
            x = relu(linear(x, weights['fc2.weight'], weights['fc2.bias']))
            x = torch.sigmoid(linear(x, weights['fc3.weight'], weights['fc3.bias']))
        return x

    def net_forward(self, x, weights=None):
        return self.forward(x, weights)
    
    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(self.args.seed)
        if device == torch.device("cuda"):
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())
    
    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
