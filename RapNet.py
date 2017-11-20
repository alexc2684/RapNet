import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.autograd as autograd

import os
import numpy as np
import random
import re


class RapNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, vocab_size, num_classes):
        super(RapNet, self).__init__()
        self.cuda = torch.cuda.is_available()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)  # 2 for bidirection

    def initHidden(self, x):
        hidden = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))#.cuda() # 2 for bidirection
        if self.cuda:
            hidden = hidden.cuda()
        return hidden

    def forward(self, x):
        # Set initial states
        x = self.embeddings(x).view(len(x), 1, -1)
        h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))#.cuda() # 2 for bidirection
        c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))#.cuda()
        if self.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        # print(out[len(out)-1,0])
        return out[len(out)-1,0]
