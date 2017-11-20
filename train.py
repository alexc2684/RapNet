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
import shutil
import sys

from LyricDataset import LyricDataset
from RapNet import RapNet

#helper functions
def openFile(path):
    f = open(path, "r")
    return f.read()

def imshow(img,text,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def convertForDict(word):
    pattern = re.compile('[\W_]+')
    word = pattern.sub('', word)
    return word.lower()

def prepare_sequence(seq, to_ix, isTest=False):
    if isTest:
        idxs = []
        for w in seq:
            try:
                idxs.append(to_ix[w])
            except KeyError:
                idxs.append(to_ix['_UNK_'])
    else:
        idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)#.cuda()

def initVocab(dataset):
    labels = dataset.data
    word_to_ix = {}
    word_to_ix["_UNK_"] = len(word_to_ix)
    for artist in labels:
        for song in os.listdir(dataset.pathToData + "/" + artist):
            if song != ".DS_Store":
                for word in openFile(dataset.pathToData + "/" + artist + "/" + song).split(" "):
                    word = convertForDict(word)
                    if word not in word_to_ix:
                        word_to_ix[word] = len(word_to_ix)
    return word_to_ix

def train(EDIM, HDIM, epochs):
    cuda = torch.cuda.is_available()
    counter = []
    loss_history = []
    avg_loss = []
    iteration = 0

    PATH = "train"
    numClasses = len([path for path in os.listdir(PATH) if path != ".DS_Store"])
    dataset = LyricDataset(PATH, numClasses)
    word_to_ix = initVocab(dataset)

    if cuda:
        model = RapNet(EDIM, HDIM, 2, len(word_to_ix), numClasses).cuda()
        loss = nn.NLLLoss().cuda()
    else:
        model = RapNet(EDIM, HDIM, 2, len(word_to_ix), numClasses)
        loss = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(epochs):
        for i, data in enumerate(dataset):
            song, label = data
            song = prepare_sequence(song, word_to_ix)
            label = Variable(torch.LongTensor([label]))  
          #label = Variable(torch.LongTensor([0 if i != label else 1 for i in range(numClasses)]))
            if cuda:
                song, label = song.cuda(), label.cuda()
            model.hidden = model.initHidden(song)
            out = model(song)
 #           label = label.view(1, -1)
 #           out = out.view(1, -1)
            print(out.size())
            optimizer.zero_grad()
            total_loss = loss(out, label)
            total_loss.backward()
            optimizer.step()
            if i % 10 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,total_loss.data[0]))
                iteration += 10
                counter.append(iteration)
                loss_history.append(total_loss.data[0])
                avg_loss.append((sum(loss_history))/len(loss_history))
            if i == 10000:
                save_checkpoint({
                   'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict()
                }, True, filename='saved_models/checkpoint'+ str(epoch) + '.pth.tar')
                f = open("loss/loss" + str(epoch) + ".txt", "w")
                [f.write(str(l)) for l in avg_loss]
                f.close()
                break

    print("Training Complete")
    print("Average Loss:", avg_loss[len(avg_loss)-1])

def main():
    args = sys.argv[1:]
    if len(args) == 3:
        EDIM = int(args[0])
        HDIM = int(args[1])
        epochs = int(args[2])
        train(EDIM, HDIM, epochs)

if __name__ == "__main__":
    main()
