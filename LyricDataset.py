import os
import random
import re
from torch.utils.data import DataLoader, Dataset

def openFile(path):
    f = open(path, "r")
    return f.read()

def convertForDict(word):
    pattern = re.compile('[\W_]+')
    word = pattern.sub('', word)
    return word.lower()


class LyricDataset(Dataset):
    def __init__(self, pathToData, numClasses, should_invert=True):
        self.pathToData = pathToData
        self.data = [path for path in os.listdir(self.pathToData) if os.path.isdir(self.pathToData + "/" + path) and path != ".DS_Store"]
        self.should_invert = should_invert
        self.numClasses = numClasses

    def __getitem__(self, index):
        labels = self.data
        label = random.randint(0,self.numClasses-1)
        classPath = self.pathToData + "/" + labels[label]
        songs = [path for path in os.listdir(classPath) if not path.startswith(".")]
        index = random.randint(0,len(songs)-1)
        songPath = classPath + "/" + songs[index]
        data = [convertForDict(word) for word in openFile(songPath).split(" ")]
        return data, label

    def __len__(self):
        return sum([len(os.listdir(self.pathToData + "/" + p)) for p in os.listdir(self.pathToData) if p != ".DS_Store"])
