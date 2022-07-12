import os
import torch
#import numbers
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import numpy as np


class AtomsData(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = os.listdir(folder)
        self.size = len(self.files)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        try:
            atoms = Atoms.from_cif(os.path.join(self.folder, self.files[idx]), use_cif2cell=False)
        except Exception as exp:
            print(os.path.join(self.folder, self.files[idx]))
            print(exp)
        #graph.ndata['coord'] = torch.from_numpy(atoms.cart_coords)
        return atoms


def get_dataset(args, config):
    return AtomsData(config.data.data), AtomsData(config.data.data_test)
