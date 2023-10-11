import os
import torch
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import numpy as np
from datasets.symmetries import reduce_atoms, get_operations
from functions.lattice import p_to_c, get_lattice_system
import logging
import json
from tqdm import tqdm


class AtomsData(Dataset):
    def __init__(self, folder, sg_folder, sg_index):
        self.folder = folder
        self.sg_folder = sg_folder
        with open(sg_index, 'r') as f:
            self.sg_index = json.load(f)
        self.atoms = []
        self.operations = []
        self.sg = []
        self.sg_type = []
        self.name = []
        logging.info(f"Loading dataset from {folder}")
        n = 0
        for file in tqdm(os.listdir(folder)):
            atoms = Atoms.from_cif(os.path.join(self.folder, file), use_cif2cell=False, get_primitive_atoms=False)
            operations, space_group, sg_type = get_operations(os.path.join(self.sg_folder, str(self.sg_index[file])))
            if space_group <= 15:
                continue
            self.operations.append(operations)
            self.sg.append(space_group)
            self.sg_type.append(sg_type)
            self.name.append(file)
            reduced_atoms, _ = reduce_atoms(atoms, operations, check_consistency=True)
            reduced_atoms = p_to_c(reduced_atoms, sg_type, get_lattice_system(space_group))
            self.atoms.append(reduced_atoms)
            n += 1
        self.size = len(self.atoms)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):  
        return {'atoms': self.atoms[idx], 'operations': self.operations[idx], 'space_group': self.sg[idx], 'sg_type': self.sg_type[idx], 'name': self.name[idx]}


def get_dataset(args, config):
    return (AtomsData(config.data.data, config.data.space_groups, config.data.sg_index),
           AtomsData(config.data.data_test, config.data.space_groups, config.data.sg_index))

def get_batch(data_iterator, batch_size):
    atoms, operations, space_group, sg_type = [], [], [], []
    for i in range(batch_size):
        data_instance = next(data_iterator, None)
        if data_instance != None:
            atoms.append(data_instance['atoms'])
            operations.append(data_instance['operations'])
            space_group.append(data_instance['space_group'])
            sg_type.append(data_instance['sg_type'])
    return atoms, operations, space_group, sg_type
