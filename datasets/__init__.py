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


class AtomsData(Dataset):
    def __init__(self, folder, sg_folder, sg_index):
        self.folder = folder
        self.sg_folder = sg_folder
        with open(sg_index, 'r') as f:
            self.sg_index = json.load(f)
        self.files = os.listdir(folder)
        self.size = len(self.files)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        logging.info(self.files[idx])
        #try:
        atoms = Atoms.from_cif(os.path.join(self.folder, self.files[idx]), use_cif2cell=False, get_primitive_atoms=False)
        operations, space_group, sg_type = get_operations(os.path.join(self.sg_folder, str(self.sg_index[self.files[idx]])))
        logging.info(f"atoms {len(atoms.elements)}, operations {len(operations)}")
        reduced_atoms, _ = reduce_atoms(atoms, operations, check_consistency=True)
        reduced_atoms = p_to_c(reduced_atoms, sg_type, get_lattice_system(space_group))
        #except Exception as exp:
        #    logging.warning(os.path.join(self.folder, self.files[idx]))
        #    logging.warning(exp)
        return {'atoms': reduced_atoms, 'operations': operations, 'space_group': space_group, 'sg_type': sg_type, 'name': self.files[idx]}


def get_dataset(args, config):
    return (AtomsData(config.data.data, config.data.space_groups, config.data.sg_index),
           AtomsData(config.data.data_test, config.data.space_groups, config.data.sg_index))
