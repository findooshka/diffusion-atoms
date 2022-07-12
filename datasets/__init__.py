import os
import torch
from torch.utils.data import Dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
import numpy as np
from datasets.symmetries import reduce_atoms, get_operations
from functions.lattice import p_to_c, get_lattice_system
import logging


class AtomsData(Dataset):
    def __init__(self, folder, phonopy_folder):
        self.folder = folder
        self.phonopy_folder = phonopy_folder
        self.files = os.listdir(folder)
        self.size = len(self.files)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        logging.info(self.files[idx])
        #try:
        atoms = Atoms.from_cif(os.path.join(self.folder, self.files[idx]), use_cif2cell=False, get_primitive_atoms=False)
        operations, space_group, sg_type = get_operations(os.path.join(self.phonopy_folder, self.files[idx]))
        reduced_atoms, _ = reduce_atoms(atoms, operations, check_consistency=False)
        reduced_atoms = p_to_c(reduced_atoms, sg_type, get_lattice_system(space_group))
        #except Exception as exp:
        #    logging.warning(os.path.join(self.folder, self.files[idx]))
        #    logging.warning(exp)
        return {'atoms': reduced_atoms, 'operations': operations, 'space_group': space_group, 'sg_type': sg_type, 'name': self.files[idx]}


def get_dataset(args, config):
    return AtomsData(config.data.data, config.data.phonopy), AtomsData(config.data.data_test, config.data.phonopy)
