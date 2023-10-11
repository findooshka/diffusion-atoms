from datasets.symmetries import reduce_atoms, get_operations
import json
import torch
import numpy
import os
from tqdm import tqdm
from jarvis.core.atoms import Atoms

with open("data/sg_index_mp20.json", 'r') as f:
    sg_index = json.load(f)
folder = "/home/arsen/data/mp20_test_primitive/"
files = os.listdir(folder)
space_groups = "data/space_groups/"
result = {}
for file in tqdm(os.listdir(folder)):
    atoms = Atoms.from_cif(os.path.join(folder, file), use_cif2cell=False, get_primitive_atoms=False)
    operations, space_group, sg_type = get_operations(os.path.join(space_groups, str(sg_index[file])))
    if space_group <= 15:
        continue
    reduced_atoms, _ = reduce_atoms(atoms, operations, check_consistency=True)
    result[file] = {'elements': list(reduced_atoms.elements), 'sg': space_group}
    #reduced_atoms = p_to_c(reduced_atoms, sg_type, get_lattice_system(space_group))
with open('element_counts_mp20_test.json', 'w') as f:
    f.write(json.dumps(result))