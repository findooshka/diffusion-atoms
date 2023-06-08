from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
import os
import logging
from argparse import ArgumentParser

def read_directory(path):
    result = []
    names = []
    for file in os.listdir(path):
        if file[-4:] == ".cif":
            file_path = os.path.join(path, file)
            try:
                result.append(CifParser(file_path).get_structures()[0])
                names.append(file[:-4])
            except:
                logging.error(f"Error while reading \"{file_path}\"")
    return result, names

def remove_redundant(structs, names):
    result = []
    unique_names = []
    for i, structure in enumerate(structs):
        for added_structure in result:
            if StructureMatcher(ltol=.2, stol=.3).fit(structure, added_structure):
                break
        else:
            result.append(structure)
            unique_names.append(names[i])
    return result, unique_names

def save_to_dir(path, structs, vasp=False):
    structs, names = structs
    if not os.path.exists(path):
        os.makedirs(path)
    for i, struct in enumerate(structs):
        if vasp:
            Poscar(struct).write_file(os.path.join(path, names[i] + ".vasp"))
        else:
            CifWriter(struct).write_file(os.path.join(path, names[i] + ".cif"))
            
def process_batch(batch_dir, vasp):
    batch_dir = batch_dir.rstrip(r'\/')
    output_dir = batch_dir + "_uniques"
    logging.info(f"Writting files to \"{output_dir}\"")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for sg_dir in os.listdir(batch_dir):
        path = os.path.join(batch_dir, sg_dir)
        if os.path.exists(os.path.join(path, "finals")):
            path = os.path.join(path, "finals")
        structures, names = read_directory(path)
        out_sg_dir_path = os.path.join(output_dir, sg_dir)
        if not os.path.exists(out_sg_dir_path):
             os.makedirs(out_sg_dir_path)
        save_to_dir(out_sg_dir_path, remove_redundant(structures, names), vasp=vasp)

        
if __name__ == "__main__":
    logging.basicConfig(level=20)
    argparser = ArgumentParser()
    argparser.add_argument("--dir", type=str, required=True, help="Generation batch folder name")
    argparser.add_argument("--vasp", type=bool, default=False, help="Output in vasp format")
    args = argparser.parse_args()
    process_batch(args.dir, vasp=args.vasp)
