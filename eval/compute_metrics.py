from collections import Counter
import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map    
from scipy.stats import wasserstein_distance

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from jarvis.core.atoms import Atoms

from eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist, compute_cov
)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

Percentiles = {
    'mp20': np.array([-3.17562208, -2.82196882, -2.52814761]),
    'carbon': np.array([-154.527093, -154.45865733, -154.44206825]),
    'perovskite': np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {'struc': 0.4, 'comp': 10.}


class Crystal(object):
    def __init__(self, path):
        #print(path)
        try:
            # loading with cifparser gets stuck for some of the CDVAE generated structures
            #parser = CifParser(path, occupancy_tolerance=0, site_tolerance=0, frac_tolerance=0)
            struct = Atoms.from_cif(path, use_cif2cell=False, get_primitive_atoms=False)
            struct = Structure(lattice=struct.lattice_mat,
                               species=struct.atomic_numbers,
                               coords=struct.frac_coords)
            self.name = path
            self.aborted = False
            self.frac_coords = struct.frac_coords
            self.atom_types = struct.atomic_numbers
            self.lengths = struct.lattice.lengths
            self.angles = struct.lattice.angles

            self.structure = struct
            self.get_composition()
            self.get_validity()
            self.get_fingerprints()
        except:
            self.aborted = True
            self.valid = False
            return
        
        #print(4)
        
   
    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        #self.comp_valid = smact_validity(self.elems, self.comps)
        self.struct_valid = structure_validity(self.structure)
        #self.valid = self.comp_valid and self.struct_valid
        self.valid = self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=986, eval_model_name=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        #comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {#'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {'comp_div': comp_div}

    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                                       c.dict for c in self.gt_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs
        cov_metrics_dict, combined_dist_dict, index = compute_cov(
            self.valid_samples, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        #for i in range(len(combined_dist_dict['struc_precision_dist'])):
        #    if combined_dist_dict['struc_precision_dist'][i] > .4:# or combined_dist_dict['struc_precision_dist'][i] > .4:
        #        print(f"{self.valid_samples[index[i]].name} {combined_dist_dict['struc_recall_dist'][i]}")# combined_dist_dict['struc_precision_dist'][i], combined_dist_dict['comp_recall_dist'][i], combined_dist_dict['comp_precision_dist'][i]}")
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics.update(self.get_prop_wdist())
        metrics.update(self.get_coverage())
        return metrics

def load_from_dir(path, n=-1):
    files = os.listdir(path)
    files = [file for file in files if len(file) > 4 and file[-4:] == '.cif']
    files = files[:n] if n > 0 else files#[1040:]
    result = p_map(lambda x: Crystal(os.path.join(path, x)), files)#, num_cpus=1)
    removed_none = []
    #for file in tqdm(os.listdir(path)[:200]):
    #    struct = Crystal(os.path.join(path, file))
    for struct in result:
        if not struct.aborted:
            removed_none.append(struct)
    return removed_none

def main(args):
    gen_crys = load_from_dir(args.gen)  #"exp/PGCGM")
    gt_crys = load_from_dir(args.true)  #"/home/arsen/data/mp20_test_primitive")

    gen_evaluator = GenEval(gen_crys, gt_crys)
    gen_metrics = gen_evaluator.get_metrics()
    

    print(gen_metrics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen', required=True)
    parser.add_argument('--true', required=True)
    #parser.add_argument('--label', default='')
    #parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt'])
    args = parser.parse_args()
    main(args)
