import msgpack
import re
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from pymatgen.core import Structure
from mace_opt import MaceOpt
from itertools import permutations

import numpy as np
from pymatgen.core.composition import Element
from scipy.spatial import distance

class StructureGenerator(object):
    def __init__(self,
                 frame,
                ):
        self.frame = frame

    def from_elements(self, 
                      structure_type :str,
                      elements :list,
                      atoms_threshold = None,
                     ):
        element_num = len(elements)
        f = np.array(sorted([Element(el).atomic_radius for el in elements], reverse= True))
        fingerprint = np.r_[f, np.var(f, keepdims = True),  np.mean(f, keepdims = True)]
        sub_types = self.frame.get(structure_type).keys()
        sub_types_keys = [ k for k in sub_types if len(re.sub(r'[0-9]+', '', k)) == element_num ]
        crystals = []
        for sub_types_key in sub_types_keys:
            sub_frame = self.frame[structure_type][sub_types_key]
            type_indice = np.argmin([distance.cosine(k.get("fingerprint"), fingerprint) for k in sub_frame])
            strcuture_elements, wyckoff_symbols, wyckoff_sites, spacegroup, matrix, numion, _ = sub_frame[type_indice].values()
            _symbols = self._sub_site(elements, strcuture_elements)
            lattice = Lattice.from_matrix(matrix)

            structure = pyxtal()
            _crystal = []

            symbol_site = zip(wyckoff_symbols, wyckoff_sites)
            sites = [[{sm:si}] for sm, si in symbol_site]
            for _symbol in _symbols:
                #structure = pyxtal()
                structure.build(group = spacegroup,
                               species = _symbol,
                               numIons = numion,
                               sites = sites,         #[[{l : sites[l]}] for l in sites.keys()]
                               lattice = lattice)
                _crystal.append(structure.to_ase())
                if atoms_threshold:
                    _crystal = [c for c in _crystal if c.get_global_number_of_atoms() <= atoms_threshold]
            #_crystal = [ c for c in _crystal if (Composition(c.get_chemical_formula()).anonymized_formula == sub_types_key)]
            crystals += _crystal

        return crystals
    def _sub_site(self,
                  elements,
                  structure_elements,
                 ):
        
        r = set(structure_elements)
        s = set(elements)
        replacements = [dict(zip(r, p)) for p in permutations(s)]
        res = [[i.get(char, char) for char in structure_elements] for i in replacements]
        
        return res

    
class MaceGenerator(StructureGenerator):
    def __init__(self, frame, model_paths,logger = None):
        super().__init__(frame)
        self.mopt = MaceOpt(model_paths = model_paths)
        if logger:
            self._logger = logger.info
        else:
            self._logger = print
    def from_elements(self, 
                      structure_type :str,
                      elements :list,
                      use_mace_predict: bool = True,
                      fmax :float = 0.0001,
                      max_steps :int = 100,
                      ignore_not_converge: bool = True,
                      fix_symmetry: bool =False,
                      atoms_threshold = None,

                     ):

        structures = []
        max_forces = []
        
        crystals = StructureGenerator.from_elements(self,
                                                    structure_type = structure_type,
                                                    elements = elements,
                                                    atoms_threshold = atoms_threshold,
                                                   )

        if use_mace_predict:
            crystals = self._mace_predict(crystals)

        for c in crystals:
            try:
                formula = c.get_chemical_formula()
                self._logger(f"Starting {formula}")
                structure, max_force, steps = self.mopt.cell_opt(c,
                                                        fmax = fmax,
                                                        max_steps = max_steps,
                                                        fix_symmetry = fix_symmetry)
                
                self._logger(f"Finish {formula} geometry optimization in {steps} steps with minimized force {max_force:.6f}")
            except Exception:
                self._logger(f"{formula} geometry optimization time out")
                continue
            
            if not ignore_not_converge or (max_force <= fmax):
                structures.append(structure)
                max_forces.append(max_force)

        return structures, max_forces
    def _mace_predict(self,
                      crystals :list,
                     ):
        energy = []
        for crystal in crystals:
            predict_e = self.mopt.eval_energy(crystal)
            energy.append(predict_e)
            
        return [x for _, x in sorted(zip(energy, crystals), key=lambda s: s[0])]
