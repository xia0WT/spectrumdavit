import torch
from math import sqrt
from mace.calculators import MACECalculator
from ase.filters import FrechetCellFilter,ExpCellFilter
from ase.constraints import FixSymmetry
from ase.optimize import BFGS, QuasiNewton
import timeout_decorator

TIME_OUT = 120
class MaceOpt(object):
    def __init__(self,
                 model_paths :str ,
                 local_rank = 0,
                ):
        device = 'cuda:%d' % local_rank if torch.cuda.is_available() else 'cpu'
        self.calculator = MACECalculator(model_paths=model_paths,
                                         device=device,
                                         dtype = 'float64')

    @timeout_decorator.timeout(TIME_OUT)
    def cell_opt(self,
                 init_crystal, 
                 fmax,
                 max_steps,
                 fix_symmetry :bool,
                ):
        crystal_sym = init_crystal.copy()
        crystal_sym.calc = self.calculator
        if fix_symmetry:
            try:
                crystal_sym.set_constraint(FixSymmetry(crystal_sym))
            except:
                crystal_sym.set_constraint()
        else:
            crystal_sym.set_constraint()
        #print(f"starting {crystal_sym.get_chemical_formula()}")
        ucf = FrechetCellFilter(crystal_sym)
        dyn = BFGS(ucf, logfile = None)
        #dyn = QuasiNewton(ucf, logfile = None)
        dyn.run(fmax=fmax , steps = max_steps)
        
        crystal_sym.set_constraint()
        
        forces = crystal_sym.get_forces()
        force_max = sqrt((forces ** 2).sum(axis=1).max())

        #print(f"Finish {crystal_sym.get_chemical_formula()} geometry optimization in {dyn.nsteps} steps with minimized force {force_max:.6f}")
        
        return crystal_sym, force_max, dyn.nsteps

    def eval_energy(self,
                    init_crystal,
                   ):
        crystal_sym = init_crystal.copy()
        crystal_sym.calc = self.calculator
        
        return crystal_sym.get_potential_energy()
