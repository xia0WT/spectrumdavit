import torch
import numpy as np

from scipy.signal import savgol_filter
from scipy import interpolate
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
from functools import reduce
import argparse
import yaml
import pickle
import os
import re

from tslearn.metrics import dtw_path
from pybaselines import Baseline

from nwtk import lortzen
from .utils import read_txt_xrd
from .model.spectrumdavit_model import SpectrumDaVit
from .find_peak import FindPeaks
from .structure_generator import MaceGenerator
from nwtk.utils import create_log


class XrdPredict(object):
    def __init__(self,
                 model_config, #"../xrd_predict_unique/20250723_162449_SpectrumDaVit/args_config.yaml",  #yaml
                 model_save_dir, #"../xrd_predict_unique/20250723_162449_SpectrumDaVit/model_best.pth.tar",  #pth.tar,
                 xrd_type_file = "template/pxrd_unique_10_60_1256_remove_duplicate_label.pkl",
                 structure_frame = "template/structure_frame_unique_10_60_1256_remove_duplicate_fingerprint_electronegativity.pkl",
                 mace_model_paths = "../data/mace-omat-0-medium.model",
                 local_rank = 2,
                 predict_topk = 3,
                 atoms_threshold = 50,
                ):

        self.device = torch.device('cuda:%d' % local_rank) if torch.cuda.is_available() else torch.device('cpu')
        parser = argparse.ArgumentParser(description='Training Config', add_help=False)
        with open(model_config) as f:
            default_arg = yaml.safe_load(f)
        parser.set_defaults(**default_arg)
        args = parser.parse_args(args = [])

        self.model = self.model_eval(args, model_save_dir, self.device)

        self.mace_model_paths = mace_model_paths
        self.profile = lortzen.simple_profile(nmax=90, ndim=4096)

        self.atoms_threshold = atoms_threshold
        self.predict_topk = predict_topk
        self.calculator = XRDCalculator(wavelength='CuKa')
        
        self.ep_fingerprint = bool(re.search("electronegativity", structure_frame))   #use eletronegativity as fingerprint
        with open(structure_frame, "rb") as f:
            self.frame = pickle.load(f)


        with open(xrd_type_file, "rb") as f:
            self.structure_type = pickle.load(f)

    def model_eval(self, args, model_save_dir, map_location):
        model =SpectrumDaVit(
                            in_chans=args.in_chans,
                            num_classes=args.num_classes,
                            drop_rate=args.drop,
                            drop_path_rate=args.drop_path,
                            depths=args.depths,  #(1, 1, 3, 1),
                            embed_dims=args.embed_dims,  #(96, 192, 384, 768),
                            num_heads=args.num_heads,  #(3, 6, 12, 24),
                            window_size=args.window_size,
                            ).to(self.device)

        checkpoint = torch.load(model_save_dir, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        return model.float()


    def find_peaks(self,
                   xrd_file,
                   work_dir = "results",
                   prominence = 8,
                   window_length=60,
                   poly_order = 3,
                   deriv = 2,
                   use_sgd=False,
                   use_dtw=True,
                   **kwargs): # make sure to match as much peaks as you can, do not match error peaks.

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir

        self._logger = create_log(os.path.join(work_dir, 'result.log'))()

        if xrd_file.endswith(".cif"):
            s = Structure.from_file(xrd_file)
            pattern = self.calculator.get_pattern(s)
            self.xrd_data = np.linspace(0,90,4096), self.profile.pseudo_voigt(pattern.x, pattern.y).squeeze()
            raw_xrd = self.xrd_data
        else:
            raw_xrd = read_txt_xrd(xrd_file)
            xrd_data = interp(*raw_xrd)
            self.xrd_data = fit_baseline(*xrd_data, poly_order=poly_order)

        self.peak = FindPeaks(*self.xrd_data).find_peak(prominence=prominence, **kwargs)
        #TODO x,y?
        self.sgdector = SGDetector(*self.xrd_data, window_length=window_length, poly_order = poly_order, deriv = deriv) if use_sgd else None #TODO
        self.dtwdetector = DTWdetector(*raw_xrd, poly_order = poly_order) if use_dtw else None
        self.sgd_score = []
        self.dtw_score = []
        
    def get_structure(self, elements: list[str], **kwargs):
        mg = MaceGenerator(frame = self.frame, ep_fingerprint= self.ep_fingerprint, model_paths = self.mace_model_paths, logger = self._logger)
        crystal = []
        indices = self._predict(self.peak, self.model, self.predict_topk).tolist()
        if not isinstance(indices, list):
            indices = [indices]
        for j,i in enumerate(indices):
            self._logger.info(f"===========STARTING TOP {j} PREDICTION===========")
            s = self.structure_type[i]
            if s in self.frame:
                c, e = mg.from_elements(s, elements,atoms_threshold = self.atoms_threshold, **kwargs)
                crystal.append(c)
        crystal = reduce(lambda a, b: a + b, crystal)
        self.compare_xrd(crystal)
        sgdsc = sorted(self.sgd_score, key=lambda i:i[1],reverse = True)
        dtwsc = sorted(self.dtw_score, key=lambda i:i[1])

        self._summary(sgdsc, dtwsc)

    def _summary(self, sgdsc, dtwsc):
        self._logger.info(f"==============GROUND TRUTH SUMMARY==============")
        for i,j in zip(sgdsc[:3],("1st", "2nd", "3rd")):
            self._logger.info(f"The {j} similar structure index is {i[0]}, score is {i[1]}")

        for i,j in zip(dtwsc[:3],("1st", "2nd", "3rd")):
            self._logger.info(f"The {j} similar structure index according DTW is {i[0]},  score is {i[1]:.2f}, similarity s {i[2]:2f} %")

    @torch.no_grad()
    def _predict(self, peaks, model, top_k):
        profiled_xrd = self.profile.pseudo_voigt(*self.peak)
        pr = torch.tensor(profiled_xrd.squeeze()[np.newaxis, np.newaxis], dtype = torch.float32).to(self.device)
        output = model(pr)
        k = torch.topk(output, top_k).indices.squeeze()
        return k.detach().to("cpu")

    def compare_xrd(self, crystals):
        for i, c in enumerate(crystals):
            strcuture = AseAtomsAdaptor.get_structure(c)
            self.plot_vs_xrd(strcuture, i, self.work_dir)

    def plot_vs_xrd(self, structure, index, pic_save_dir):
        
        pattern = self.calculator.get_pattern(structure)
        xrd_x , xrd_y = self.xrd_data
        formula = structure.formula.replace(" ","")
        fig, ax = plt.subplots()
        ax.plot(xrd_x, xrd_y*100/ max(xrd_y), color = "r", label="XRD")
        ax.vlines(pattern.x, 0, pattern.y, color = "b", label="characterise peaks")
        ax.set_ylim(0)
        ax.set_ylabel("intensity")
        ax.set_xlabel("2θ")
        ax.grid()
        ax.legend()
        ax.set_title(f"Exp XRD vs Theory {formula}--{index}")
        
        if self.dtwdetector:
            profiled = self.profile.pseudo_voigt(pattern.x, pattern.y)
            _dtw_score = self.dtwdetector(profiled)
            _similarity = self._similarity_calculator(self.peak[0].shape[0], _dtw_score)
            
            self.dtw_score.append((index, _dtw_score, _similarity *100))
            self._logger.info(f"Structure index: {index}, formula: {formula}, DTW score: {_dtw_score}")
        if self.sgdector:
            _sgd_score = self.sgdector(pattern)
            self.sgd_score.append((index, _sgd_score))
            self._logger.info(f"Structure index: {index}, formula: {formula}, similarity score: {_sgd_score}")
        if not os.path.exists(pic_save_dir):
            os.mkdir(pic_save_dir)
        #plt.close(fig)
        fig.savefig(os.path.join(pic_save_dir, f"{formula}--{index}.png"), dpi = 300)
        structure.to(os.path.join(pic_save_dir, f"{formula}--{index}.cif"), fmt = "cif")
        
    def _similarity_calculator(self, peak_num, cost):
        gamma = - ( peak_num )/ np.log(0.9) # 0.9
        return np.exp( - cost ** 2 / ( 200 * gamma ) )  #exponential similarity, 200 is standard c-DTW score


class SGDetector(object):
    def __init__(self, x, y, window_length, poly_order, deriv):
        self.f = savgol_filter(y,
                        window_length = window_length,
                        polyorder=poly_order,
                        deriv = deriv)
        self.x = x

    def __call__(self, pattern):
        index = (pattern.x - self.x.min()) / (self.x.max() - self.x.min()) * len(self.x)
        c = np.select([index<len(self.x)],[index]).astype(int)
        c = c[c != 0]
        t = abs(self.f[c])
        k = np.sum(t)
        return k

class DTWdetector(object):
    def __init__(self, x, y, poly_order):
        interp_xrd = interp(x, y, extrapolate=True) # fit_baseline(x, y, poly_order=poly_order)
        _, self.xrd_y = fit_baseline(*interp_xrd, poly_order = poly_order)

    def __call__(self, y, sakoe_chiba_radius=50):
        _, cost = dtw_path(self.xrd_y, y, sakoe_chiba_radius=sakoe_chiba_radius)
        return cost


def interp(x, y, extrapolate=False, kind = "nearest-up"):
    y = y / y.max() *100
    fill_value = "extrapolate" if extrapolate else None
    f = interpolate.interp1d(x, y, kind = "nearest-up", fill_value=fill_value)
    if extrapolate:
        x = np.linspace(0, 90, 4096)
    else:
        x = np.linspace(x.min(), x.max(), 4096)
    y = f(x)
    return x, y

def fit_baseline(x, y, poly_order):
    baseline_fitter = Baseline(x_data=x)
    bkg_1, params_1 = baseline_fitter.modpoly(y, poly_order=poly_order)
    return x, y-bkg_1
