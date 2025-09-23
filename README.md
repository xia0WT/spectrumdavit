## Requirements
```bash
pip install -r requirements.txt
```

## Checkpoint
The pre-trained SpectrumDavit model checkpoint is available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17172997.svg)](https://doi.org/10.5281/zenodo.17172997)

## Data Download
The dataset used in this project is available on [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17173064.svg)](https://doi.org/10.5281/zenodo.17173064)

The MACE pre-train model is available on [medium-omat-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model)

## Usage

To recover CdAs2O6, PDF number:00-053-0363

```python
import os
import torch
from workflow.xrdapi import XrdPredict

xrdp = XrdPredict(
                model_config = "path_to_model/args_config.yaml",              # model config 
                model_save_dir = "path_to_model/model_best.pth.tar",          # model checkpoint
                mace_model_paths = "path_to_mace/mace-omat-0-medium.model",   # mace pre-trained model
                predict_topk=3,
                local_rank = 3,
)

xrd_file = "test/xrd_icdd_PDF/00-053-0363 (PD3).csv"  #raw xrd file path
pdf = os.path.basename(xrd_file).split(" ")[0]
work_path = f"result/{pdf}"

xrdp.find_peaks(xrd_file=xrd_file,work_dir = work_path, prominence=2, distance = 5, height = 0 ,save_dir = work_path)

elements = ["Cd", "As", "O"]   #element information

xrdp.get_structure(elements, fmax =0.001, max_steps=100)

torch.cuda.empty_cache()
```

## Train
```bash
python train_davitcef.py -c config-classify.yaml
```
