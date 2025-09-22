import re
import numpy as np

def read_txt_xrd(file):
    if file.endswith(".txt"):
        with open(file, "r") as f:
            content = f.read().splitlines()
            content = [c for c in content if not re.findall("[A-Z]", c)]
        x = []
        y = []
        for d in content:
            dc = d.split(' ')
            dc = [c for c in dc if c]
            x.append(float(dc[0]))
            y.append(float(dc[1]))
        return np.array(x), np.array(y)

    if file.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(file, header=None)
        x = df[0].values
        y = df[1].values
        return x, y