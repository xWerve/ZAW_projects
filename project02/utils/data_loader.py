import pandas as pd
import numpy as np

def load_detections(path):
    df = pd.read_csv(path, header=None)
    df = df.iloc[:, :7]
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf']
    df['x2'] = df['x'] + df['w']
    df['y2'] = df['y'] + df['h']
    return df