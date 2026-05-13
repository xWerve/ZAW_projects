import pandas as pd


def load_detections(det_path, conf_threshold=0.4):
    """
    Wczytuje detekcje z pliku MOT-format det.txt.
    Format: frame, id, x, y, w, h, conf, -1, -1, -1
    Zwraca DataFrame z kolumnami: frame, x, y, x2, y2, conf
    """
    df = pd.read_csv(det_path, header=None,
                     names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'a', 'b', 'c'])

    df = df[df['conf'] >= conf_threshold].copy()

    df['x2'] = df['x'] + df['w']
    df['y2'] = df['y'] + df['h']

    return df[['frame', 'x', 'y', 'x2', 'y2', 'conf']].reset_index(drop=True)