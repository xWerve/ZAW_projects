import pandas as pd
import numpy as np
import os

data_root = 'data/evs_mot-train'
results_root = 'results'

for seq in sorted(os.listdir(data_root)):
    res_path = os.path.join(results_root, f'{seq}.txt')
    gt_path = os.path.join(data_root, seq, 'gt/gt.txt')
    if not os.path.exists(res_path) or not os.path.exists(gt_path):
        continue

    res = pd.read_csv(res_path, header=None).iloc[:, :6]
    res.columns = ['frame', 'id', 'x', 'y', 'w', 'h']

    gt_raw = pd.read_csv(gt_path, header=None)
    gt_raw.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'active', 'class', 'visibility'] + \
                     [f'e{i}' for i in range(gt_raw.shape[1] - 9)]
    gt = gt_raw[(gt_raw['active'] == 1) & (gt_raw['class'] == 1) & (gt_raw['visibility'] >= 0.1)]

    tl = res.groupby('id').size()

    gaps = []
    for tid, grp in res.groupby('id'):
        frames = sorted(grp['frame'].values)
        for i in range(len(frames) - 1):
            g = frames[i + 1] - frames[i]
            if g > 1:
                gaps.append(g)

    print(f"\n=== {seq} ===")
    print(f"Unique track IDs: {res['id'].nunique()} | GT IDs: {gt['id'].nunique()}")
    print(f"Track len: mean={tl.mean():.1f} median={tl.median():.1f} min={tl.min()} max={tl.max()}")
    print(f"Short tracks (<5 frames): {(tl < 5).sum()}")
    print(f"Track gaps: count={len(gaps)} mean={np.mean(gaps):.1f} max={max(gaps) if gaps else 0}")
    print(f"Frames per GT obj: {len(gt) / gt['id'].nunique():.1f} vs frames per track: {len(res) / res['id'].nunique():.1f}")
