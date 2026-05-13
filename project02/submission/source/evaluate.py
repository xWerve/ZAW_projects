import numpy as np
import pandas as pd
import os
import motmetrics as mm

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)


def calculate_iou_distance(bboxes1, bboxes2, threshold=0.5):
    if len(bboxes1) == 0 or len(bboxes2) == 0:
        return np.empty((len(bboxes1), len(bboxes2)))

    dist_matrix = np.zeros((len(bboxes1), len(bboxes2)), dtype=np.float64)

    for i, b1 in enumerate(bboxes1):
        for j, b2 in enumerate(bboxes2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2

            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)

            iw = max(0, xi2 - xi1)
            ih = max(0, yi2 - yi1)
            area_i = iw * ih

            area1 = w1 * h1
            area2 = w2 * h2
            area_u = area1 + area2 - area_i

            iou = area_i / area_u if area_u > 0 else 0

            if iou >= (1 - threshold):
                dist_matrix[i, j] = 1.0 - iou
            else:
                dist_matrix[i, j] = np.nan

    return dist_matrix


def load_gt(gt_path, classes_to_keep=(1,), min_visibility=0.0):
    df = pd.read_csv(gt_path, header=None)
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'active', 'class', 'visibility'] + \
                 [f'extra_{i}' for i in range(df.shape[1] - 9)] if df.shape[1] >= 9 else \
                 ['frame', 'id', 'x', 'y', 'w', 'h'] + [f'extra_{i}' for i in range(df.shape[1] - 6)]

    if 'active' in df.columns:
        df = df[df['active'] == 1]

    if 'class' in df.columns and classes_to_keep is not None:
        df = df[df['class'].isin(classes_to_keep)]

    if 'visibility' in df.columns and min_visibility > 0:
        df = df[df['visibility'] >= min_visibility]

    return df[['frame', 'id', 'x', 'y', 'w', 'h']].reset_index(drop=True)


def evaluate_results(data_root, results_root):
    acc = mm.MOTAccumulator(auto_id=True)
    sequences = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    for seq in sorted(sequences):
        gt_path = os.path.join(data_root, seq, 'gt', 'gt.txt')
        res_path = os.path.join(results_root, f'{seq}.txt')

        if not os.path.exists(gt_path) or not os.path.exists(res_path):
            continue

        print(f"Ewaluacja: {seq}...")

        gt = load_gt(gt_path, classes_to_keep=(1,), min_visibility=0.1)

        res = pd.read_csv(res_path, header=None).iloc[:, :6]
        res.columns = ['frame', 'id', 'x', 'y', 'w', 'h']

        for frame in sorted(gt['frame'].unique()):
            g = gt[gt['frame'] == frame]
            r = res[res['frame'] == frame]

            dist_matrix = calculate_iou_distance(g[['x', 'y', 'w', 'h']].values,
                                                 r[['x', 'y', 'w', 'h']].values)

            acc.update(g['id'].values, r['id'].values, dist_matrix)

    mh = mm.metrics.create()
    metrics_to_compute = ['mota', 'motp', 'num_switches', 'precision', 'recall']
    summary = mh.compute(acc, metrics=metrics_to_compute, name='Mój Tracker')

    print("\n--- WYNIKI KOŃCOWE ---")
    print(mm.io.render_summary(summary,
                               formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    evaluate_results('data/evs_mot-train', results_dir)