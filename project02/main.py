import os
import cv2
import numpy as np
from src.tracker import Tracker
from utils.data_loader import load_detections


def process_sequence(seq_path, output_path):
    det_path = os.path.join(seq_path, 'det/det.txt')
    df_dets = load_detections(det_path)
    tracker = Tracker()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    frames = df_dets['frame'].unique()
    for frame_idx in sorted(frames):
        frame_data = df_dets[df_dets['frame'] == frame_idx]
        dets = frame_data[['x', 'y', 'x2', 'y2', 'conf']].values

        online_targets = tracker.update(dets)

        for t in online_targets:
            tid, x1, y1, x2, y2, conf = t
            results.append([frame_idx, tid, x1, y1, x2 - x1, y2 - y1, conf, -1, -1, -1])

    np.savetxt(output_path, np.array(results), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')


if __name__ == "__main__":
    train_path = 'data/evs_mot-train'
    sequences = os.listdir(train_path)
    for seq in sequences:
        print(f"Processing {seq}...")
        process_sequence(os.path.join(train_path, seq), f'results/{seq}.txt')