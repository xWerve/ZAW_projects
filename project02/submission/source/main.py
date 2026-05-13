import os
import numpy as np
from src.tracker import Tracker
from src.track import Track
from utils.data_loader import load_detections
from interpolate import interpolate_tracks

TRACKER_PARAMS = dict(
    max_age=40,
    min_hits=1,
    iou_threshold=0.3,
    high_det_thresh=0.4,
    low_det_thresh=0.1,
)
CONF_THRESHOLD = 0.05


def process_sequence(seq_path, output_path):
    det_path = os.path.join(seq_path, 'det/det.txt')
    df_dets = load_detections(det_path, conf_threshold=CONF_THRESHOLD)

    Track.count = 0
    tracker = Tracker(**TRACKER_PARAMS)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    for frame_idx in sorted(df_dets['frame'].unique()):
        frame_data = df_dets[df_dets['frame'] == frame_idx]
        dets = frame_data[['x', 'y', 'x2', 'y2', 'conf']].values

        online_targets = tracker.update(dets)

        for t in online_targets:
            tid, x1, y1, x2, y2, conf = t
            results.append([int(frame_idx), int(tid),
                            x1, y1, x2 - x1, y2 - y1,
                            conf, -1, -1, -1])

    if results:
        np.savetxt(output_path, np.array(results),
                   fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
    else:
        open(output_path, 'w').close()


def run(data_path, output_dir):
    temp_dir = output_dir + "_temp"
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    sequences = sorted([s for s in os.listdir(data_path)
                        if os.path.isdir(os.path.join(data_path, s))])

    for seq in sequences:
        print(f"Processing {seq}...")
        temp_out = os.path.join(temp_dir, f'{seq}.txt')
        final_out = os.path.join(output_dir, f'{seq}.txt')

        process_sequence(os.path.join(data_path, seq), temp_out)
        interpolate_tracks(temp_out, final_out, max_gap=30)


if __name__ == "__main__":
    run('../data/evs_mot-test', 'results')