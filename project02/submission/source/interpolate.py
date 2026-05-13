

import numpy as np
import pandas as pd
import os


def interpolate_tracks(results_path, output_path, max_gap=20):

    if os.path.getsize(results_path) == 0:
        open(output_path, 'w').close()
        return

    df = pd.read_csv(results_path, header=None)
    df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'a', 'b', 'c']

    interpolated_rows = []

    for track_id in df['id'].unique():
        track = df[df['id'] == track_id].sort_values('frame').reset_index(drop=True)

        for _, row in track.iterrows():
            interpolated_rows.append(row.values)

        frames = track['frame'].values
        for i in range(len(frames) - 1):
            f_start = frames[i]
            f_end = frames[i + 1]
            gap = f_end - f_start

            if 1 < gap <= max_gap:
                row_start = track.iloc[i]
                row_end = track.iloc[i + 1]

                for f in range(f_start + 1, f_end):
                    alpha = (f - f_start) / gap  # 0..1
                    interp = [
                        f,
                        track_id,
                        row_start['x'] + alpha * (row_end['x'] - row_start['x']),
                        row_start['y'] + alpha * (row_end['y'] - row_start['y']),
                        row_start['w'] + alpha * (row_end['w'] - row_start['w']),
                        row_start['h'] + alpha * (row_end['h'] - row_start['h']),
                        min(row_start['conf'], row_end['conf']),
                        -1, -1, -1
                    ]
                    interpolated_rows.append(interp)

    result = pd.DataFrame(interpolated_rows, columns=df.columns)
    result = result.sort_values(['frame', 'id']).reset_index(drop=True)

    result = result.drop_duplicates(subset=['frame', 'id'])

    np.savetxt(output_path, result.values,
               fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')


def interpolate_all(results_root, output_root, max_gap=20):
    os.makedirs(output_root, exist_ok=True)
    for fname in os.listdir(results_root):
        if not fname.endswith('.txt'):
            continue
        src = os.path.join(results_root, fname)
        dst = os.path.join(output_root, fname)
        print(f"Interpolating {fname}...")
        interpolate_tracks(src, dst, max_gap=max_gap)
    print("Done.")


if __name__ == "__main__":
    interpolate_all('results', 'results_interp', max_gap=20)
