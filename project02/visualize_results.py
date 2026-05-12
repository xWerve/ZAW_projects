import cv2
import pandas as pd
import os


def visualize(seq_name, data_root='data/evs_mot-train'):
    img_dir = os.path.join(data_root, seq_name, 'img1')
    res_path = f'results/{seq_name}.txt'

    results = pd.read_csv(res_path, header=None)
    results.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3d', 'y3d', 'z3d']

    for frame_idx in sorted(results['frame'].unique()):
        img_path = os.path.join(img_dir, f"{str(int(frame_idx)).zfill(6)}.jpg")
        img = cv2.imread(img_path)

        frame_data = results[results['frame'] == frame_idx]
        for _, row in frame_data.iterrows():
            x, y, w, h, obj_id = int(row['x']), int(row['y']), int(row['w']), int(row['h']), int(row['id'])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(img, f"ID: {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Tracking Result', img)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize('MOT_02')